"""Online adaptive selective-SQM evaluation on time-varying traces.

This script adds the proposed cross-packet rank controller without changing the
blockwise SQM geometry. Validation traces are used for: (i) selecting the best
fixed rank, (ii) SNR/class lookup, (iii) rank-specific operating-class oracle
labels, and (iv) deterministic controller thresholds. Final traces use disjoint
seeds. The adaptive receivers observe only decoder gap/residual statistics,
except S2 (explicit slow class lookup) and S7 (optional ACK-assisted UCB).
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, json
import numpy as np
import pandas as pd

from fractional_tdla_core import tdla_operators, exact_fractional_channel, random_tdla_coefficients
from run_final_headtohead import randA, apply_iq_dc_noise, precompute_dmm_bases

ROOT = Path(__file__).resolve().parents[1]
M, L = 256, 16
RANKS = [3, 4, 5, 6]
GRID_COUNT = 11
FULL_EVALS = M * GRID_COUNT
ETA = 1e-12


def make_trace(blocks_per_segment: int, seed: int, snr_trace, delay_trace):
    """Generate a piecewise-stationary trace; only the trace generator knows regimes."""
    rng = np.random.default_rng(seed)
    X = np.load(ROOT / 'Artifact/fractional_incidence_codebook.npz')['X_opt']
    Xd = X[:, 0] + 1j * X[:, 1]
    _, powers, D_nom = tdla_operators(L, 30.0)
    ys, idxs, snrs, delays, segs = [], [], [], [], []
    for seg, (snr, delay) in enumerate(zip(snr_trace, delay_trace)):
        n = int(blocks_per_segment)
        _, p_true, D_true = tdla_operators(L, float(delay))
        idx = rng.integers(M, size=n)
        h = random_tdla_coefficients(rng, n, p_true)
        omega = rng.uniform(-math.radians(4), math.radians(4), n)
        A = randA(rng, n, 1.5)
        ang = rng.uniform(0, 2 * math.pi, n)
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        N0 = 10 ** (-float(snr) / 10.0)
        noise = np.sqrt(N0 / 2) * rng.normal(size=(n, 2, L))
        s = exact_fractional_channel(Xd[idx], h, D_true)
        y = apply_iq_dc_noise(s, omega, A, d, noise)
        ys.append(y.astype(np.float32)); idxs.append(idx)
        snrs.extend([float(snr)] * n); delays.extend([float(delay)] * n); segs.extend([seg] * n)
    return (np.concatenate(ys), np.concatenate(idxs), np.asarray(snrs),
            np.asarray(delays), np.asarray(segs), Xd, powers, D_nom)


def rank_stats(Y: np.ndarray, bases: np.ndarray, chunk: int = 32):
    """Return argmax message, top-two gap, and normalized residual per packet."""
    Yc = Y - Y.mean(axis=2, keepdims=True)
    energy = np.sum(Yc * Yc, axis=(1, 2))
    n = len(Y)
    pred = np.empty(n, dtype=np.int32)
    gap = np.empty(n, dtype=np.float32)
    resid = np.empty(n, dtype=np.float32)
    top = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        z = np.einsum('ncl,mglr->nmgcr', Yc[s:e], bases, optimize=True)
        score = np.sum(z * z, axis=(3, 4))  # n x messages x CFO grid
        ms = score.max(axis=2)
        order = np.argpartition(ms, -2, axis=1)[:, -2:]
        vals = np.take_along_axis(ms, order, axis=1)
        hi = vals.max(axis=1); lo = vals.min(axis=1)
        pred[s:e] = np.argmax(ms, axis=1)
        top[s:e] = hi
        gap[s:e] = (hi - lo) / (hi + ETA)
        resid[s:e] = np.clip(1.0 - hi / (energy[s:e] + ETA), 0.0, 1.0)
    return pred, gap.astype(float), resid.astype(float), top.astype(float)


def all_rank_stats(Y, bases_by_rank):
    return {r: rank_stats(Y, bases_by_rank[r]) for r in RANKS}


def segment_metric(stats, idx, segments, snrs, delays):
    rows = []
    for r in RANKS:
        pred, gap, resid, _ = stats[r]
        for seg in np.unique(segments):
            mask = segments == seg
            rows.append({'rank': r, 'segment': int(seg), 'snr_db': float(snrs[mask][0]),
                         'delay_ns': float(delays[mask][0]), 'PER': float(np.mean(pred[mask] != idx[mask])),
                         'mean_gap': float(np.mean(gap[mask])), 'mean_residual': float(np.mean(resid[mask]))})
    return pd.DataFrame(rows)


def thresholds_from_validation(stats, idx, segments, W=8):
    """Small deterministic grid; validation-only and disjoint from final test traces."""
    g = np.concatenate([stats[r][1] for r in RANKS]); e = np.concatenate([stats[r][2] for r in RANKS])
    # Quantile-derived candidates keep the search tied to actual decoder scales.
    gq = np.quantile(g, [0.10, 0.20, 0.30, 0.40])
    eq = np.quantile(e, [0.35, 0.50, 0.65, 0.80])
    best = None
    for tau_g in gq:
        for tau_hi in eq:
            for tau_lo in [float(np.quantile(e, q)) for q in (0.20, 0.30, 0.40) if np.quantile(e, q) < tau_hi]:
                out = simulate_controller(stats, idx, segments, 'S4', W=W, tau_g=float(tau_g),
                                           tau_e_hi=float(tau_hi), tau_e_lo=float(tau_lo), min_dwell=2)
                loss = out['errors'].mean() + 0.0002 * out['switches']
                if best is None or loss < best['loss']:
                    best = {'W': int(W), 'tau_g': float(tau_g), 'tau_e_hi': float(tau_hi),
                            'tau_e_lo': float(tau_lo), 'min_dwell': 2, 'loss': float(loss)}
    return best


def simulate_controller(stats, idx, segments, scheme, *, W=8, alpha=0.2,
                         tau_g=0.08, tau_e_hi=0.45, tau_e_lo=0.25,
                         min_dwell=2, initial_rank=4, snr_to_rank=None,
                         regime_to_rank=None, rank_bias=None):
    n = len(idx); rank_to_i = {r: i for i, r in enumerate(RANKS)}
    selected = np.empty(n, dtype=int); pred = np.empty(n, dtype=int)
    gaps = np.empty(n); residuals = np.empty(n); evals = np.full(n, FULL_EVALS, dtype=float)
    active = int(initial_rank); dwell = 0; hist_g, hist_e = [], []
    ew_g = ew_e = None
    switches = 0
    # ACK-UCB state for optional S7; reward is observed after a packet ACK/NACK.
    counts = {r: 0 for r in RANKS}; rewards = {r: 0.0 for r in RANKS}
    for t in range(n):
        if scheme == 'S2':
            active = int(snr_to_rank[float(round(float(t), 0))]) if False else int(snr_to_rank[float(segments[t])])
        if scheme == 'S8':
            active = int(regime_to_rank[int(segments[t])])
        if scheme == 'S7':
            # forced warm-up, then optimistic ACK/NACK bandit with one-step feedback
            if t < len(RANKS):
                active = RANKS[t]
            else:
                total = max(1, t)
                ucb = {r: (rewards[r] / counts[r] if counts[r] else 1.0) +
                          0.7 * math.sqrt(math.log(total + 1) / max(1, counts[r])) for r in RANKS}
                active = max(RANKS, key=lambda r: ucb[r])
        selected[t] = active
        pred[t], gaps[t], residuals[t], _ = (stats[active][0][t], stats[active][1][t],
                                             stats[active][2][t], stats[active][3][t])
        correct = int(pred[t] == idx[t])
        if scheme == 'S7':
            counts[active] += 1; rewards[active] += correct
        hist_g.append(float(gaps[t])); hist_e.append(float(residuals[t]))
        if scheme in ('S3', 'S4', 'S5'):
            if scheme == 'S5':
                ew_g = float(gaps[t]) if ew_g is None else alpha * gaps[t] + (1-alpha) * ew_g
                ew_e = float(residuals[t]) if ew_e is None else alpha * residuals[t] + (1-alpha) * ew_e
                G, E = ew_g, ew_e
            else:
                G = float(np.mean(hist_g[-W:])); E = float(np.mean(hist_e[-W:]))
            candidate = active
            if dwell >= min_dwell and G < tau_g:
                if scheme == 'S3':
                    candidate = RANKS[min(len(RANKS)-1, rank_to_i[active] + 1)]
                elif E > tau_e_hi:
                    candidate = RANKS[min(len(RANKS)-1, rank_to_i[active] + 1)]
                elif E < tau_e_lo:
                    candidate = RANKS[max(0, rank_to_i[active] - 1)]
            if candidate != active:
                switches += 1; active = candidate; dwell = 0
            else:
                dwell += 1
        elif scheme == 'S6':
            # Confidence-triggered one-step refinement. The primary packet is
            # decoded at r=4; ambiguous packets compare adjacent ranks using a
            # validation-calibrated utility, not raw projection energy.
            if gaps[t] < tau_g:
                candidates = [active]
                if active > RANKS[0]: candidates.append(active-1)
                if active < RANKS[-1]: candidates.append(active+1)
                def util(r):
                    j = t
                    fit = 1.0 - stats[r][2][j]
                    return (stats[r][1][j] - (rank_bias or {}).get(r, 0.0) +
                            0.25 * (fit - (rank_bias or {}).get(('fit', r), 0.0)))
                chosen = max(candidates, key=util)
                evals[t] += (len(candidates)-1) * FULL_EVALS
                if chosen != active:
                    switches += 1; active = chosen
            dwell += 1
    out = {'scheme': scheme, 'errors': (pred != idx).astype(float), 'pred': pred,
           'selected': selected, 'gap': gaps, 'residual': residuals,
           'evals': evals, 'switches': switches}
    return out


def rank_maps(val_stats, val_idx, val_segments, val_snrs, val_delays):
    # Segment/regime map and SNR-only lookup are trained from validation PER.
    sm = segment_metric(val_stats, val_idx, val_segments, val_snrs, val_delays)
    overall = sm.groupby('rank').PER.mean()
    best_fixed = int(overall.idxmin())
    regime = {int(seg): int(g.loc[g.PER.idxmin(), 'rank']) for seg, g in sm.groupby('segment')}
    # S2 gets one rank per SNR, pooling validation segments at the same SNR.
    snr_map = {}
    for snr, g0 in sm.groupby('snr_db'):
        agg = g0.groupby('rank').PER.mean(); snr_map[float(snr)] = int(agg.idxmin())
    # Rank-specific centering constants for S6's confidence utility.
    rb = {r: float(np.mean(val_stats[r][1])) for r in RANKS}
    rb.update({('fit', r): float(np.mean(1-val_stats[r][2])) for r in RANKS})
    return best_fixed, snr_map, regime, rb, sm


def metrics(result, segments, target_rank=None):
    selected = result['selected']; n = len(selected)
    out = {'PER': float(result['errors'].mean()), 'mean_selected_rank': float(selected.mean()),
           'switches': int(result['switches']), 'switch_rate': float(result['switches'] / max(1,n-1)),
           'mean_score_evals': float(result['evals'].mean()),
           'p95_score_evals': float(np.quantile(result['evals'], .95))}
    if target_rank is not None:
        delays=[]
        for seg in sorted(set(segments)):
            ix=np.flatnonzero(segments==seg); target=int(target_rank[int(seg)])
            hit=None
            for k in range(len(ix)):
                tail=selected[ix[k:min(len(ix),k+10)]]
                if len(tail) >= 10 and np.all(tail == target): hit=k; break
            delays.append(float(hit if hit is not None else len(ix)))
        out['mean_adaptation_delay_packets'] = float(np.mean(delays))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks-per-segment', type=int, default=300)
    ap.add_argument('--val-blocks-per-segment', type=int, default=250)
    ap.add_argument('--seeds', default='13,59,151')
    ap.add_argument('--output', default='Data/online_adaptive_results.csv')
    ap.add_argument('--summary-output', default='Data/online_adaptive_summary.csv')
    ap.add_argument('--config-output', default='Data/online_adaptive_config.json')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    snr_trace = [5.0, 10.0, 15.0, 7.0]
    delay_trace = [30.0, 60.0, 100.0, 30.0]
    # Validation and final use disjoint seed families.  Validation is pooled
    # over the supplied seeds so that the lookup/controller is not tuned to a
    # single Monte-Carlo realization.
    val_traces = [make_trace(args.val_blocks_per_segment, 91000 + s, snr_trace, delay_trace)
                  for s in seeds]
    _, _, _, _, _, Xd, powers, D = val_traces[0]
    bases = {r: precompute_dmm_bases(Xd, powers, D, r, 4.0)[1] for r in RANKS}
    val_stats_list = []
    val_idx_list, val_snr_list, val_delay_list, val_seg_list = [], [], [], []
    for val in val_traces:
        val_stats_list.append(all_rank_stats(val[0], bases))
        val_idx_list.append(val[1]); val_snr_list.append(val[2])
        val_delay_list.append(val[3]); val_seg_list.append(val[4])
    vstats = {r: tuple(np.concatenate([s[r][k] for s in val_stats_list])
                       for k in range(4)) for r in RANKS}
    vidx = np.concatenate(val_idx_list); vsnrs = np.concatenate(val_snr_list)
    vdelays = np.concatenate(val_delay_list); vsegs = np.concatenate(val_seg_list)
    best_fixed, snr_map, regime_map, rank_bias, sm = rank_maps(vstats, vidx, vsegs, vsnrs, vdelays)
    th = thresholds_from_validation(vstats, vidx, vsegs, W=8)
    # Add a separately chosen EWMA setting; same residual/gap thresholds, alpha tuned on val.
    ewma_candidates = [0.10, 0.20, 0.35, 0.50]
    ewma_score = []
    for a in ewma_candidates:
        rr = simulate_controller(vstats, vidx, vsegs, 'S5', alpha=a, tau_g=th['tau_g'],
                                 tau_e_hi=th['tau_e_hi'], tau_e_lo=th['tau_e_lo'], min_dwell=2)
        ewma_score.append((rr['errors'].mean(), a))
    alpha = min(ewma_score)[1]
    # S6 confidence threshold tuned on the same validation trace, then frozen.
    s6_grid = np.quantile(np.concatenate([vstats[r][1] for r in RANKS]), [0.10, .20, .30, .40])
    s6_best = min((simulate_controller(vstats, vidx, vsegs, 'S6', initial_rank=4,
                                        tau_g=float(g), rank_bias=rank_bias) for g in s6_grid),
                  key=lambda x: x['errors'].mean())
    s6_tau = float(s6_grid[np.argmin([simulate_controller(vstats, vidx, vsegs, 'S6', initial_rank=4,
                                          tau_g=float(g), rank_bias=rank_bias)['errors'].mean() for g in s6_grid])])

    rows = []
    segment_rows = []
    for seed in seeds:
        tr = make_trace(args.blocks_per_segment, 120000 + seed, snr_trace, delay_trace)
        Y, idx, snrs, delays, segs = tr[:5]
        stats = all_rank_stats(Y, bases)
        schemes = {
            'S0_fixed_r4': ('fixed', {}),
            'S1_best_fixed': ('fixed', {'rank': best_fixed}),
            'S2_SNR_class_lookup': ('S2', {'snr_to_rank': {float(i): snr_map[float(snr_trace[int(i)])] for i in range(4)}}),
            'S3_window_confidence': ('S3', {'W': 8, 'tau_g': th['tau_g'], 'min_dwell': th['min_dwell']}),
            'S4_window_confidence_residual': ('S4', {'W': th['W'], 'tau_g': th['tau_g'], 'tau_e_hi': th['tau_e_hi'], 'tau_e_lo': th['tau_e_lo'], 'min_dwell': th['min_dwell']}),
            'S5_EWMA_confidence_residual': ('S5', {'alpha': alpha, 'tau_g': th['tau_g'], 'tau_e_hi': th['tau_e_hi'], 'tau_e_lo': th['tau_e_lo'], 'min_dwell': th['min_dwell']}),
            'S6_confidence_triggered_refinement': ('S6', {'tau_g': s6_tau, 'rank_bias': rank_bias}),
            'S7_ACK_UCB_optional': ('S7', {}),
            'S8_oracle_rank': ('S8', {'regime_to_rank': regime_map}),
        }
        for name, (scheme, kw) in schemes.items():
            if scheme == 'fixed':
                r = 4 if name.startswith('S0') else int(kw['rank'])
                res = {'scheme': scheme, 'errors': (stats[r][0] != idx).astype(float),
                       'pred': stats[r][0], 'selected': np.full(len(idx), r, int),
                       'gap': stats[r][1], 'residual': stats[r][2], 'evals': np.full(len(idx), FULL_EVALS, float), 'switches': 0}
            else:
                res = simulate_controller(stats, idx, segs, scheme, **kw)
            # Adaptation delay is meaningful for policies that can change
            # rank; fixed-rank references are reported as N/A.
            mm = metrics(res, segs, regime_map if scheme not in ('fixed',) else None)
            mm.update({'seed': seed, 'method': name, 'blocks': len(idx), 'best_fixed_validation_rank': best_fixed,
                       'snr_trace': '5-10-15-7', 'delay_trace_ns': '30-60-100-30'})
            rows.append(mm)
            for seg in sorted(set(segs)):
                mask = segs == seg
                segment_rows.append({
                    'seed': seed, 'method': name, 'segment': int(seg),
                    'snr_db': float(snrs[mask][0]), 'delay_ns': float(delays[mask][0]),
                    'PER': float(np.mean(res['errors'][mask])),
                    'mean_selected_rank': float(np.mean(res['selected'][mask])),
                    'mean_score_evals': float(np.mean(res['evals'][mask])),
                    'switches_within_segment': int(np.sum(np.diff(res['selected'][mask]) != 0)),
                })
    df = pd.DataFrame(rows)
    # Aggregate over seeds with 95% CI half-widths.
    summary = []
    for method, g in df.groupby('method', sort=False):
        row = {'method': method, 'seeds': len(g), 'blocks_total': int(g.blocks.sum())}
        for col in ['PER','mean_selected_rank','switch_rate','mean_adaptation_delay_packets','mean_score_evals','p95_score_evals']:
            row[col] = float(g[col].mean()); row[col+'_95CI_halfwidth'] = float(0 if len(g)<2 else 1.96*g[col].std(ddof=1)/math.sqrt(len(g)))
        summary.append(row)
    summary = pd.DataFrame(summary)
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    sout = ROOT / args.summary_output; summary.to_csv(sout, index=False)
    (ROOT / args.summary_output).with_name(Path(args.summary_output).stem + '_by_segment.csv').write_text(
        pd.DataFrame(segment_rows).to_csv(index=False), encoding='utf-8')
    cfg = {'validation_seed': 91000 + seeds[0], 'final_seeds': seeds, 'snr_trace_db': snr_trace,
           'delay_trace_ns': delay_trace, 'ranks': RANKS, 'best_fixed_validation_rank': best_fixed,
           'snr_lookup_rank': snr_map, 'regime_oracle_rank': regime_map, 'thresholds': th,
           'ewma_alpha': alpha, 's6_confidence_threshold': s6_tau,
           'notes': 'S2 receives slow SNR class; S7 receives one-bit ACK/NACK; S8 receives regime oracle.'}
    (ROOT / args.config_output).write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('validation rank map:', regime_map, 'SNR lookup:', snr_map, 'best fixed:', best_fixed)
    print('thresholds:', th, 'EWMA alpha:', alpha, 'S6 tau:', s6_tau)
    print(summary.to_string(index=False))

if __name__ == '__main__': main()
