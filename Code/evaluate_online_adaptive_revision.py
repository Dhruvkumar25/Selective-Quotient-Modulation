"""Reproducible reviewer-requested extensions for online SQM rank selection.

This script keeps the blockwise SQM decoder unchanged and adds only the
requested evaluation/reporting details: an independent held-out rank-sweep
oracle, a second randomized trace with unequal regime durations, packet-level
rank/error logs, transient PER windows, selected-rank transitions, and
rank-dependent approximate MAC counts.
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

from evaluate_online_adaptive import (
    make_trace, all_rank_stats, simulate_controller, RANKS, FULL_EVALS,
    thresholds_from_validation, rank_maps, metrics,
)
from run_final_headtohead import precompute_dmm_bases

ROOT = Path(__file__).resolve().parents[1]
M, L = 256, 16
T975_DF2 = 4.3026527299
ETA = 1e-12

SEEDS = [13, 59, 151]
VALIDATION_SEEDS = [91013, 91059, 91151]
CANONICAL_STATES = [(5.0, 30.0), (10.0, 60.0), (15.0, 100.0), (7.0, 30.0)]
CANONICAL_LENGTHS = [300, 300, 300, 300]
# Same operating states, randomized order and unequal durations.  The repeated
# 10-dB states are intentional: an SNR-only lookup cannot distinguish them,
# whereas the independent state oracle can.
RANDOMIZED_STATES = [(10.0, 30.0), (15.0, 60.0), (7.0, 30.0), (10.0, 100.0)]
RANDOMIZED_LENGTHS = [180, 240, 300, 360]
ALL_STATES = list(dict.fromkeys(CANONICAL_STATES + RANDOMIZED_STATES))


def ci(values):
    values = np.asarray(values, dtype=float)
    return float(T975_DF2 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def state_trace(lengths, states, seed):
    ys, idxs, snrs, delays, segs = [], [], [], [], []
    Xd = powers = D_nom = None
    for seg, (n, (snr, delay)) in enumerate(zip(lengths, states)):
        tr = make_trace(int(n), int(seed + 1009 * seg), [snr], [delay])
        ys.append(tr[0]); idxs.append(tr[1]); snrs.append(tr[2]); delays.append(tr[3]); segs.append(np.full(n, seg, dtype=int))
        Xd, powers, D_nom = tr[5], tr[6], tr[7]
    return (np.concatenate(ys), np.concatenate(idxs), np.concatenate(snrs),
            np.concatenate(delays), np.concatenate(segs), Xd, powers, D_nom)


def build_bases():
    tiny = make_trace(2, 777, [5.0], [30.0])
    Xd, powers, D_nom = tiny[5], tiny[6], tiny[7]
    return {r: precompute_dmm_bases(Xd, powers, D_nom, r, 4.0)[1] for r in RANKS}


def heldout_rank_sweeps(bases):
    rows = []
    for state in ALL_STATES:
        seed_values = {r: [] for r in RANKS}
        for seed in VALIDATION_SEEDS:
            tr = make_trace(250, seed + int(state[0] * 17 + state[1]), [state[0]], [state[1]])
            stats = all_rank_stats(tr[0], bases)
            for r in RANKS:
                seed_values[r].append(float(np.mean(stats[r][0] != tr[1])))
        for r in RANKS:
            vals = seed_values[r]
            rows.append({
                'snr_db': state[0], 'delay_ns': state[1], 'rank': r,
                'validation_PER_mean': float(np.mean(vals)),
                'validation_PER_95CI': ci(vals),
                'validation_seed_values': ';'.join(f'{v:.8f}' for v in vals),
            })
    df = pd.DataFrame(rows)
    oracle = {}
    for (snr, delay), g in df.groupby(['snr_db', 'delay_ns']):
        oracle[(float(snr), float(delay))] = int(g.loc[g['validation_PER_mean'].idxmin(), 'rank'])
    # Practical lookup pools all held-out states sharing the same coarse SNR.
    lookup = {}
    for snr, g0 in df.groupby('snr_db'):
        agg = g0.groupby('rank').validation_PER_mean.mean()
        lookup[float(snr)] = int(agg.idxmin())
    best_fixed = int(df.groupby('rank').validation_PER_mean.mean().idxmin())
    return df, oracle, lookup, best_fixed


def thresholds_and_rank_bias(bases):
    # Reproduce the existing validation selection on the original four-state
    # trace; these values are frozen before either final trace is evaluated.
    traces = [make_trace(250, s, [x[0] for x in CANONICAL_STATES], [x[1] for x in CANONICAL_STATES]) for s in VALIDATION_SEEDS]
    stats_list = [all_rank_stats(t[0], bases) for t in traces]
    vstats = {r: tuple(np.concatenate([s[r][k] for s in stats_list]) for k in range(4)) for r in RANKS}
    vidx = np.concatenate([t[1] for t in traces])
    vsegs = np.concatenate([t[4] for t in traces])
    vsnrs = np.concatenate([t[2] for t in traces])
    vdelays = np.concatenate([t[3] for t in traces])
    # These controller parameters are chosen on validation, then frozen.
    th = thresholds_from_validation(vstats, vidx, vsegs, W=8)
    rank_bias = {r: float(np.mean(vstats[r][1])) for r in RANKS}
    rank_bias.update({('fit', r): float(np.mean(1 - vstats[r][2])) for r in RANKS})
    # Current released configuration fixes alpha at 0.20 from the validation
    # candidates; use it unchanged for both final traces.
    return th, rank_bias, vstats, vidx, vsegs, vsnrs, vdelays


def state_to_segment_map(states, state_map):
    return {int(i): int(state_map[(float(s), float(d))]) for i, (s, d) in enumerate(states)}


def run_one_scheme(stats, idx, segs, snrs, delays, scheme, kwargs, target_map):
    if scheme == 'fixed':
        r = int(kwargs['rank'])
        res = {
            'errors': (stats[r][0] != idx).astype(float),
            'pred': stats[r][0], 'selected': np.full(len(idx), r, dtype=int),
            'gap': stats[r][1], 'residual': stats[r][2],
            'evals': np.full(len(idx), FULL_EVALS, dtype=float), 'switches': 0,
        }
    else:
        res = simulate_controller(stats, idx, segs, scheme, **kwargs)
    selected = res['selected']
    selected_transitions = np.r_[False, selected[1:] != selected[:-1]]
    # Approximate real MACs: each score costs 2 L r real multiply-accumulates.
    # For S6 this is deliberately an upper-bound approximation because the
    # reproducibility implementation evaluates a full bank for refinement.
    approx_macs = res['evals'] * (2 * L * selected)
    target = np.array([target_map[int(s)] for s in segs], dtype=int)
    return res, selected_transitions.astype(int), approx_macs, target


def adaptation_delay(selected, segs, target, window=10):
    delays = []
    for seg in sorted(np.unique(segs)):
        ix = np.flatnonzero(segs == seg)
        targ = int(target[ix[0]])
        if np.all(selected[ix[:window]] == targ):
            delays.append(0.0); continue
        hit = None
        for k in range(0, max(0, len(ix) - window + 1)):
            if np.all(selected[ix[k:k + window]] == targ):
                hit = k; break
        delays.append(float(hit if hit is not None else len(ix)))
    return float(np.mean(delays))


def scheme_kwargs(name, lookup_by_snr, oracle_by_state, states, th, rank_bias, best_fixed, s6_tau):
    lookup_seg = {float(i): int(lookup_by_snr[(float(s), float(d)) if False else float(s)]) for i, (s, d) in enumerate(states)}
    oracle_seg = state_to_segment_map(states, oracle_by_state)
    common = {'tau_g': float(th['tau_g']), 'tau_e_hi': float(th['tau_e_hi']),
              'tau_e_lo': float(th['tau_e_lo']), 'min_dwell': int(th['min_dwell'])}
    if name == 'S0_fixed_r4': return 'fixed', {'rank': 4}
    if name == 'S1_best_fixed': return 'fixed', {'rank': best_fixed}
    if name == 'S2_SNR_class_lookup': return 'S2', {'snr_to_rank': lookup_seg}
    if name == 'S3_window_confidence': return 'S3', {'W': 8, 'tau_g': common['tau_g'], 'min_dwell': 2}
    if name == 'S4_window_confidence_residual': return 'S4', {'W': 8, **common}
    if name == 'S5_EWMA_confidence_residual': return 'S5', {'alpha': 0.20, **common}
    if name == 'S6_confidence_triggered_refinement': return 'S6', {'tau_g': s6_tau, 'rank_bias': rank_bias}
    if name == 'S7_ACK_UCB_optional': return 'S7', {}
    if name == 'S8_regime_oracle_rank': return 'S8', {'regime_to_rank': oracle_seg}
    raise ValueError(name)


def main():
    bases = build_bases()
    sweep_df, oracle_by_state, lookup_by_snr, best_fixed = heldout_rank_sweeps(bases)
    th, rank_bias, vstats, vidx, vsegs, vsnrs, vdelays = thresholds_and_rank_bias(bases)
    s6_grid = np.quantile(np.concatenate([vstats[r][1] for r in RANKS]), [0.10, .20, .30, .40])
    s6_scores = []
    for g in s6_grid:
        rr = simulate_controller(vstats, vidx, vsegs, 'S6', initial_rank=4, tau_g=float(g), rank_bias=rank_bias)
        s6_scores.append(float(rr['errors'].mean()))
    s6_tau = float(s6_grid[int(np.argmin(s6_scores))])
    names = ['S0_fixed_r4','S1_best_fixed','S2_SNR_class_lookup','S3_window_confidence',
             'S4_window_confidence_residual','S5_EWMA_confidence_residual',
             'S6_confidence_triggered_refinement','S7_ACK_UCB_optional','S8_regime_oracle_rank']
    all_seed_rows, packet_rows, segment_rows = [], [], []
    sequence_specs = {
        'canonical': (CANONICAL_STATES, CANONICAL_LENGTHS),
        'randomized_unequal': (RANDOMIZED_STATES, RANDOMIZED_LENGTHS),
    }
    for sequence, (states, lengths) in sequence_specs.items():
        target_map = state_to_segment_map(states, oracle_by_state)
        for seed in SEEDS:
            if sequence == 'canonical':
                tr = make_trace(int(lengths[0]), 120000 + seed, [x[0] for x in states], [x[1] for x in states])
            else:
                tr = state_trace(lengths, states, 120000 + seed + 50000)
            stats = all_rank_stats(tr[0], bases)
            idx, snrs, delays, segs = tr[1], tr[2], tr[3], tr[4]
            for name in names:
                sch, kw = scheme_kwargs(name, lookup_by_snr, oracle_by_state, states, th, rank_bias, best_fixed, s6_tau)
                res, transitions, macs, target = run_one_scheme(stats, idx, segs, snrs, delays, sch, kw, target_map)
                selected = res['selected']
                controller_switches = int(res.get('switches', 0))
                selected_changes = int(transitions.sum())
                target_delay = adaptation_delay(selected, segs, target) if name not in ('S0_fixed_r4','S1_best_fixed') else np.nan
                # One seed summary row.
                all_seed_rows.append({
                    'sequence': sequence, 'seed': seed, 'method': name, 'blocks': len(idx),
                    'PER': float(res['errors'].mean()), 'mean_selected_rank': float(selected.mean()),
                    'controller_switches': controller_switches,
                    'controller_switch_rate': float(controller_switches / max(1, len(idx)-1)),
                    'selected_rank_transitions': selected_changes,
                    'switch_rate': float(selected_changes / max(1, len(idx)-1)),
                    'mean_adaptation_delay_packets': target_delay,
                    'mean_score_evals': float(res['evals'].mean()),
                    'p95_score_evals': float(np.quantile(res['evals'], .95)),
                    'mean_approx_macs': float(macs.mean()),
                    'p95_approx_macs': float(np.quantile(macs, .95)),
                })
                # Packet-level artifact.
                for t in range(len(idx)):
                    packet_rows.append({
                        'sequence': sequence, 'seed': seed, 'method': name, 'packet': t,
                        'segment': int(segs[t]), 'snr_db': float(snrs[t]), 'delay_ns': float(delays[t]),
                        'error': int(res['errors'][t]), 'selected_rank': int(selected[t]),
                        'selected_rank_transition': int(transitions[t]),
                        'controller_switch': int(t > 0 and selected[t] != selected[t-1]) if name not in ('S2_SNR_class_lookup','S8_regime_oracle_rank') else 0,
                        'score_evals': float(res['evals'][t]), 'approx_macs': float(macs[t]),
                    })
                for seg in sorted(np.unique(segs)):
                    mask = segs == seg
                    segment_rows.append({
                        'sequence': sequence, 'seed': seed, 'method': name, 'segment': int(seg),
                        'snr_db': float(snrs[mask][0]), 'delay_ns': float(delays[mask][0]),
                        'packets': int(mask.sum()), 'PER': float(res['errors'][mask].mean()),
                        'mean_selected_rank': float(selected[mask].mean()),
                        'selected_rank_transitions_within_segment': int(np.sum(np.diff(selected[mask]) != 0)),
                        'mean_approx_macs': float(macs[mask].mean()),
                    })
    seed_df = pd.DataFrame(all_seed_rows)
    packet_df = pd.DataFrame(packet_rows)
    segment_df = pd.DataFrame(segment_rows)
    summary_rows = []
    for (sequence, method), g in seed_df.groupby(['sequence','method'], sort=False):
        row = {'sequence': sequence, 'method': method, 'seeds': len(g), 'blocks_total': int(g.blocks.sum())}
        for col in ['PER','mean_selected_rank','controller_switch_rate','switch_rate',
                    'mean_adaptation_delay_packets','mean_score_evals','p95_score_evals',
                    'mean_approx_macs','p95_approx_macs']:
            vals = g[col].dropna().to_numpy(float)
            row[col] = float(vals.mean()) if len(vals) else np.nan
            row[col + '_95CI_halfwidth'] = ci(vals) if len(vals) > 1 else 0.0
        row['controller_switches_mean'] = float(g.controller_switches.mean())
        row['selected_rank_transitions_mean'] = float(g.selected_rank_transitions.mean())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Transient windows after each transition, versus the remaining steady state.
    transient_rows = []
    for (sequence, seed, method), g in packet_df.groupby(['sequence','seed','method'], sort=False):
        if sequence != 'canonical': continue
        for seg in [1,2,3]:
            h = g[g.segment == seg].sort_values('packet')
            e = h.error.to_numpy()
            for label, a, b in [('first_25',0,25),('first_50',0,50),('first_100',0,100),('steady_after_100',100,len(e))]:
                transient_rows.append({'sequence':sequence,'seed':seed,'method':method,'transition_segment':seg,
                                       'window':label,'packets':max(0,b-a),'PER':float(e[a:b].mean())})
    transient_seed = pd.DataFrame(transient_rows)
    transient_summary = (transient_seed.groupby(['sequence','method','transition_segment','window'], sort=False)
                         .PER.agg(['mean','count','std']).reset_index())
    transient_summary['PER_95CI_halfwidth'] = transient_summary.apply(lambda r: T975_DF2*r['std']/math.sqrt(r['count']) if r['count']>1 else 0.0, axis=1)
    # Sensitivity is already a separate requested study; retain the shipped
    # values and expose them with the same t-based interval convention below.
    sens = pd.read_csv(ROOT/'Data/online_controller_sensitivity.csv')
    sens.to_csv(ROOT/'Data/online_controller_sensitivity_revision.csv', index=False)

    out = ROOT/'Data'
    seed_df.to_csv(out/'online_adaptive_revision_seed_metrics.csv', index=False)
    summary_df.to_csv(out/'online_adaptive_revision_summary.csv', index=False)
    packet_df.to_csv(out/'online_adaptive_packet_results.csv', index=False)
    segment_df.to_csv(out/'online_adaptive_revision_by_segment.csv', index=False)
    sweep_df.to_csv(out/'online_regime_rank_sweeps.csv', index=False)
    transient_seed.to_csv(out/'online_transient_seed_metrics.csv', index=False)
    transient_summary.to_csv(out/'online_transient_summary.csv', index=False)
    cfg = {
        'validation_seeds': VALIDATION_SEEDS, 'final_seeds': SEEDS, 'candidate_ranks': RANKS,
        'physical_cfo_support_deg_per_sample': [-4,4], 'decoder_grid_deg_per_sample': [-5,5],
        'controller_parameters': {
            'window_W': 8, 'ewma_alpha': 0.20, 'tau_gap': float(th['tau_g']),
            'tau_residual_high': float(th['tau_e_hi']), 'tau_residual_low': float(th['tau_e_lo']),
            'minimum_dwell_packets': int(th['min_dwell']), 'initial_rank': 4,
            'rank_min': 3, 'rank_max': 6, 'confidence_refinement_threshold': s6_tau,
            'ucb_exploration_coefficient': 0.7, 'eta': ETA,
        },
        'snr_lookup_rank': {str(k): int(v) for k,v in sorted(lookup_by_snr.items())},
        'state_oracle_rank': {f'{s:g}dB_{d:g}ns': int(v) for (s,d),v in sorted(oracle_by_state.items())},
        'definition': {
            'switch_rate': 'selected-rank transitions divided by T-1; controller switch rate is also reported separately',
            'adaptation_delay': 'first packet offset after a regime boundary with 10 consecutive packets at the held-out oracle rank; mean over internal boundaries',
            'steady_state_window': 'packets 100 through end of each 300-packet canonical segment',
            'approx_macs': 'score evaluations times 2*L*selected_rank; S6 is an upper-bound approximation',
        },
        'traces': {
            'canonical': {'states': CANONICAL_STATES, 'lengths': CANONICAL_LENGTHS},
            'randomized_unequal': {'states': RANDOMIZED_STATES, 'lengths': RANDOMIZED_LENGTHS},
        },
    }
    (out/'online_adaptive_revision_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(summary_df[['sequence','method','PER','switch_rate','mean_adaptation_delay_packets','mean_approx_macs']].to_string(index=False))
    print('oracle:', cfg['state_oracle_rank'])
    print('snr lookup:', cfg['snr_lookup_rank'])

if __name__ == '__main__':
    main()
