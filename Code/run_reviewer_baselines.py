"""Reviewer-requested coded-OFDM controls on a shared-seed TDL-A experiment.

The script keeps the 8-bit/16-complex-sample budget and reuses the same channel,
CFO, front-end matrix, DC offset, and noise draws for every method within a seed.
It reports per-seed results so t-based intervals and paired comparisons can be
computed without treating individual packets as independent replicates.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, random_tdla_coefficients
from run_final_headtohead import (
    M, L, NFFT, CP, PILOT_BINS, DATA_BINS, randA, apply_iq_dc_noise,
    make_ofdm_codebooks, precompute_dmm_bases, decode_dmm,
)
ROOT = Path(__file__).resolve().parents[1]
T_CRIT_DF2 = 4.302652729911275  # two-sided 95% t critical value, df=2


def normalized_A(kappa: float) -> np.ndarray:
    """Diagonal I/Q matrix with ||A||_F^2=2 for every condition number."""
    s_hi = math.sqrt(2.0) * kappa / math.sqrt(1.0 + kappa*kappa)
    s_lo = math.sqrt(2.0) / math.sqrt(1.0 + kappa*kappa)
    return np.diag([s_hi, s_lo])


def messages8() -> np.ndarray:
    return np.array([[(m >> k) & 1 for k in range(8)] for m in range(M)], dtype=np.int8)


def code_full_bpsk() -> np.ndarray:
    bits = messages8()
    rs = np.random.default_rng(20260919)
    bestd, bestP = -1, None
    for _ in range(20000):
        P = rs.integers(0, 2, (8, 4), dtype=np.int8)
        G = np.concatenate([np.eye(8, dtype=np.int8), P], 1)
        C = (bits @ G) % 2
        d = int(C[1:].sum(1).min())
        if d > bestd:
            bestd, bestP = d, P.copy()
        if d >= 3:
            break
    G = np.concatenate([np.eye(8, dtype=np.int8), bestP], 1)
    coded = (bits @ G) % 2
    return (1 - 2*coded).astype(complex)


def code_pilot_qpsk_repeat() -> np.ndarray:
    """Four pilots + eight QPSK data symbols carrying a rate-matched 8-bit payload.

    Each information bit is repeated twice in the 16 real coded labels. This is
    intentionally simple and transparent: four pilots and eight QPSK symbols
    still occupy the same 12 active tones and 16 total time samples.
    """
    bits = np.repeat(messages8(), 2, axis=1)
    data = np.empty((M, 8), complex)
    for q in range(8):
        data[:, q] = ((1 - 2*bits[:, 2*q]) + 1j*(1 - 2*bits[:, 2*q+1]))/math.sqrt(2)
    F = np.zeros((M, NFFT), complex)
    F[:, PILOT_BINS] = 1
    F[:, DATA_BINS] = data
    return F


def scale_ofdm(F: np.ndarray, cp: int = CP):
    x = np.fft.ifft(F, axis=1, norm='ortho')
    x = np.concatenate([x[:, -cp:], x], axis=1)
    scale = math.sqrt(L / np.mean(np.sum(np.abs(x)**2, axis=1)))
    return F * scale, x * scale


def channel_freq(h: np.ndarray, delays: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    k = np.arange(nfft)
    E = np.exp(-1j * 2*np.pi*k[:, None] * delays[None, :] / nfft)
    return h @ E.T


def ofdm_received(Fsel, h, delays, omega, A, d, noise):
    H = channel_freq(h, delays)
    yf = H * Fsel
    yu = np.fft.ifft(yf, axis=1, norm='ortho')
    y = np.concatenate([yu[:, -CP:], yu], axis=1)
    return apply_iq_dc_noise(y, omega, A, d, noise)


def inv2(A):
    det = A[..., 0, 0]*A[..., 1, 1] - A[..., 0, 1]*A[..., 1, 0]
    R = np.empty_like(A)
    R[..., 0, 0] = A[..., 1, 1]/det; R[..., 1, 1] = A[..., 0, 0]/det
    R[..., 0, 1] = -A[..., 0, 1]/det; R[..., 1, 0] = -A[..., 1, 0]/det
    return R


def pilot_lmmse(R, Fscaled, powers, delays, pilot_bins):
    E = np.exp(-1j * 2*np.pi*np.arange(NFFT)[:, None] * delays[None, :] / NFFT)
    RH = (E*powers[None, :]) @ E.conj().T
    pilot_amp2 = float(np.mean(np.abs(Fscaled[0, pilot_bins])**2))
    Rpp = RH[np.ix_(pilot_bins, pilot_bins)]
    Rap = RH[:, pilot_bins]
    # This is the proper-complex approximation used only for the pilot estimate;
    # the condition-number sweep below uses the augmented improper-noise update.
    Hobs = R[:, pilot_bins] / Fscaled[0, pilot_bins][None, :]
    out = np.empty((len(R), NFFT), complex)
    for n in range(len(R)):
        out[n] = Rap @ np.linalg.solve(Rpp + (1.0/pilot_amp2)*np.eye(len(pilot_bins)), Hobs[n])
    return out


def decode_pilot_coded(Y, A, d, omega, Fscaled, powers, delays, data_bins):
    IA = inv2(A)
    z = np.einsum('nij,njk->nik', IA, Y - d[:, :, None])
    z = z[:, 0] + 1j*z[:, 1]
    z *= np.exp(-1j*omega[:, None]*np.arange(L))
    R = np.fft.fft(z[:, CP:], axis=1, norm='ortho')
    Hhat = pilot_lmmse(R, Fscaled, powers, delays, PILOT_BINS)
    pred = np.empty(len(Y), int)
    for s in range(0, len(Y), 200):
        e = min(len(Y), s+200)
        expct = Hhat[s:e, None, data_bins] * Fscaled[None, :, data_bins]
        dist = np.sum(np.abs(R[s:e, None, data_bins] - expct)**2, axis=2)
        pred[s:e] = np.argmin(dist, axis=1)
    return pred


def marginal_covariances(Fscaled, delays, powers, N0, A):
    """Real Gaussian TDL-A marginal covariances after known IQ/DC/CFO removal."""
    IA = np.linalg.inv(A)
    sigma = (N0/2.0) * IA @ IA.T
    Knoise = np.kron(np.eye(L), sigma)  # vector order: [I,Q] per sample
    invs, logdets = [], []
    for m in range(M):
        Hf = np.exp(-1j*2*np.pi*np.arange(NFFT)[:, None]*delays[None, :]/NFFT)
        cols = []
        for p, pw in enumerate(powers):
            y = np.fft.ifft(Hf[:, p] * Fscaled[m], norm='ortho')
            y = np.concatenate([y[-CP:], y])
            u = np.stack([y.real, y.imag], axis=1).reshape(-1)
            v = np.stack([(1j*y).real, (1j*y).imag], axis=1).reshape(-1)
            cols += [math.sqrt(pw/2.0)*u, math.sqrt(pw/2.0)*v]
        G = np.stack(cols, axis=1)
        C = G @ G.T + Knoise
        sign, ld = np.linalg.slogdet(C)
        if sign <= 0: raise RuntimeError('non-positive covariance')
        invs.append(np.linalg.inv(C)); logdets.append(ld)
    return np.asarray(invs), np.asarray(logdets)


def decode_marginal(Y, A, d, omega, invs, logdets):
    IA = np.linalg.inv(A)
    z = np.einsum('ij,bjk->bik', IA, Y - d[:, :, None])
    zc = z[:, 0] + 1j*z[:, 1]
    zc *= np.exp(-1j*omega[:, None]*np.arange(L))
    yv = np.stack([zc.real, zc.imag], axis=2).reshape(len(Y), -1)
    # Constant 1/2 factors and the common log(2pi) term cancel.
    q = np.einsum('bi,mij,bj->bm', yv, invs, yv, optimize=True) + logdets[None, :]
    return np.argmin(q, axis=1)


def decode_unknown_affine(Y, Fscaled, powers, delays):
    """Practical pilot receiver: CFO from CP, pilot LMMSE, affine profile over 256 data hypotheses."""
    # Remove the sample mean as a DC estimate; residual DC is profiled below.
    yc = Y - Y.mean(axis=2, keepdims=True)
    zraw = yc[:, 0] + 1j*yc[:, 1]
    corr = np.sum(np.conj(zraw[:, :CP]) * zraw[:, NFFT:NFFT+CP], axis=1)
    omhat = np.angle(corr)/NFFT
    zraw *= np.exp(-1j*omhat[:, None]*np.arange(L))
    R = np.fft.fft(zraw[:, CP:], axis=1, norm='ortho')
    # Pilot estimate without a calibrated front end; the affine profile below
    # absorbs the remaining static I/Q/DC action.
    Hhat = pilot_lmmse(R, Fscaled, powers, delays, PILOT_BINS)
    pred = np.empty(len(Y), int)
    Xf = Fscaled[:, :]
    for b in range(len(Y)):
        yu = np.fft.ifft(Hhat[b, None, :] * Xf, axis=1, norm='ortho')
        x = np.concatenate([yu[:, -CP:], yu], axis=1) * np.exp(1j*omhat[b]*np.arange(L))[None, :]
        X = np.stack([x.real, x.imag], axis=1)  # M,2,L
        Z = np.concatenate([X, np.ones((M,1,L))], axis=1)  # M,3,L
        ZZ = np.einsum('mkl,mjl->mkj', Z, Z)
        Yb = Y[b]
        YZ = np.einsum('al,mjl->maj', Yb, Z)
        C = np.einsum('maj,mjk->mak', YZ, np.linalg.pinv(ZZ))
        fit = np.einsum('mak,mkl->mal', C, Z)
        resid = np.sum((Yb[None, :, :] - fit)**2, axis=(1,2))
        pred[b] = int(np.argmin(resid))
    return pred


def run_point(snr, blocks, seed, rank=4, kappa=1.5):
    rng = np.random.default_rng(seed)
    delays, powers, D = tdla_operators(L)
    X = np.load(ROOT/'Artifact'/'fractional_incidence_codebook.npz')['X_opt']
    Xd = X[:, 0] + 1j*X[:, 1]
    _, bases = precompute_dmm_bases(Xd, powers, D, rank, 4)
    Ffull, _ = scale_ofdm(code_full_bpsk())
    Fpilot, _ = scale_ofdm(code_pilot_qpsk_repeat())
    Fraw, _, _, _, _ = make_ofdm_codebooks()
    idx = rng.integers(M, size=blocks)
    h = random_tdla_coefficients(rng, blocks, powers)
    omega = rng.uniform(-math.radians(4), math.radians(4), blocks)
    A0 = normalized_A(kappa)
    A = np.repeat(A0[None, :, :], blocks, axis=0)
    ang = rng.uniform(0, 2*np.pi, blocks); d = np.stack([np.cos(ang), np.sin(ang)], 1)
    N0 = 10**(-snr/10)
    noise = math.sqrt(N0/2)*rng.normal(size=(blocks,2,L))
    # SQM uses the same unknown front-end and channel realization.
    sd = np.zeros((blocks, L), complex)
    for p in range(len(powers)):
        sd += h[:, p, None]*(Xd[idx] @ D[p].T)
    Ysqm = apply_iq_dc_noise(sd, omega, A, d, noise)
    pred_sqm = decode_dmm(Ysqm, bases)
    # Pilot-free coded marginal ML: A, DC and CFO are supplied only for this
    # controlled baseline; TDL-A channel is marginalized, not estimated.
    Yfull = ofdm_received(Ffull[idx], h, delays, omega, A, d, noise)
    invs, logdets = marginal_covariances(Ffull, delays, powers, N0, A0)
    pred_marg = decode_marginal(Yfull, A0, d, omega, invs, logdets)
    # Rate-matched pilot-aided coded variant: known A/DC/CFO and 8-bit payload.
    Ypilot = ofdm_received(Fpilot[idx], h, delays, omega, A, d, noise)
    pred_pilot = decode_pilot_coded(Ypilot, A, d, omega, Fpilot, powers, delays, DATA_BINS)
    # Practical uncalibrated pilot receiver: estimates CFO, profiles A/DC, and
    # uses the pilot-domain LMMSE estimate; no A or d is given to the decoder.
    Yraw = ofdm_received(Fraw[idx], h, delays, omega, A, d, noise)
    pred_unknown = decode_unknown_affine(Yraw, Fraw, powers, delays)
    return {
        'SQM_rank4': float(np.mean(pred_sqm != idx)),
        'OFDM_pilot_free_marginal': float(np.mean(pred_marg != idx)),
        'OFDM_rate_matched_pilot_coded': float(np.mean(pred_pilot != idx)),
        'OFDM_practical_unknown_IQ_DC': float(np.mean(pred_unknown != idx)),
    }


def summarize(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]
    for (snr, method), g in per_seed.groupby(['snr_db','method']):
        x=g.PER.to_numpy(float); mean=float(x.mean()); half=T_CRIT_DF2*x.std(ddof=1)/math.sqrt(len(x)) if len(x)>1 else 0.0
        rows.append({'snr_db':snr,'method':method,'PER_mean':mean,'PER_95CI_halfwidth_t_df2':half,'seeds':len(x),'blocks_total':int(g.blocks.sum())})
    sm=pd.DataFrame(rows)
    paired=[]
    for snr, g in per_seed.groupby('snr_db'):
        pivot=g.pivot(index='seed', columns='method', values='PER')
        for method in pivot.columns:
            if method=='SQM_rank4': continue
            x=(pivot['SQM_rank4']-pivot[method]).dropna().to_numpy(float)
            half=T_CRIT_DF2*x.std(ddof=1)/math.sqrt(len(x)) if len(x)>1 else 0.0
            paired.append({'snr_db':snr,'comparison':'SQM - '+method,'paired_delta_mean':float(x.mean()),'paired_95CI_halfwidth_t_df2':half,'seeds':len(x)})
    return sm, pd.DataFrame(paired)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snr',default='10'); ap.add_argument('--blocks',type=int,default=100); ap.add_argument('--seeds',default='13,59,151'); ap.add_argument('--rank',type=int,default=4); ap.add_argument('--output',default='Data/reviewer_baselines_per_seed.csv'); ap.add_argument('--summary-output',default='Data/reviewer_baselines_summary.csv'); ap.add_argument('--paired-output',default='Data/reviewer_baselines_paired.csv')
    a=ap.parse_args(); rows=[]
    for snr in [float(v) for v in a.snr.split(',')]:
        for seed in [int(v) for v in a.seeds.split(',')]:
            print('running',snr,seed,flush=True)
            out=run_point(snr,a.blocks,20260929+seed+int(100*snr),a.rank)
            for method, per in out.items(): rows.append({'snr_db':snr,'seed':seed,'method':method,'PER':per,'blocks':a.blocks})
    per=pd.DataFrame(rows); sm, paired=summarize(per)
    for path,df in [(a.output,per),(a.summary_output,sm),(a.paired_output,paired)]:
        p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False); print('saved',p)
    print('\nSUMMARY\n',sm.to_string(index=False)); print('\nPAIRED\n',paired.to_string(index=False))

if __name__=='__main__': main()
