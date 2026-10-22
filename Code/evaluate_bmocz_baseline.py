"""Reproduce the direct BMOCZ vs two-tap Incidence-trained SQM screening experiment.

BMOCZ encoder follows the conjugate-reciprocal zero construction of
Walk, Jung, and Hassibi (IEEE TWC 2019/2020).  The decoder used here is
stronger than low-complexity DiZeT: it is a true-PDP Gaussian marginal ML
search over all 256 messages and a CFO grid.  The exact channel PDP and
noise variance are supplied to BMOCZ.

Defaults reproduce the manuscript-scale screening counts (3 seeds x 800
packets per SNR point).  Radius selection is stored separately in Data/.
"""
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "Data"
A = ROOT / "Artifact"
M, K, L = 256, 8, 10
PDP = np.array([0.8, 0.2])
RADIUS = 1.60
MESSAGES = np.array([[(m >> k) & 1 for k in range(K)] for m in range(M)], dtype=np.int8)
X_DMM = np.load(A / "two_tap_incidence_codebook.npz")["X_inc"]
XD = X_DMM[:, 0] + 1j * X_DMM[:, 1]
T_CRIT_DF2 = 4.302652729911275


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def bmocz_codebook(R=RADIUS):
    X = np.empty((M, K + 1), complex)
    ang = 2 * np.pi * np.arange(K) / K
    for m in range(M):
        roots = np.where(MESSAGES[m] == 1, R, 1 / R) * np.exp(1j * ang)
        x = np.poly(roots)[::-1]
        X[m] = x / np.linalg.norm(x) * np.sqrt(L)
    return X


def convmat2(X):
    C = np.zeros((len(X), L, 2), complex)
    C[:, :K + 1, 0] = X
    C[:, 1:K + 2, 1] = X
    return C


def rand_h(rng, N):
    return ((rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2))) / np.sqrt(2)) * np.sqrt(PDP)


def mocz_conv(X, h):
    y = np.zeros((len(X), L), complex)
    y[:, :K + 1] += h[:, 0, None] * X
    y[:, 1:K + 2] += h[:, 1, None] * X
    return y


def dmm_conv(x, h):
    y = np.zeros_like(x, dtype=complex)
    y[:, 0] = h[:, 0] * x[:, 0]
    y[:, 1:] = h[:, 0, None] * x[:, 1:] + h[:, 1, None] * x[:, :-1]
    return y


def grid(cfo):
    if cfo == 0:
        return np.array([0.0])
    return np.deg2rad(np.linspace(-cfo-1, cfo+1, int(round(2*cfo+2))+1))


def mocz_bank(X, N0, wgrid):
    C = convmat2(X)
    t = np.arange(L)
    Uall = np.empty((M, len(wgrid), L, 2), complex)
    Binv = np.empty((M, len(wgrid), 2, 2), complex)
    logterm = np.empty((M, len(wgrid)), float)
    Dinv = np.diag(1 / PDP)
    sqrtD = np.diag(np.sqrt(PDP))
    for j, w in enumerate(wgrid):
        U = C * np.exp(1j * w * t)[None, :, None]
        Uall[:, j] = U
        G = np.einsum("mli,mlj->mij", U.conj(), U)
        Binv[:, j] = np.linalg.inv(Dinv[None] + G / N0)
        for m in range(M):
            Q = np.eye(2) + sqrtD @ G[m] @ sqrtD / N0
            logterm[m, j] = np.linalg.slogdet(Q)[1].real
    return Uall, Binv, logterm


def decode_mocz(y, U, Binv, logterm, N0):
    out = np.empty(len(y), int)
    W = logterm.shape[1]
    for s in range(0, len(y), 100):
        yy = y[s:s + 100]
        z = np.einsum("mwlq,bl->bmwq", U.conj(), yy, optimize=True)
        gain = np.einsum("bmwi,mwij,bmwj->bmw", z.conj(), Binv, z, optimize=True).real
        metric = logterm[None] - gain / (N0 * N0)
        best = np.argmin(metric.reshape(len(yy), -1), axis=1)
        out[s:s + len(yy)] = best // W
    return out


def dmm_projectors(wgrid):
    arr = np.empty((M, len(wgrid), L, L), np.float32)
    t = np.arange(L)
    for j, w in enumerate(wgrid):
        R = rot(w * t)
        U = np.einsum("tij,...jt->...it", R, X_DMM)
        V = np.zeros_like(U); V[..., 1:] = U[..., :-1]
        F = np.concatenate([U, V], -2)
        F -= F.mean(-1, keepdims=True)
        Q = np.linalg.qr(F.swapaxes(-2, -1))[0][..., :4]
        arr[:, j] = np.einsum("...ik,...jk->...ij", Q, Q).astype(np.float32)
    return arr.reshape(M * len(wgrid), L * L)


def decode_dmm(Y, P, W):
    Y = Y - Y.mean(2, keepdims=True)
    C = np.einsum("bil,bik->blk", Y, Y).reshape(len(Y), -1).astype(np.float32)
    out = np.empty(len(Y), int)
    for s in range(0, len(Y), 300):
        out[s:s + 300] = np.argmax(C[s:s + 300] @ P.T, 1) // W
    return out


def run(blocks=800, seed_list=(17,83,173)):
    X_mocz = bmocz_codebook()
    rows = []
    per_seed = []
    for cfo in [0.0, 8.0]:
        wg = grid(cfo)
        Pd = dmm_projectors(wg)
        for snr in [5,7,10,12,15]:
            N0 = 10 ** (-snr / 10)
            U, Bi, ld = mocz_bank(X_mocz, N0, wg)
            vals = []
            for sd in seed_list:
                rng = np.random.default_rng(20260919 + int(cfo)*10000 + snr*1000 + sd)
                idx = rng.integers(M, size=blocks)
                h = rand_h(rng, blocks)
                om = rng.uniform(-math.radians(cfo), math.radians(cfo), blocks) if cfo else np.zeros(blocks)
                n = np.sqrt(N0/2) * (rng.normal(size=(blocks,L)) + 1j*rng.normal(size=(blocks,L)))
                t = np.arange(L)
                ym = mocz_conv(X_mocz[idx], h) * np.exp(1j*om[:,None]*t) + n
                yd = dmm_conv(XD[idx], h) * np.exp(1j*om[:,None]*t) + n
                pred_dmm = decode_dmm(np.stack([yd.real, yd.imag],1), Pd, len(wg))
                pred_mocz = decode_mocz(ym, U, Bi, ld, N0)
                pair=[np.mean(pred_dmm != idx), np.mean(pred_mocz != idx)]
                vals.append(pair)
                per_seed.extend([
                    {"cfo_range_deg_per_sample":cfo,"snr_db":snr,"seed":sd,
                     "method":"Incidence-trained SQM","PER":pair[0],"blocks":blocks},
                    {"cfo_range_deg_per_sample":cfo,"snr_db":snr,"seed":sd,
                     "method":"BMOCZ true-PDP ML","PER":pair[1],"blocks":blocks},
                ])
            vals = np.asarray(vals)
            for j,name in enumerate(["Incidence-trained SQM","BMOCZ true-PDP ML"]):
                x = vals[:,j]
                rows.append({"cfo_range_deg_per_sample":cfo,"snr_db":snr,"method":name,
                             "PER_mean":x.mean(),"PER_95CI_halfwidth":T_CRIT_DF2*x.std(ddof=1)/np.sqrt(len(x)),
                             "blocks":blocks*len(x)})
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=800)
    ap.add_argument("--out", default=str(D / "dmm_vs_bmocz_rerun.csv"))
    args = ap.parse_args()
    df, ps = run(args.blocks)
    df.to_csv(args.out, index=False)
    ps.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_per_seed.csv")), index=False)
    print(df.to_string(index=False))
