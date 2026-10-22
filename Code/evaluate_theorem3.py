"""Empirical validation of the selective-margin noise-radius bound.

For sampled noiseless TDL-A cases, compute epsilon_r, delta_r, and the certified
radius Delta_r/2 over the configured CFO grid.  Add Gaussian perturbations at
specified multiples of that radius and measure SQM decoding error.  The result
is a diagnostic of theorem conservatism, not a replacement for a global proof.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, exact_fractional_channel, random_tdla_coefficients, weighted_nuisance_dictionary
from run_final_headtohead import decode_dmm
from experiment_config import cfo_grid_rad, physical_cfo_sample, M, L

ROOT = Path(__file__).resolve().parents[1]


def make_bases(Xd, powers, D, rank):
    grid = cfo_grid_rad()
    B = np.empty((M, len(grid), L, rank), np.float32)
    for m in range(M):
        for j, w in enumerate(grid):
            F = weighted_nuisance_dictionary(Xd[m], w, powers, D)
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            B[m, j] = Vh[:rank].T.astype(np.float32)
    return grid, B


def sample_case(rng, Xd, powers, D):
    i = int(rng.integers(M))
    h = random_tdla_coefficients(rng, 1, powers)
    om = float(physical_cfo_sample(rng))
    Sx = exact_fractional_channel(Xd[i:i+1], h, D)[0] * np.exp(1j * om * np.arange(L))
    S = np.stack([Sx.real, Sx.imag])
    S = S - S.mean(axis=1, keepdims=True)
    return i, S


def distances(S, idx, B):
    energy = float(np.sum(S*S))
    # B is (M,G,L,r); projected energy is ||S Q||^2.
    Z = np.einsum('al,mglr->mgar', S, B, optimize=True)
    fit = np.sum(Z*Z, axis=(2, 3))
    dist = np.sqrt(np.maximum(0.0, energy - fit))
    eps = float(dist[idx].min())
    wrong = np.delete(dist, idx, axis=0)
    delta = float(wrong.min())
    return eps, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=40)
    ap.add_argument('--noise-draws', type=int, default=40)
    ap.add_argument('--ranks', default='3,4,6')
    ap.add_argument('--fractions', default='0.25,0.5,0.75,1.0,1.5,2.0')
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--output', default='Data/theorem3_validation.csv')
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(',')]
    fractions = [float(x) for x in a.fractions.split(',')]
    X = np.load(ROOT/'Artifact'/'fractional_incidence_codebook.npz')['X_opt']
    Xd = X[:,0] + 1j*X[:,1]
    _, powers, D = tdla_operators(L)
    rng = np.random.default_rng(a.seed)
    banks = {r: make_bases(Xd, powers, D, r)[1] for r in ranks}
    rows = []
    for case in range(a.cases):
        idx, S = sample_case(rng, Xd, powers, D)
        signal_norm = np.linalg.norm(S)
        for r in ranks:
            B = banks[r]
            eps, delta = distances(S, idx, B)
            margin = delta - eps
            radius = max(margin, 0.0)/2.0
            for frac in fractions:
                errs = 0
                for _ in range(a.noise_draws):
                    noise = rng.normal(size=S.shape)
                    noise /= max(np.linalg.norm(noise), 1e-12)
                    noise = noise * (frac * radius)
                    pred = int(decode_dmm((S + noise)[None], B)[0])
                    errs += int(pred != idx)
                rows.append({
                    'case': case, 'rank': r, 'noise_fraction_of_certified_radius': frac,
                    'epsilon_norm': eps/signal_norm, 'delta_norm': delta/signal_norm,
                    'margin_norm': margin/signal_norm,
                    'certified_radius_norm': radius/signal_norm,
                    'empirical_error_rate': errs/a.noise_draws,
                    'noise_draws': a.noise_draws,
                })
    df = pd.DataFrame(rows)
    print(df.groupby(['rank','noise_fraction_of_certified_radius']).empirical_error_rate.mean().to_string())
    out = ROOT/a.output; out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); print('saved', out)

if __name__ == '__main__': main()
