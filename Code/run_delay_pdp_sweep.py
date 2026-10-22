"""SQM robustness experiments: delay-spread sweep and PDP mismatch.

The SQM codebook is fixed to the shipped nominal 30-ns artifact.  The receiver
can be evaluated with a matched or mismatched nuisance dictionary.  OFDM uses
the same true physical channel but its LMMSE estimator is supplied either the
nominal or the true delay/power covariance, making the mismatch assumption
explicit in the output.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, exact_fractional_channel, random_tdla_coefficients
from run_final_headtohead import (
    M, L, make_ofdm_codebooks, randA, apply_iq_dc_noise, ofdm_received,
    precompute_dmm_bases, decode_dmm, decode_practical
)
from experiment_config import PHYSICAL_CFO_MIN, PHYSICAL_CFO_MAX

ROOT = Path(__file__).resolve().parents[1]


def true_powers(nominal: np.ndarray, exponent: float) -> np.ndarray:
    p = np.power(nominal, exponent)
    return p / p.sum()


def one_point(snr, true_scale, assumed_scale, pdp_exponent, blocks, seed, rank=4):
    rng = np.random.default_rng(seed)
    X = np.load(ROOT / 'Artifact' / 'fractional_incidence_codebook.npz')['X_opt']
    Xd = X[:, 0] + 1j * X[:, 1]
    # Physical channel and receiver model.
    true_delays, nominal_powers, D_true = tdla_operators(L, true_scale)
    assumed_delays, assumed_powers, D_assumed = tdla_operators(L, assumed_scale)
    phys_powers = true_powers(nominal_powers, pdp_exponent)
    _, bases = precompute_dmm_bases(Xd, assumed_powers, D_assumed, rank, 4)
    Fp, Fo, _, _, _ = make_ofdm_codebooks()
    idx = rng.integers(M, size=blocks)
    h = random_tdla_coefficients(rng, blocks, phys_powers)
    omega = rng.uniform(np.deg2rad(PHYSICAL_CFO_MIN), np.deg2rad(PHYSICAL_CFO_MAX), blocks)
    A = randA(rng, blocks, 1.5)
    ang = rng.uniform(0, 2*np.pi, blocks)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    noise = np.sqrt(10**(-snr/10)/2) * rng.normal(size=(blocks, 2, L))
    sd = exact_fractional_channel(Xd[idx], h, D_true)
    Yd = apply_iq_dc_noise(sd, omega, A, d, noise)
    Yp = ofdm_received(Fp[idx], h, true_delays, omega, A, d, noise)
    dmm_err = np.mean(decode_dmm(Yd, bases) != idx)
    # Practical OFDM deliberately uses the assumed covariance, while waveform
    # generation uses the true channel.
    ofdm_err = np.mean(decode_practical(Yp, A, d, 10**(-snr/10), Fp,
                                        assumed_powers, assumed_delays) != idx)
    return dmm_err, ofdm_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['delay', 'pdp'], default='delay')
    ap.add_argument('--snr', type=float, default=10.0)
    ap.add_argument('--blocks', type=int, default=200)
    ap.add_argument('--seeds', default='13,59')
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--output', default='Data/delay_pdp_sweep.csv')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    rows = []
    if args.mode == 'delay':
        settings = [(float(s), 30.0, 1.0) for s in [0, 15, 30, 60, 100]]
    else:
        settings = [(30.0, 30.0, float(a)) for a in [0.5, 0.75, 1.0, 1.25, 1.5]]
    for true_scale, assumed_scale, exponent in settings:
        for sd in seeds:
            dmm, ofdm = one_point(args.snr, true_scale, assumed_scale, exponent,
                                   args.blocks, 20260929 + sd + int(true_scale*10), args.rank)
            tag = 'delay_spread_ns' if args.mode == 'delay' else 'pdp_power_exponent'
            val = true_scale if args.mode == 'delay' else exponent
            rows.extend([
                {'experiment': args.mode, tag: val, 'method': 'Incidence-trained SQM',
                 'PER': dmm, 'snr_db': args.snr, 'seed': sd,
                 'true_delay_scale_ns': true_scale, 'assumed_delay_scale_ns': assumed_scale,
                 'pdp_power_exponent': exponent, 'rank': args.rank},
                {'experiment': args.mode, tag: val, 'method': 'CP-OFDM TDL-LMMSE',
                 'PER': ofdm, 'snr_db': args.snr, 'seed': sd,
                 'true_delay_scale_ns': true_scale, 'assumed_delay_scale_ns': assumed_scale,
                 'pdp_power_exponent': exponent, 'rank': args.rank},
            ])
    df = pd.DataFrame(rows)
    print(df.groupby(['experiment', df.columns[1], 'method']).PER.mean().to_string())
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('saved', out)

if __name__ == '__main__':
    main()
