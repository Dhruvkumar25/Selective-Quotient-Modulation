# Selective-Quotient Modulation — Expanded Revision Package

See `REVISION_NOTES.md` for the implementation summary and reproducibility
commands. The paper source is `Manuscript/main.tex`.

## Fresh reruns and online adaptive extension

The current package includes a fresh three-seed main rerun in `Data/final_rerun_fractional_tdla_headtohead.csv`, the corresponding figure `Figure/fig_main_rerun.png`, and the online adaptive-rank experiment in `Code/evaluate_online_adaptive.py`. The updated paper is `Manuscript/main.tex`.

Main rerun:

```bash
python Code/run_final_headtohead.py --blocks 1200 --seeds 13,59,151 \
  --snrs 5,7,10,12,15 \
  --output Data/final_rerun_fractional_tdla_headtohead.csv
```

Online adaptive rerun:

```bash
python Code/evaluate_online_adaptive.py --blocks-per-segment 300 \
  --val-blocks-per-segment 250 --seeds 13,59,151 \
  --output Data/online_adaptive_results.csv \
  --summary-output Data/online_adaptive_summary.csv \
  --config-output Data/online_adaptive_config.json
```

The adaptive study is deliberately scoped: decoder-only confidence/residual controllers are not claimed to improve universally, and S2 is an operating-class lookup that receives slow SNR-class information. See `ADAPTIVE_RESULTS.md` for the exact results and caveats.

Additional diagnostics:

```bash
python Code/online_adaptive_diagnostics.py
python Code/evaluate_delay_transition.py
```

These produce proxy-calibration/AUC results, window and EWMA sensitivity tables, rank-regret summaries, and a fixed-SNR delay-spread transition experiment.

## Reviewer-strengthened controls

The current package additionally includes:

- `Code/run_reviewer_baselines.py`: shared-seed pilot-free coded OFDM with TDL-A-marginal ML, a payload-rate-matched pilot-aided coded control, and a practical unknown-I/Q/DC affine-profile receiver. It writes per-seed, t-based summary, and paired-difference CSVs.
- `Code/run_iq_correction.py`: fixed-`\|A\|_F^2=2` condition-number sweep with the induced post-equalization noise variance and covariance-aware pilot update.
- `Code/make_clean_figures.py`: regenerated coded-baseline, I/Q-control, and coarse-to-fine figures.
- `Code/train_selective_incidence.py --hard-negative-k K`: optional top-`K` hard-negative training mode; it is not mixed into the shipped test artifact.

Full reviewer baseline run:

```bash
python Code/run_reviewer_baselines.py --snr 5,7,10,12,15 --blocks 1200 \
  --seeds 13,59,151 \
  --output Data/reviewer_baselines_per_seed.csv \
  --summary-output Data/reviewer_baselines_summary.csv \
  --paired-output Data/reviewer_baselines_paired.csv
```

The main rerun uses two-sided 95% t intervals with two degrees of freedom and stores paired comparisons in `Data/final_rerun_fractional_tdla_headtohead_paired.csv`. The BMOCZ rerun is stored in `Data/dmm_vs_bmocz_rerun.csv` with seed-level data beside it.

## Reviewer-requested adaptive-rank revisions

The final adaptive extension is reproduced with:

```bash
python Code/evaluate_online_adaptive_revision.py
python Code/make_adaptive_revision_figures.py
```

This produces the independent held-out state-rank sweeps, canonical and randomized unequal-duration traces, packet-level selected-rank/error logs, transient PER windows, rank-dependent approximate MAC counts, and the W/alpha sensitivity figure. The exact controller values are stored in `Data/online_adaptive_revision_config.json`; the submission manuscript uses the canonical trace and reports the randomized trace as a robustness check.
