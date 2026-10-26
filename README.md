# Selective-Quotient Modulation (SQM)
### Calibration-light communication for short wireless packets

Research and reproducibility materials for **“Selective-Quotient Modulation for Calibration-Light Short-Packet Wireless Links”** by **Dhruv Kumar and Michael Georgiades**.

SQM investigates how short packets can tolerate selected channel and receiver uncertainties without estimating every nuisance parameter separately. The central design question is not simply how much invariance is possible, but **how much invariance preserves message distinguishability**.

**Quick links:** [Results](#selected-results) · [Installation](#installation) · [Run experiments](#run-experiments) · [Validation report](docs/V8_VALIDATION_REPORT.md) · [Reproducibility guide](docs/V8_REPRODUCIBILITY.md)

> **Research status.** This repository contains simulation code, frozen SQM codebooks, released numerical results, and additional V8 validation/comparison routines. The manuscript is not included in the public repository; publication details will be added when available. Reproduction scope and outstanding numerical differences are documented below.

## Method overview

SQM combines three elements:

1. **Static affine I/Q quotient.** Temporal centering removes a constant DC offset. The row space of a noiseless rank-two real I/Q block is unchanged by a common invertible real 2 × 2 left transformation.
2. **Selective nuisance representation.** Message-dependent low-rank explanation spaces represent CFO and frequency-selective fading. Small ranks can leave true signal energy unexplained; large ranks can make incorrect messages easier to explain.
3. **Incidence-aware codebook design.** Offline training discourages a channel-selected received plane from fitting a wrong message’s explanation space.

The affine invariance is a **noiseless algebraic identity**. It does not remove noise, recover information lost through distortion, or cover arbitrary time-varying hardware impairments. CFO is searched over rather than eliminated without computation.

## Selected results

**PER = packet error rate; lower is better.** Unless otherwise stated, “±” denotes a two-sided 95% Student-t interval half-width across three independent seed-level PERs, with two degrees of freedom. Results from different experiments are reported separately and must not be pooled.

### 1. Exact fractional-delay 3GPP TDL-A comparison

The main released experiment uses **8 information bits, 256 messages, 16 complex samples, mean block energy 16**, I/Q condition number 1.5, and 1,200 packets/SNR/seed over three seeds. Physical CFO spans ±4 degrees/sample; decoding uses a guarded −5 to +5 degree/sample grid with one-degree spacing.

| SNR | SQM, rank 4 | Practical pilot-aided OFDM | Full-oracle OFDM |
|---|---:|---:|---:|
| 5 dB | 0.2403 ± 0.0104 | 0.4472 ± 0.0592 | 0.0847 ± 0.0135 |
| 7 dB | 0.1586 ± 0.0256 | 0.3536 ± 0.0281 | 0.0539 ± 0.0221 |
| 10 dB | 0.0658 ± 0.0435 | 0.2264 ± 0.0509 | 0.0181 ± 0.0104 |
| 12 dB | 0.0367 ± 0.0062 | 0.1550 ± 0.0301 | 0.0081 ± 0.0024 |
| 15 dB | 0.0175 ± 0.0164 | 0.1047 ± 0.0181 | 0.0019 ± 0.0032 |

Source: [released main TDL-A results](Data/final_rerun_fractional_tdla_headtohead.csv). This table summarizes stored released results; it is not a new main-experiment execution performed during the V8 port validation.

**Interpretation:** SQM improves on the tested practical pilot-aided receiver, while full-oracle OFDM remains better. This supports a short-packet acquisition/calibration trade-off, not superiority to perfect-state coherent reception.

### 2. Stronger coded-OFDM controls at 10 dB

A separate shared-seed experiment compares SQM with stronger coded and uncalibrated receivers.

| Receiver | PER |
|---|---:|
| SQM, rank 4 | 0.0650 ± 0.0090 |
| Pilot-free coded OFDM, TDL-A-marginal ML | 0.0831 ± 0.0150 |
| Rate-matched pilot-aided coded OFDM | 0.2442 ± 0.0075 |
| Practical OFDM, unknown I/Q and DC | 0.3061 ± 0.0209 |

The paired difference **SQM minus pilot-free coded OFDM is −0.0181 ± 0.0240**. Its interval includes zero: these three seeds do **not** establish a statistically separated advantage over that strong pilot-free receiver.

Sources: [coded-control summary](Data/reviewer_baselines_summary.csv) and [paired differences](Data/reviewer_baselines_paired.csv). The SQM mean in this experiment must not be substituted into the separate main experiment above.

### 3. Five-scheme comparison: fresh V8 rerun

The following values are from the **fresh manuscript-budget V8 run included in the validation evidence**, not a transcription of every manuscript result. The matrix uses eight information bits and equal energy per bit, five independent seeds, and 600 packets/seed/SNR: **3,000 packets per scheme and condition at each SNR**.

**Mean PER at 10 dB Eb/N0:**

| Condition | SQM | HWI-Grassmannian | Cube-Split | SE-MOCZ | IM-MOCZ |
|---|---:|---:|---:|---:|---:|
| A — Clean flat Rayleigh fading | 0.1067 | **0.0833** | 0.1890 | 0.1230 | 0.2253 |
| B — Flat fading + CFO/I/Q/DC | 0.1107 | **0.1030** | 0.6903 | 0.1187† | 0.2170† |
| C — Clean two-tap multipath | **0.0980** | 0.1457 | 0.8683 | 0.2093 | 0.3233 |
| D — Two-tap multipath + CFO/I/Q/DC | **0.1087** | 0.1703 | 0.8300 | 0.2080† | 0.3387† |

Bold marks the lowest observed mean, **not** a declaration of statistical significance. Five-seed t-intervals are available in the [full matrix summary](Validation/v8_port_check/results/checked_matrix_manuscript/comparative_baselines_ablation/complete_1plus4_matrix/complete_1plus4_summary.csv).

**Comparison assumptions matter:**
- Native observation durations differ; equal payload and energy per bit do not establish equal spectral efficiency.
- HWI-Grassmannian is a single-stream source-inspired adaptation, not an exact reproduction of the original multi-antenna design.
- HWI-Grassmannian and Cube-Split receive matched rank-four selective-quotient decoding in the multipath panels.
- † SE-MOCZ and IM-MOCZ receive competitor-favorable oracle front-end-cleaned references in the hardware-stress panels. These are not blind full-impairment receivers.

**Numerical provenance:** 15 of the 20 rounded 10-dB means agree with the supplied draft. Four HWI means and the flat-HWI Cube-Split mean differ. The saved author-provided HTML reports using a cached HWI codebook; the checked rerun trained that codebook afresh. The cause of all differences has not been isolated. See the [direct comparison audit](Validation/v8_port_check/matrix_vs_manuscript.csv) and [validation report](docs/V8_VALIDATION_REPORT.md). No parameters were tuned to force numerical agreement.

The regime-dependent ordering is clear in this tested run: HWI-Grassmannian is stronger in flat fading; SQM has the lowest observed mean in the tested two-tap panels. This does **not** establish universal dominance.

### 4. Corrected adaptation and independent checks

| Check | Verified outcome |
|---|---|
| Corrected same-packet S6 refinement | PER **0.1853 ± 0.0424**, versus **0.1864 ± 0.0330** for fixed rank four; mean approximate cost **7.44 × 10⁵** versus **3.60 × 10⁵ MACs/packet** |
| S6 boundary adaptation | **7/9** successful boundaries; restricted delay **133.3 packets** |
| Independent coarse-to-fine decoding | Top-eight and full search both PER **0.0611 ± 0.0226**, with **zero observed disagreement in 2,700 packets** |
| Top-eight score count | **1,368 / 2,816 = 48.6%** of full search; **1,328 / 2,816 = 47.2%** if coarse scores are reused |
| All-message margin diagnostic | Over 500 cases and all 255 wrong messages, mean margin is largest at **r = 4 (0.4956)**; the 1% lower-tail margin is largest at **r = 6 (0.2956)** |
| Independent rank sweep | Rank four is preferred at 5/10 dB; the high-SNR optimum moves toward ranks five–six |

Sources and execution scope: [validation report](docs/V8_VALIDATION_REPORT.md) and [checked result files](Validation/v8_port_check/results/).

S6 offers no clear PER advantage over fixed rank four in this trace and requires substantially more computation. Adaptive boundary counts are descriptive diagnostics from nine events clustered within three seeds. All-message margins remain relative to a finite CFO grid, not continuous-orbit global certificates. Score counts do not establish measured hardware energy or latency gains.

## Installation

Use **Python 3.10+** and a virtual environment. The port was tested with Python 3.13; exact tested package versions are recorded in [validation evidence](Validation/v8_port_check/tested-package-versions.txt).

```bash
git clone https://github.com/Dhruvkumar25/Selective-Quotient-Modulation.git
cd Selective-Quotient-Modulation
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell — use this instead
.venv\Scripts\Activate.ps1
```

Install the validation dependencies and check the package:

```bash
python -m pip install -r requirements-validation.txt
python Code/run_validation_v8.py --check
python -m unittest discover -s tests -v
```

The prepared package passed **nine deterministic regression tests**, covering source provenance, affine invariance, rank saturation, corrected S6 semantics/cost, censoring, search counts, normalization, configuration, and output safeguards. This does not replace manuscript-scale Monte Carlo evaluation.

The V8 validation runner does not require PyTorch. Original reference-training scripts use the separate `requirements.txt`. Training a new codebook is not guaranteed to reproduce the released optimized artifact exactly.

## Run experiments

Run commands from the repository root. Start with a small execution check:

```bash
python Code/run_validation_v8.py --task matrix --mode smoke
```

**Smoke mode checks execution only.** It reduces packets and HWI training; its single-seed zero interval half-width is a placeholder, not evidence of certainty or publication-quality performance.

For the notebook's manuscript-budget settings:

```bash
# Corrected adaptive S6, boundary censoring and transient statistics
python Code/run_validation_v8.py --task adaptive --mode manuscript

# Complete five-scheme/four-condition matrix
python Code/run_validation_v8.py --task matrix --mode manuscript

# Matched fractional-TDL-A closest-comparator test
python Code/run_validation_v8.py --task closest-hwi --mode manuscript

# Rank/training factorial ablation
python Code/run_validation_v8.py --task ablation --mode manuscript

# Independent equation-based validation
python Code/run_validation_v8.py --task math
python Code/run_validation_v8.py --task intersection
python Code/run_validation_v8.py --task rank --mode manuscript
python Code/run_validation_v8.py --task coarse-fine --mode manuscript
python Code/run_validation_v8.py --task cfo --mode manuscript
python Code/run_validation_v8.py --task margins --mode manuscript
python Code/run_validation_v8.py --task sensitivity --mode manuscript
```

Use `--list` for all tasks, `--dry-run` to inspect a plan, or `--output Runs/my_new_run` to choose a **new** output directory. Adaptive/sensitivity tasks preserve the full notebook budgets and require manuscript mode. Full runs are more expensive than smoke checks; the notebook's named manuscript mode does not guarantee exact equality to every reported value.

Each run writes CSVs and a manifest containing source/input hashes, settings, dependency versions and a historical-input integrity check. Existing output directories are rejected. New outputs are not silently promoted to released data. Optional `--hwi-codebook PATH` accepts a compatible frozen HWI NPZ for comparison runs and records its hash; without it, the comparator is trained afresh.

Numerical CLI stages do not render the manuscript figures. The unchanged notebook retains the original plotting cells, which have notebook-order/filesystem dependencies. **Do not run the entire notebook against your only working copy:** its legacy sections include scripts that write directly to historical paths.

## Repository structure

| Path | Contents |
|---|---|
| `Code/` | Original evaluators plus the V8 task runner and comparison audit |
| `Code/validation_v8/stages/` | Named numerical stages ported from the V8 notebook |
| `Code/validation_v8/source_map.json` | Cell/line mapping, source hashes and explicit portability edits |
| `Code/config.json` | Shared configuration restored from notebook cell 14 |
| `Artifact/` | Released frozen SQM codebooks |
| `Data/` | Historical released CSVs, configurations and packet logs |
| `Notebooks/` | Original author-provided V8 notebook, preserved unchanged |
| `Validation/v8_port_check/` | Checked rerun evidence, tests, provenance and discrepancy audit |
| `docs/` | Reproducibility scope and detailed validation findings |
| `tests/` | Deterministic regression tests |
| `Runs/` | Generated outputs/caches; excluded from version control |

Legacy DMM/Incidence-DMM labels remain in historical filenames and method fields for compatibility. The archived earlier README is retained for provenance, not as the current execution guide.

## Reproducibility scope and limitations

This repository distinguishes **released-data summaries**, **corrected analysis**, **independent equation-based reruns**, and **adapted related-work comparisons**. They are not interchangeable forms of evidence.

- The legacy CFO-sensitivity curve retains a documented numerical provenance gap. The independent rerun reproduces the qualitative trend and the disclosed reconstruction, not the exact legacy levels.
- The fresh HWI matrix/depth-test results are not numerically identical to the draft. Their paired-significance pattern must not be substituted for the draft's pattern without review.
- Adaptive reanalysis uses released logs for the non-S6 schemes; the port validation did not freshly simulate every historical controller or rerun every legacy main comparison.
- Evidence is simulation-only. No SDR/RF performance, hardware energy/latency advantage, universally optimal rank, or robustness to arbitrary phase noise, timing/clock drift and PA nonlinearity is established.
- Three/five-seed experiments and the available packet budgets do not establish ultra-low error probabilities such as 10⁻⁵.

For exact task budgets, source assumptions and executed checks, consult the [reproducibility guide](docs/V8_REPRODUCIBILITY.md) and [validation report](docs/V8_VALIDATION_REPORT.md).

## Reporting issues and citation

When reporting an issue, include the **commit SHA**, command, Python/dependency versions, traceback and whether the result comes from released CSVs or a fresh simulation. Do not include credentials or private information.

Please identify this work by its title and authors until a public preprint or accepted-publication identifier is available; no journal DOI is assigned by this repository. Public availability is not itself a software reuse license. Code, data and third-party licensing terms should be checked before reuse.
