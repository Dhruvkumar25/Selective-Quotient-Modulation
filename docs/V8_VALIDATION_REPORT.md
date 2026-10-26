# V8 port validation report

## Outcome

The supervisor's notebook has been preserved unchanged and its numerical additions exposed through a safe CLI. **Execution succeeds for the supported task paths, but exact agreement with all final comparative-draft numbers is not established.** No remote GitHub update was performed during this preparation.

Historical tracked `Data/`, `Artifact/` and original Python files were not changed. The old README was archived and replaced with instructions matching the files actually shipped. The missing configuration was restored from notebook cell 14. New runs are separate, not silently promoted to manuscript data.

## Checks actually executed

- Stage compilation/dependency check: passed.
- **9 deterministic regression tests: passed.** These cover source hashes, affine invariance, rank saturation, current-packet S6 decisions and complete candidate cost, censoring, coarse-to-fine counts, energy normalization, config restoration and unsafe-path/mode rejection.
- Equation audit and 100-trial-per-length constructive intersection experiment: passed the notebook's executable assertions.
- Full corrected adaptive S6 plus boundary-censoring and transient reanalysis: completed.
- Full sensitivity rerun: completed; means match the released sensitivity means while corrected intervals/delay definitions differ as intended.
- Full independent rank, CFO and coarse-to-fine experiments: completed.
- Full 500-case/all-255-wrong-message margin analysis: completed.
- Full five-scheme matrix: completed with five seeds, 600 packets/seed/SNR, all five SNRs/four panels.
- Full closest-HWI experiment, its held-out rank tuning, and the four-configuration ablation dependency: completed.
- Separate Cube-flat and HWI-flat diagnostic paths: smoke-tested only, not full-budget rerun.
- Every completed run's post-run integrity check found its historical input hashes unchanged.

Early development smoke runs caught two missing notebook-global dependencies (core imports and the closest-comparator method-order list). Both wiring errors were fixed. Affected independent stages and the full closest-comparator task were then successfully rerun. Failed development outputs are not included as validation evidence.

This does **not** include fresh execution of the legacy main OFDM comparison, every historical controller, stochastic SQM retraining, RF hardware tests, or a LaTeX build of the final comparative draft.

## Reproduced key results

| Check | Fresh result | Relation to supplied draft |
|---|---|---|
| Corrected S6, 3 seeds | PER 0.185278 ± 0.042442 (95% t half-width); mean MACs 743774.44/packet | Matches reported 0.1853 ± 0.0424 and 7.44×10^5 |
| S6 boundary events | 7/9 successes; conditional successful-event delay 85.7143; restricted delay 133.3333 packets | Matches reported rounded values |
| S7 boundary reanalysis | 6/9 successes; conditional delay 93.6667; restricted delay 162.4444 | Matches reported rounded values; uses released packet log |
| Independent rank sweep | Best screened ranks 4 at 5/10 dB, 5 at 15 dB; rank-5 PER 0.014444 at 15 dB | Supports 5–6 region, not unique rank-six optimum |
| Independent top-eight/full decoder | Both PER 0.061111 ± 0.022593, zero disagreement in 2700 packets | Matches reported independent screen |
| All-message margins | Mean margin r4 0.495583; 1% margin r6 0.295637; r4/5/6/8 positive in all 500 cases | Matches reported rounded values; remains grid-relative |
| Independent CFO sweep | PER 0.0340, 0.041333, 0.0424, 0.052133, 0.0580; mean absolute gap from released curve 0.0152267 | Reproduces the already disclosed independent reconstruction, **not** the legacy curve |

Intervals are based on the notebook's seed-level methodology, not independent-packet intervals. Boundary fractions remain descriptive clustered diagnostics.

## Comparative results that still need author review

The checked matrix matches **15 of the 20 rounded Table VI PER means**. All four SQM, all four SE-MOCZ, all four IM-MOCZ and three Cube-Split means agree to the draft's four decimal places. It does not match four HWI values and one Cube-Split value:

| 10-dB condition/method | Draft Table VI | Fresh manuscript-budget run |
|---|---:|---:|
| A clean flat / HWI | 0.0793 | 0.083333 |
| B flat+HWI / HWI | 0.1050 | 0.103000 |
| C clean two-tap / HWI | 0.1500 | 0.145667 |
| D two-tap+HWI / HWI | 0.1550 | 0.170333 |
| B flat+HWI / Cube-Split | 0.6910 | 0.690333 |

The saved supplied HTML explicitly reports **“Loaded cached single-stream HWI-aware Grassmannian codebook.”** The supplied IPYNB has code but no saved cell outputs; no corresponding frozen HWI NPZ was attached. The port trains it afresh using the notebook's manuscript settings. That is an artifact/provenance issue to resolve, **not proof of its cause**: cached identity/training history and numerical-environment effects have not been isolated. No settings were tuned to force agreement.

The separate closest-comparator run also differs on HWI values: held-out tuning again selects rank 3, but at 10 dB HWI+quotient gives 0.087917 instead of the draft's 0.0817; SQM gives 0.066667 as reported. At 15 dB HWI gives 0.021667 instead of 0.0229; SQM gives 0.0150. Consequently this fresh run's paired-significance pattern must **not** be substituted for the draft's pattern (the fresh adapted-HWI intervals exclude zero at 5/7/10 dB, not 12/15 dB).

Request the exact cached HWI NPZ and associated per-seed comparison/depth-test CSVs, with package versions and generation settings. Also check the small Cube-Split B discrepancy. The CLI accepts a supplied HWI NPZ and records its hash. Keep the final PDF unchanged until the authors decide which artifact/result set is authoritative.

## Evidence and environment

Selected successful-run CSVs and manifests are under `Validation/v8_port_check/results/`; numerical cache banks and development-run folders are omitted to keep the handoff small. Original manifest paths describe the execution layout under `Runs/`, not the copied evidence layout. Omitted cache file hashes remain recorded for traceability. `matrix_vs_manuscript.csv` is a direct rounded-mean audit, not a statistical test. Its source-reference JSON identifies the supplied PDF and its SHA-256.

`tested-package-versions.txt` records the actual package versions. Validation used Linux/Python 3.13, single-threaded BLAS settings and the original frozen SQM artifacts. Portability to every platform/package combination has not been established. New training artifacts need not be numerically identical on other numerical-library builds.

The final 16-page PDF is supplied as received. Its matching comparative Overleaf/LaTeX source is still needed; the earlier V4 source is not a substitute.
