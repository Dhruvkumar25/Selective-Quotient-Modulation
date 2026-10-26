# V8 reproducibility guide

## Provenance and design

Base repository: commit `218b7899f31b2182c172e893afcbd206a113cf82`.
The supplied supervisor V8 notebook is preserved byte-for-byte under `Notebooks/`.
`Code/validation_v8/source_map.json` identifies every extracted stage by zero-based notebook cell, inclusive source lines, original-cell SHA-256 and ported-stage SHA-256. Numerical algorithms, RNG seeds and manuscript-mode budgets are not retuned. The runner executes named stages in an explicit shared namespace because the notebook uses shared globals. Individual stage files are not standalone entry points.

Portability changes:
- Replace Jupyter `display` import with bounded console display supplied by the runner.
- Direct comparative outputs to the chosen `Runs/` workspace; corrected/independent stages receive the same workspace through `RESULTS`.
- Read original margin references directly from unchanged `Data/` instead of requiring a prior notebook backup step.
- Add a smoke-only environment control for margin cases; manuscript default remains 500/all 255 wrong messages.
- Separate numerical/statistical blocks from plotting blocks; retain original plots only in the unchanged notebook. Equation/figure labels inside extracted source are historical notebook labels, not automatically renumbered final-PDF references.
- Explicitly restore missing imports required by independent helpers, and retain `method_order` used by closest-comparator paired statistics even though its adjacent plot is omitted.
- Restore `Code/config.json` exactly from the notebook cell 14 dictionary. No original tracked code/data/artifact is edited.

The notebook's Part I is **not** automatically executed: it contains legacy paths, optional retraining, and evaluators that write directly to `Data/` or `Artifact/`. Do not use “Run all” against your only working copy. Use the CLI for the supported stages.

## Budgets

| Task | Manuscript settings | Smoke settings |
|---|---|---|
| `matrix` | 5 seeds × 600 packets × 5 SNRs × 4 panels × 5 schemes | 1 seed × 60 packets at 10 dB, reduced HWI training |
| `closest-hwi` | 3 final seeds × 800 packets/SNR; 2 held-out tuning seeds × 220 | 1 final seed × 80; 2 tuning seeds × 40 |
| `ablation` | 3 seeds × 1400 packets/SNR, 4 configurations | 1 seed × 100 packets at 10 dB |
| `rank` | 3 seeds × 1200 packets, SNR 5/10/15, ranks 2/3/4/5/6/8 | 3 seeds × 8 packets, same rank/SNR grid |
| `cfo` | 3 seeds × 2500 packets/support | 3 seeds × 8 packets/support |
| `coarse-fine` | 3 seeds × 900 packets | 3 seeds × 8 packets |
| `margins` | 500 cases, all 255 wrong messages | 2 cases, all 255 wrong messages |
| `adaptive` | Original 3 validation + 3 final seeds; 300 packets/regime in canonical final traces | Not offered |
| `sensitivity` | Original notebook sensitivity validation/final traces | Not offered |
| `math`, `intersection`, `transient` | Fixed equation checks / 100 intersections per length / reanalysis of released transient CSV | Same fixed checks/reanalysis |

Modes are isolated by fresh directories. HWI training in smoke mode is materially different, not a cheap estimator of the manuscript result. All 14 task names can be listed with `--list` (including separate Cube/HWI flat diagnostics).

## Interpretation and remaining gaps

- The new matrix is 8 information bits with energy 8, i.e. energy per bit 1. Do not mix its Eb/N0/observation durations with the separate 16-sample, energy-16 main TDL-A results.
- SE/IM oracle cleanup is a favorable *clean front-end reference* as implemented by the notebook, not a demonstrated blind calibration algorithm. Noisy inverse-I/Q cleanup is not silently substituted.
- Seed-level t intervals are used with the notebook's declared budgets. One-seed comparison smoke runs retain half-width zero only as a source-compatible placeholder.
- Nine adaptive boundaries are clustered within three seeds. Success counts are descriptive. Transient windows average boundaries within seed first.
- Existing legacy results are inputs, not new executions. Corrected adaptive S6 is regenerated; other adaptive scheme rows are reanalyzed from released logs. This is not a fresh run of every controller.
- The CFO task quantifies the already disclosed legacy numerical gap. Do not tune undocumented assumptions to remove it.
- A freshly trained HWI codebook does not match the draft cross-section in the tested environment. The supplied HTML reports a cached codebook. Obtain that exact NPZ, hash, training settings and corresponding result CSVs before claiming exact comparative reproduction. The script accepts an explicit supplied NPZ and records its hash.
- Final comparative Overleaf/LaTeX source is still needed for a source-to-PDF build.

## Practical safeguards

Every output directory must be new and inside `Runs/`. The runner records input and source hashes, dependency versions, mode and relevant environment controls. It checks `Data/` and `Artifact/` hashes after both success and failure. Existing caches are never silently borrowed from a different mode/run. This is an integrity check, not an operating-system sandbox. It is intentionally not a replacement for reviewing arbitrary future code changes.

Reference helper imports require NumPy/Pandas/SciPy; Matplotlib is retained as a notebook dependency even though CLI plotting blocks are omitted. PyTorch is only needed for original training scripts, not the added validation tasks. The supplied notebook still has original Jupyter dependencies and filesystem assumptions; the CLI is the recommended portable path.

The cached-HWI observation does not by itself prove why a fresh HWI run differs. Cached artifact identity, training history, and numerical environment are not yet isolated. Request the frozen artifact rather than treating the discrepancy as evidence that the source paper or manuscript is wrong.
