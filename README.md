# Selective-Quotient Modulation (SQM)

Research code for **“Selective-Quotient Modulation for Calibration-Light Short-Packet Wireless Links”**, by Dhruv Kumar and Michael Georgiades.

## V8 notebook → GitHub update

This update preserves the supervisor's original V8 notebook and adds command-line entry points for its corrected analyses, independent validation, ablation, and five-scheme comparisons. Historical `Code/`, `Data/` and `Artifact/` files are retained; new experiment outputs go to **`Runs/`**, never silently over the released results.

**Important: this is a reproducibility update, not a claim that every final-draft number has been regenerated.** The corrected S6 results were reproduced in the tested environment. A fresh manuscript-scale five-scheme run matches the key SQM/SE-MOCZ/IM-MOCZ cross-section, but HWI-Grassmannian values differ from the PDF. The saved HTML reports loading a cached HWI codebook; that cached file was not supplied with the notebook. See [validation report](docs/V8_VALIDATION_REPORT.md). Do not replace manuscript numbers with fresh-run values without author review.

## Start here

Python 3.10+; validation performed with Python 3.13. Use a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements-validation.txt
python Code/run_validation_v8.py --check
python -m unittest discover -s tests -v
```

A short execution check (not a performance estimate):

```bash
python Code/run_validation_v8.py --task matrix --mode smoke
```

One seed cannot estimate between-seed uncertainty. The notebook's single-seed zero half-width is retained for traceability; it does **not** mean certainty. Smoke mode also reduces HWI training and must not be used as evidence for the paper.

## Run the corrected/final-scale experiments

```bash
# Corrected same-packet S6, boundary success/censoring, transient statistics:
python Code/run_validation_v8.py --task adaptive --mode manuscript

# Five schemes × four conditions × five SNRs × five seeds:
python Code/run_validation_v8.py --task matrix --mode manuscript

# Independent validation and ablation:
python Code/run_validation_v8.py --task math
python Code/run_validation_v8.py --task intersection
python Code/run_validation_v8.py --task rank --mode manuscript
python Code/run_validation_v8.py --task cfo --mode manuscript
python Code/run_validation_v8.py --task coarse-fine --mode manuscript
python Code/run_validation_v8.py --task margins --mode manuscript
python Code/run_validation_v8.py --task ablation --mode manuscript
python Code/run_validation_v8.py --task closest-hwi --mode manuscript
python Code/run_validation_v8.py --task sensitivity --mode manuscript
```

Use `--list` to see tasks, `--dry-run` to inspect a plan, and `--output Runs/my_new_run` to choose a **new** output directory. Existing directories are rejected. Adaptive/sensitivity preserve all notebook validation/final seeds and therefore require manuscript mode. Numerical stages produce CSVs and a manifest recording settings, dependency versions, source/input hashes and input-integrity checks. Figure-rendering cells are not run by these commands; the unchanged notebook preserves them.

If the supervisor supplies the cached HWI codebook:

```bash
python Code/run_validation_v8.py --task matrix --mode manuscript \
  --hwi-codebook /path/to/hwi_aware_grassmann_single_stream_codebook.npz
```

The file is validated and copied only into the new run directory. No automatic tuning or historical-artifact replacement is done.

## Repository layout

| Directory/file | Purpose |
|---|---|
| `Notebooks/SQM_Complete_Validation_Comparisons_Ablation_V8.ipynb` | Original supervisor notebook, unchanged |
| `Code/run_validation_v8.py` | Safe task runner |
| `Code/validation_v8/stages/` | Named Python stages extracted from V8, preserving numerical routines |
| `Code/validation_v8/source_map.json` | Notebook cell/line mapping, hashes and explicit portability edits |
| `Code/config.json` | Missing configuration restored from V8 cell 14 |
| `Code/` (other original scripts) | Historical implementation and evaluators |
| `Data/`, `Artifact/` | Unchanged released results and frozen SQM codebooks |
| `Runs/` | Generated results/caches; ignored by Git |
| `Manuscript/Selective_Quotient_Modulation_Comparative_Revision_Blue.pdf` | Supplied 16-page comparative draft, not a newly compiled PDF |
| `tests/` | Small deterministic regression checks |
| `docs/` | Run guide, validation report, remaining author decisions |

The final comparative LaTeX/Overleaf source was **not** supplied; the older V4 source is not presented as the final source. This repository therefore does not yet provide a build of the comparative PDF from LaTeX.

## What the comparisons establish

SQM is calibration-light in the tested short-packet regimes, not universally superior. Its exact affine invariance is noiseless; noise and information loss remain. Full-oracle OFDM is better. Clean flat fading favors the HWI comparator in the final draft. The new matrix uses equal payload/energy per bit but **different native observation durations**. HWI/Cube multipath receivers are selective-quotient adaptations; MOCZ-family hardware-stress references receive oracle front-end cleanup. They are not blind full-impairment reproductions.

The legacy CFO-sensitivity numerical provenance gap remains explicitly documented. Corrected S6 has similar PER to fixed rank four at much greater computation. Grid-based/all-message margins are diagnostics, not continuous-orbit global certificates. See [reproducibility guide](docs/V8_REPRODUCIBILITY.md).

The previous README is archived in [docs/README_BEFORE_V8.md](docs/README_BEFORE_V8.md), for history only; its unshipped paths/build instructions are not the current recipe. Legacy DMM names remain for compatibility. Reference PyTorch training still uses the original `requirements.txt`; the validation runner does not require PyTorch.

No new license or publication DOI is assigned here. Authors should review third-party rights and select appropriate reuse terms before release.
