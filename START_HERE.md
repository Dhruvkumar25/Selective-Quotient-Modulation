# Start here — simple handoff

1. **Read `docs/V8_VALIDATION_REPORT.md`.** The corrected adaptive results match. Some HWI comparison values still differ; no manuscript/data values have been changed to hide this.
2. The original supervisor notebook is in **`Notebooks/`**. New commands and installation instructions are in **`README.md`**. The supplied final PDF is in **`Manuscript/`**.
3. Ask the supervisor for **`hwi_aware_grassmann_single_stream_codebook.npz`**, the exact comparison/depth-test CSVs used in the final PDF, and the matching comparative Overleaf source. A short email is provided below.
4. Review locally, then update GitHub. **Nothing has been pushed yet.** The code can be uploaded as a reproducibility update with the documented caveat; do not claim every comparative number is independently reproduced yet.

Suggested email:

> Thank you, sir. I have organized the V8 notebook into runnable GitHub scripts and checked the corrected adaptive results. A fresh full comparison reproduces the SQM and MOCZ values, but some HWI values differ. The HTML reports loading a cached HWI codebook. Could you please share `hwi_aware_grassmann_single_stream_codebook.npz`, the exact matrix/TDL-A comparison CSVs, and the latest comparative Overleaf source? There is also a small Cube-Split flat-HWI difference (0.69033 in the fresh run versus 0.6910 in the draft), which I would like to check before finalizing the release.

## If using Git locally

Copy the extracted package contents into a clean clone of your repository, **excluding any `Runs/` folder**. Check the changes before committing:

```bash
git status --short
git diff -- README.md
python -m pip install -r requirements-validation.txt
python Code/run_validation_v8.py --check
python -m unittest discover -s tests -v
```

New files/folders to review: `.github/`, `.gitignore`, `Code/config.json`, `Code/run_validation_v8.py`, `Code/audit_v8_cross_section.py`, `Code/validation_v8/`, `Notebooks/`, `Manuscript/`, `docs/`, `tests/`, `Validation/`, `requirements-validation.txt`, and this guide. Existing `Data/` and `Artifact/` must remain unchanged. Commit/push only after your review; no credentials belong in these files.
