# Ported from supervisor V8 notebook; see source_map.json.

from pathlib import Path
import os, sys, math, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.linalg import toeplitz
# display is supplied by the CLI runner

# Comparative section is self-contained with respect to the released repository.
ROOT = Path.cwd()
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from fractional_tdla_core import (
    tdla_operators,
    weighted_nuisance_dictionary,
    exact_fractional_channel,
    random_tdla_coefficients,
)

COMP_DIR = RUN_DIR / "comparative_baselines_ablation"
COMP_DIR.mkdir(parents=True, exist_ok=True)

# "manuscript" is the intended final run. Change to "smoke" only for a quick test.
COMP_MODE = os.environ.get("SQM_COMPARISON_MODE", "manuscript").strip().lower()
COMP_FORCE = False

if COMP_MODE == "manuscript":
    COMP_SEEDS = [13, 59, 151]
    COMP_SNRS = [5, 7, 10, 12, 15]
    COMP_ABLATION_BLOCKS = 1400
    COMP_FLAT_BLOCKS = 1000
    COMP_MOCZ_BLOCKS = 600
    COMP_RADIUS_BLOCKS = 180
    COMP_UNIFIED_BLOCKS = 700
    HWI_NITER_CFO = 50
    HWI_NCFO = 20
    HWI_NITER_IQI = 20
else:
    COMP_SEEDS = [13]
    COMP_SNRS = [10]
    COMP_ABLATION_BLOCKS = 100
    COMP_FLAT_BLOCKS = 100
    COMP_MOCZ_BLOCKS = 80
    COMP_RADIUS_BLOCKS = 50
    COMP_UNIFIED_BLOCKS = 80
    HWI_NITER_CFO = 4
    HWI_NCFO = 6
    HWI_NITER_IQI = 4

T_CRIT_DF2 = 4.302652729911275

def t_halfwidth(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    # Manuscript mode uses 3 independent seeds -> df=2.
    if len(x) == 3:
        crit = T_CRIT_DF2
    else:
        from scipy.stats import t
        crit = float(t.ppf(0.975, len(x)-1))
    return crit * x.std(ddof=1) / np.sqrt(len(x))

def summarize_per_seed(df, group_cols=("snr_db", "method")):
    rows = []
    for key, g in df.groupby(list(group_cols), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        x = g["PER"].to_numpy(float)
        row.update(
            PER_mean=float(x.mean()),
            PER_95CI_halfwidth=float(t_halfwidth(x)),
            seeds=int(g["seed"].nunique()),
            blocks=int(g["blocks"].sum()),
        )
        rows.append(row)
    return pd.DataFrame(rows)

def save_current_figure(stem):
    png = COMP_DIR / f"{stem}.png"
    pdf = COMP_DIR / f"{stem}.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print("Saved:", png.relative_to(ROOT))
    print("Saved:", pdf.relative_to(ROOT))

source_scope = pd.DataFrame([
    ["SQM", "8 bits / 16 complex samples", "TDL-A + CFO + static I/Q + DC", "Released implementation"],
    ["Cube-Split", "CS(4,1): 256 symbols in T=4", "Flat block fading, SIMO; N=1 used here", "Exact source construction; optional 16-sample isometric embedding"],
    ["HWI-aware Grassmannian", "Source paper: K=64, M=2, T=4/8", "Flat block fading + CFO + TX I/Q imbalance", "Single-stream M=1, T=16, K=256 adaptation"],
    ["SE-MOCZ", "Source examples use K=7/40/70", "Frequency-selective multipath + FTN pulse model", "8-bit SE-BMOCZ source-model implementation, tau=0.9, beta=0.3"],
    ["IM-MOCZ", "Source equal-bit example N=10, K=8/6/4", "Frequency-selective multipath + AWGN", "Matched payload N=8, K=6 using source DiZeT/majority-vote equations"],
], columns=["scheme", "source/native setting", "native nuisance model", "comparative study role"])

print("=== Comparative benchmark configuration ===")
print("Run mode:", COMP_MODE)
print("Seeds:", COMP_SEEDS)
print("SNRs :", COMP_SNRS)
print("Output:", COMP_DIR)
display(source_scope)
source_scope.to_csv(COMP_DIR / "source_scope_and_fairness.csv", index=False)

