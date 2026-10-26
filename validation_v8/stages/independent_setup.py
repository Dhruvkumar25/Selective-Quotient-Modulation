# Ported from supervisor V8 notebook; see source_map.json.
# independent validation workspace and run controls
from pathlib import Path
import os, math, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# display is supplied by the CLI runner

IND_DIR = RESULTS / "independent_validation"
IND_FIG_DIR = IND_DIR / "figures"
IND_DIR.mkdir(parents=True, exist_ok=True)
IND_FIG_DIR.mkdir(parents=True, exist_ok=True)

IND_T_CRIT_DF2 = 4.302652729911275
IND_SEEDS = [13, 59, 151]

# Defaults are manuscript-scale or near-manuscript-scale, but can be overridden
# from the shell before launching Jupyter if a faster smoke run is desired.
IND_RANK_BLOCKS_PER_SEED = int(os.getenv("SQM_IND_RANK_BLOCKS", "1200"))
IND_CFO_BLOCKS_PER_SEED = int(os.getenv("SQM_IND_CFO_BLOCKS", "2500"))
IND_CTF_BLOCKS_PER_SEED = int(os.getenv("SQM_IND_CTF_BLOCKS", "900"))


def ind_t_halfwidth(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(IND_T_CRIT_DF2 * x.std(ddof=1) / np.sqrt(len(x)))


def ind_savefig(fig, stem):
    png = IND_FIG_DIR / f"{stem}.png"
    pdf = IND_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png.relative_to(ROOT))
    print("Saved:", pdf.relative_to(ROOT))
    return png, pdf

print("independent validation output directory:", IND_DIR)
print("Rank sweep blocks/seed:", IND_RANK_BLOCKS_PER_SEED)
print("CFO sweep blocks/seed :", IND_CFO_BLOCKS_PER_SEED)
print("Coarse/fine blocks/seed:", IND_CTF_BLOCKS_PER_SEED)

