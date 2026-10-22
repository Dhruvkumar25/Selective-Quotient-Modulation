"""Single source of truth for the SQM revision experiments."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
with open(Path(__file__).with_name("config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

METHOD_NAME = CONFIG["method_name"]
M = int(CONFIG["messages"])
L = int(CONFIG["block_length"])
NFFT = int(CONFIG["nfft"])
CP = int(CONFIG["cp_length"])
PILOT_BINS = np.asarray(CONFIG["pilot_bins"], dtype=int)
DATA_BINS = np.asarray(CONFIG["data_bins"], dtype=int)
PHYSICAL_CFO_MIN, PHYSICAL_CFO_MAX = CONFIG["physical_cfo_support_deg_per_sample"]
GRID_MIN, GRID_MAX, GRID_STEP = CONFIG["cfo_search_grid_deg_per_sample"]


def cfo_grid_deg(min_deg=GRID_MIN, max_deg=GRID_MAX, step_deg=GRID_STEP):
    """Return the explicitly documented CFO search grid."""
    n = int(round((max_deg - min_deg) / step_deg))
    return np.linspace(min_deg, max_deg, n + 1)


def cfo_grid_rad(min_deg=GRID_MIN, max_deg=GRID_MAX, step_deg=GRID_STEP):
    return np.deg2rad(cfo_grid_deg(min_deg, max_deg, step_deg))


def physical_cfo_sample(rng, size=None):
    return rng.uniform(np.deg2rad(PHYSICAL_CFO_MIN), np.deg2rad(PHYSICAL_CFO_MAX), size)


def label(name: str) -> str:
    """Normalize legacy artifact labels for manuscript-facing output."""
    replacements = {
        "Incidence-trained SQM": "Incidence-trained SQM",
        "Selective SQM": "Selective SQM",
        "SQM manifold packing": "SQM manifold packing",
        "DMM": "SQM",
    }
    return replacements.get(name, name)
