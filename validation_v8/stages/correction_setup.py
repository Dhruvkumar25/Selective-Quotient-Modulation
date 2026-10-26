# Ported from supervisor V8 notebook; see source_map.json.

# V2 correction workspace
from pathlib import Path
import sys, os, json, math, time
import numpy as np
import pandas as pd

V2_DIR = RESULTS / "v2_corrected"
V2_DIR.mkdir(parents=True, exist_ok=True)

# Make repository code importable without installing anything.
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

T_CRIT_DF2 = 4.302652729911275

print("=== V2 Corrected-Analysis Workspace ===")
print("Repository :", ROOT)
print("V2 outputs :", V2_DIR)
print("Original source files will NOT be modified.")

