# Ported from supervisor V8 notebook; see source_map.json.

# The released Figure-1 diagnostic sampled a subset of wrong messages.
# V2 adds a stricter screen over ALL 255 wrong messages for each true message.
#
# This does not overwrite the released Figure-1 data.  It provides an
# all-message diagnostic under the same evaluator/CFO grid.

import matplotlib.pyplot as plt
import subprocess
import sys
from tqdm.auto import tqdm

MARGIN_V2 = V2_DIR / "selective_margin_all_wrong"
MARGIN_V2.mkdir(parents=True, exist_ok=True)

margin_all_file = (
    MARGIN_V2
    / "selective_margin_samples_all_255_wrong_v2.csv"
)
margin_log = (
    MARGIN_V2
    / "selective_margin_all_255_wrong_v2.log"
)

cmd = [
    sys.executable,
    "Code/evaluate_selective_margin.py",
    "--cases", os.environ.get("SQM_MARGIN_CASES", "500"),
    "--wrong", "255",
    "--ranks", "2,3,4,5,6,8",
    "--seed", "20260918",
    "--output",
    str(margin_all_file.relative_to(ROOT)),
]

print("Running stricter all-message selective-margin screen:")
print(" ".join(cmd))

start = time.time()

with open(margin_log, "w") as log:
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    with tqdm(
        desc="All-255-wrong margin screen",
        unit="s",
        bar_format="{desc}: {n:.0f} s elapsed",
    ) as pbar:
        while process.poll() is None:
            time.sleep(1)
            pbar.update(1)

return_code = process.returncode
elapsed = time.time() - start

print("Return code :", return_code)
print(f"Runtime     : {elapsed:.1f} s")
print("Output      :", margin_all_file)
print("Log         :", margin_log)

if return_code != 0:
    print(margin_log.read_text()[-5000:])
    raise RuntimeError(
        "All-message selective-margin screen failed."
    )

margin_all = pd.read_csv(margin_all_file)

# Radius associated with the measured all-message selective margin.
margin_all["selective_margin_radius_norm"] = (
    margin_all["selective_margin_norm"] / 2.0
)

margin_all_summary = (
    margin_all
    .groupby("rank", as_index=False)
    .agg(
        mean_true_mismatch=(
            "true_mismatch_norm", "mean"
        ),
        p90_true_mismatch=(
            "true_mismatch_norm",
            lambda x: x.quantile(0.90)
        ),
        mean_wrong_distance=(
            "nearest_wrong_dist_norm", "mean"
        ),
        p01_wrong_distance=(
            "nearest_wrong_dist_norm",
            lambda x: x.quantile(0.01)
        ),
        mean_selective_margin=(
            "selective_margin_norm", "mean"
        ),
        p01_selective_margin=(
            "selective_margin_norm",
            lambda x: x.quantile(0.01)
        ),
        frac_positive_margin=(
            "selective_margin_norm",
            lambda x: float((x > 0).mean())
        ),
        mean_margin_radius=(
            "selective_margin_radius_norm",
            "mean"
        ),
    )
)

margin_all_summary.to_csv(
    MARGIN_V2
    / "selective_margin_summary_all_255_wrong_v2.csv",
    index=False
)

print("\n=== All-message selective-margin summary ===")
display(margin_all_summary)

# Compare against released sampled-wrong-message summary.
released_margin_summary = pd.read_csv(
    ROOT
    / "Data"
    / "selective_margin_summary.csv"
)

margin_summary_compare = (
    released_margin_summary.merge(
        margin_all_summary,
        on="rank",
        suffixes=("_released_sampled", "_all255_v2")
    )
)

print("\n=== Released sampled vs all-255 V2 ===")
display(margin_summary_compare)

