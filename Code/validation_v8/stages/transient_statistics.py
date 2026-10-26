# Ported from supervisor V8 notebook; see source_map.json.

# Correct Figure 14 statistical unit:
# 1) average the three transitions within each seed;
# 2) treat the three seeds as the independent replicates;
# 3) use t_{0.975,2} for the 95% half-width.

import matplotlib.pyplot as plt

transient = pd.read_csv(
    ROOT / "Data" / "online_transient_seed_metrics.csv"
)

transient = transient[
    transient["sequence"].eq("canonical")
].copy()

# Average transitions 1,2,3 within each seed first.
transient_seedavg = (
    transient
    .groupby(
        ["sequence", "seed", "method", "window"],
        sort=False,
        as_index=False
    )["PER"]
    .mean()
    .rename(columns={"PER": "PER_seed_transition_mean"})
)

transient_v2 = (
    transient_seedavg
    .groupby(
        ["sequence", "method", "window"],
        sort=False
    )["PER_seed_transition_mean"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

transient_v2["PER_95CI_halfwidth_t_df2"] = (
    T_CRIT_DF2
    * transient_v2["std"]
    / np.sqrt(transient_v2["count"])
)

transient_seedavg.to_csv(
    V2_DIR / "online_transient_seed_averages_v2.csv",
    index=False
)
transient_v2.to_csv(
    V2_DIR / "online_transient_summary_v2.csv",
    index=False
)

print("=== Corrected Figure 14 summary ===")
display(transient_v2)

