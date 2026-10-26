# Ported from supervisor V8 notebook; see source_map.json.

# Correct adaptation delay:
# measure ONLY after internal regime boundaries (segments 1, 2, 3),
# not from the initial segment 0.

packet_file = ROOT / "Data" / "online_adaptive_packet_results.csv"
packet = pd.read_csv(packet_file)

canonical = packet[
    packet["sequence"].eq("canonical")
].copy()

# Derive the held-out oracle target rank directly from S8.
oracle_targets = (
    canonical[
        canonical["method"].eq("S8_regime_oracle_rank")
    ]
    .groupby(["seed", "segment"])["selected_rank"]
    .first()
)

def adaptation_delay_boundaries_only(
    selected,
    segments,
    target_by_segment,
    *,
    window=10
):
    selected = np.asarray(selected, dtype=int)
    segments = np.asarray(segments, dtype=int)

    unique_segments = sorted(np.unique(segments))
    internal_segments = unique_segments[1:]  # skip segment 0

    delays = []

    for seg in internal_segments:
        ix = np.flatnonzero(segments == seg)

        if isinstance(target_by_segment, dict):
            target = int(target_by_segment[int(seg)])
        else:
            target = int(target_by_segment[int(seg)])

        hit = None

        for k in range(0, len(ix) - window + 1):
            if np.all(selected[ix[k:k+window]] == target):
                hit = k
                break

        delays.append(
            float(hit if hit is not None else len(ix))
        )

    return float(np.mean(delays)), delays


delay_rows = []

for (seed, method), g in canonical.groupby(
    ["seed", "method"],
    sort=False
):
    if method in ("S0_fixed_r4", "S1_best_fixed"):
        continue

    g = g.sort_values("packet")
    target_map = {
        int(seg): int(oracle_targets.loc[(seed, seg)])
        for seg in sorted(g["segment"].unique())
    }

    mean_delay, per_boundary = adaptation_delay_boundaries_only(
        g["selected_rank"].to_numpy(),
        g["segment"].to_numpy(),
        target_map,
        window=10
    )

    delay_rows.append({
        "seed": int(seed),
        "method": method,
        "adaptation_delay_v2": mean_delay,
        "boundary_1_delay": per_boundary[0],
        "boundary_2_delay": per_boundary[1],
        "boundary_3_delay": per_boundary[2],
    })

delay_v2 = pd.DataFrame(delay_rows)

delay_summary_v2 = (
    delay_v2
    .groupby("method", sort=False)["adaptation_delay_v2"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

delay_summary_v2["95CI_halfwidth_t_df2"] = (
    T_CRIT_DF2
    * delay_summary_v2["std"]
    / np.sqrt(delay_summary_v2["count"])
)

# Compare with the released canonical seed-level metric.
released_seed_metrics = pd.read_csv(
    ROOT / "Data" / "online_adaptive_revision_seed_metrics.csv"
)
released_canonical = released_seed_metrics[
    released_seed_metrics["sequence"].eq("canonical")
][
    ["seed", "method", "mean_adaptation_delay_packets"]
].copy()

delay_compare = released_canonical.merge(
    delay_v2[["seed", "method", "adaptation_delay_v2"]],
    on=["seed", "method"],
    how="inner"
)

delay_compare["difference_v2_minus_released"] = (
    delay_compare["adaptation_delay_v2"]
    - delay_compare["mean_adaptation_delay_packets"]
)

delay_v2.to_csv(
    V2_DIR / "adaptive_delay_corrected_per_seed.csv",
    index=False
)
delay_summary_v2.to_csv(
    V2_DIR / "adaptive_delay_corrected_summary.csv",
    index=False
)

print("=== Corrected per-seed adaptation delay ===")
display(delay_v2)

print("\n=== Corrected three-seed summary ===")
display(delay_summary_v2)

print("\n=== Released vs corrected ===")
display(delay_compare)

print("\nExpected corrected canonical means:")
for _, r in delay_summary_v2.iterrows():
    print(f"{r['method']:38s} {r['mean']:.6f} packets")

