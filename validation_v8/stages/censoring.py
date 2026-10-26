# Ported from supervisor V8 notebook; see source_map.json.
# The old mean delay encoded failures as a full-segment delay, hiding whether
# the controller ever reached the target. independent validation separates event occurrence from time.

released_packet = pd.read_csv(ROOT / "Data" / "online_adaptive_packet_results.csv")
packet_ind = released_packet[released_packet["sequence"].eq("canonical")].copy()

# Prefer the corrected same-packet S6 generated in Part II when available.
s6_v2_path = V2_DIR / "s6_same_packet_packet_results_v2.csv"
if s6_v2_path.exists():
    s6_corr = pd.read_csv(s6_v2_path)
    s6_corr = s6_corr[s6_corr["sequence"].eq("canonical")].copy()
    packet_ind = packet_ind[~packet_ind["method"].eq("S6_confidence_triggered_refinement")]
    # Align the released packet schema; missing controller_switch is safely filled.
    if "controller_switch" not in s6_corr.columns:
        s6_corr["controller_switch"] = s6_corr.get("selected_rank_transition", 0)
    packet_ind = pd.concat([packet_ind, s6_corr[packet_ind.columns]], ignore_index=True)
    s6_source = "corrected V2 same-packet S6"
else:
    s6_source = "released S6 fallback (run Part II first for corrected S6)"

oracle = (
    packet_ind[packet_ind["method"].eq("S8_regime_oracle_rank")]
    .groupby(["seed", "segment"])["selected_rank"]
    .first()
)


def ind_adaptation_events(df, window=10):
    rows = []
    for (seed, method), g in df.groupby(["seed", "method"], sort=False):
        g = g.sort_values("packet")
        segments = sorted(g["segment"].unique())
        for seg in segments[1:]:  # true regime boundaries only
            q = g[g["segment"].eq(seg)].sort_values("packet")
            selected = q["selected_rank"].to_numpy(dtype=int)
            target = int(oracle.loc[(seed, seg)])
            hit = None
            for k in range(0, len(selected) - window + 1):
                if np.all(selected[k:k+window] == target):
                    hit = k
                    break
            success = hit is not None
            seg_len = len(selected)
            rows.append({
                "seed": int(seed),
                "method": method,
                "segment": int(seg),
                "target_rank": target,
                "segment_packets": int(seg_len),
                "success": bool(success),
                "censored": bool(not success),
                "delay_to_run_start": float(hit) if success else np.nan,
                "delay_to_run_confirmation": float(hit + window - 1) if success else np.nan,
                # Restricted time-to-adaptation: failures contribute the observation horizon.
                "restricted_delay": float(hit) if success else float(seg_len),
                "censor_time": float(seg_len),
            })
    return pd.DataFrame(rows)

adapt_events_ind = ind_adaptation_events(packet_ind, window=10)
adapt_events_ind.to_csv(IND_DIR / "adaptation_events_ind.csv", index=False)

# Primary adaptive methods; fixed ranks are retained in the event CSV as negative controls.
adaptive_methods = [
    "S2_SNR_class_lookup",
    "S3_window_confidence",
    "S4_window_confidence_residual",
    "S5_EWMA_confidence_residual",
    "S6_confidence_triggered_refinement",
    "S7_ACK_UCB_optional",
    "S8_regime_oracle_rank",
]

seed_summary_rows = []
for (seed, method), g in adapt_events_ind[adapt_events_ind["method"].isin(adaptive_methods)].groupby(["seed", "method"], sort=False):
    succ = g[g["success"]]
    seed_summary_rows.append({
        "seed": int(seed),
        "method": method,
        "boundaries": len(g),
        "success_rate": float(g["success"].mean()),
        "successful_boundaries": int(g["success"].sum()),
        "censored_boundaries": int(g["censored"].sum()),
        "conditional_delay_successes": float(succ["delay_to_run_start"].mean()) if len(succ) else np.nan,
        "restricted_delay": float(g["restricted_delay"].mean()),
    })
seed_adapt_ind = pd.DataFrame(seed_summary_rows)

summary_rows = []
for method, g in seed_adapt_ind.groupby("method", sort=False):
    raw = adapt_events_ind[adapt_events_ind["method"].eq(method)]
    success_seed = g["success_rate"].to_numpy(float)
    restricted_seed = g["restricted_delay"].to_numpy(float)
    cond_seed = g["conditional_delay_successes"].dropna().to_numpy(float)
    summary_rows.append({
        "method": method,
        "boundary_events": int(len(raw)),
        "successes": int(raw["success"].sum()),
        "censored": int(raw["censored"].sum()),
        "pooled_success_rate": float(raw["success"].mean()),
        "seed_mean_success_rate": float(success_seed.mean()),
        "success_rate_95CI_halfwidth_t_df2": ind_t_halfwidth(success_seed),
        "mean_delay_given_success": float(raw.loc[raw["success"], "delay_to_run_start"].mean()) if raw["success"].any() else np.nan,
        "seed_mean_conditional_delay": float(cond_seed.mean()) if len(cond_seed) else np.nan,
        "conditional_delay_95CI_halfwidth": ind_t_halfwidth(cond_seed) if len(cond_seed) > 1 else np.nan,
        "seed_mean_restricted_delay": float(restricted_seed.mean()),
        "restricted_delay_95CI_halfwidth_t_df2": ind_t_halfwidth(restricted_seed),
    })

adapt_summary_ind = pd.DataFrame(summary_rows)
adapt_summary_ind.to_csv(IND_DIR / "adaptation_success_censoring_summary_ind.csv", index=False)
seed_adapt_ind.to_csv(IND_DIR / "adaptation_success_censoring_per_seed_ind.csv", index=False)

print("S6 source used:", s6_source)
print("\nPer-boundary event table:")
display(adapt_events_ind[adapt_events_ind["method"].isin(adaptive_methods)])
print("\nIND adaptation success/censoring summary:")
display(adapt_summary_ind)

