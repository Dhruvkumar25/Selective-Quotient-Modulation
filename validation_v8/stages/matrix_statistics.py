# Ported from supervisor V8 notebook; see source_map.json.
# ---------- Pairwise SQM-vs-competitor deltas and 10 dB ranking ----------

pair_rows = []
for (panel, snr), g in v8_ps.groupby(["panel","snr_db"]):
    p = g.pivot(index="seed", columns="method", values="PER")
    if "SQM" not in p.columns:
        continue
    for method in V8_METHODS[1:]:
        d = p[method] - p["SQM"]
        pair_rows.append({
            "panel": panel,
            "snr_db": snr,
            "competitor": method,
            "mean_delta_PER_competitor_minus_SQM": float(d.mean()),
            "paired_95CI_halfwidth": float(t_halfwidth(d.to_numpy())),
            "seeds": int(len(d)),
            "SQM_lower_PER_if_positive": bool(d.mean() > 0),
        })

v8_deltas = pd.DataFrame(pair_rows)
v8_deltas.to_csv(V8_DIR / "sqm_vs_four_paired_deltas.csv", index=False)
display(v8_deltas)

# Ranking snapshot near 10 dB.
target_snr = min(V8_SNRS, key=lambda x: abs(float(x)-10.0))
v8_rank10 = (
    v8_summary_df[np.isclose(v8_summary_df.snr_db, float(target_snr))]
    .sort_values(["panel","PER_mean"])
    .copy()
)
v8_rank10["rank"] = v8_rank10.groupby("panel")["PER_mean"].rank(method="min")
v8_rank10.to_csv(V8_DIR / "ranking_at_10dB.csv", index=False)
display(v8_rank10[["panel","rank","method","PER_mean","PER_95CI_halfwidth",
                   "receiver","calibration_assumption"]])


