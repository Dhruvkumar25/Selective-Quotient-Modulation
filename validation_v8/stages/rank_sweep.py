# Ported from supervisor V8 notebook; see source_map.json.
# Independent paired rank sweep: every rank sees the same packet/channel/noise
# realization within a seed, making the rank comparison cleaner than separate draws.

fig9_rows = []
fig9_seed_rows = []
for snr_db in [5.0, 10.0, 15.0]:
    per_rank_seed = {r: [] for r in IND_RANKS}
    for seed in IND_SEEDS:
        sim_seed = 2026102400 + int(100 * snr_db) + seed
        Y, idx = ind_generate_tdla_trace(snr_db, IND_RANK_BLOCKS_PER_SEED, sim_seed)
        for r in IND_RANKS:
            scores = ind_score_cube(Y, ind_tdla_bank, rank=r)
            pred = ind_decode_from_scores(scores)
            per = float(np.mean(pred != idx))
            per_rank_seed[r].append(per)
            fig9_seed_rows.append({"snr_db": snr_db, "rank": r, "seed": seed, "PER": per, "blocks": len(idx)})
    for r in IND_RANKS:
        vals = np.asarray(per_rank_seed[r], float)
        fig9_rows.append({
            "snr_db": snr_db,
            "rank": r,
            "PER_mean": float(vals.mean()),
            "PER_95CI_halfwidth": ind_t_halfwidth(vals),
            "blocks": int(IND_RANK_BLOCKS_PER_SEED * len(vals)),
        })

fig9_ind = pd.DataFrame(fig9_rows)
fig9_seed_ind = pd.DataFrame(fig9_seed_rows)
fig9_ind.to_csv(IND_DIR / "fig9_rank_vs_snr_equation_ind.csv", index=False)
fig9_seed_ind.to_csv(IND_DIR / "fig9_rank_vs_snr_equation_per_seed_ind.csv", index=False)

released_fig9 = pd.read_csv(ROOT / "Data" / "rank_vs_snr.csv")
print("independent validation independent Fig. 9:")
display(fig9_ind)
print("Released Fig. 9 data:")
display(released_fig9)

best_ind = fig9_ind.loc[fig9_ind.groupby("snr_db")["PER_mean"].idxmin(), ["snr_db", "rank", "PER_mean"]].sort_values("snr_db")
print("Best screened independent validation rank at each SNR:")
display(best_ind)
print("Status: INDEPENDENT EQUATION-BASED RE-SIMULATION. Exact equality is not required because independent validation uses a new paired draw design.")

