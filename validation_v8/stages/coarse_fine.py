# Ported from supervisor V8 notebook; see source_map.json.
coarse_deg = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
coarse_idx = np.array([int(np.where(IND_GRID_DEG == d)[0][0]) for d in coarse_deg])
K_LIST = [4, 8, 16]


def ind_ctf_predictions(scores, K):
    coarse_msg = scores[:, :, coarse_idx].max(axis=2)
    shortlist = np.argpartition(coarse_msg, -K, axis=1)[:, -K:]
    pred = np.empty(len(scores), int)
    for b in range(len(scores)):
        mids = shortlist[b]
        refined = scores[b, mids, :]
        flat = int(np.argmax(refined))
        pred[b] = int(mids[flat // refined.shape[1]])
    return pred

ctf_seed_rows = []
for seed in IND_SEEDS:
    Y, idx = ind_generate_tdla_trace(10.0, IND_CTF_BLOCKS_PER_SEED, 2026102600 + seed)
    scores = ind_score_cube(Y, ind_tdla_bank, rank=4, batch=24)
    pred_full = ind_decode_from_scores(scores)
    per_full = float(np.mean(pred_full != idx))
    ctf_seed_rows.append({
        "decoder": "full search", "seed": seed, "PER": per_full,
        "disagreement_with_full": 0.0, "candidate_CFO_scores": 256 * 11,
    })
    for K in K_LIST:
        pred = ind_ctf_predictions(scores, K)
        ctf_seed_rows.append({
            "decoder": f"coarse-to-fine top-{K}",
            "seed": seed,
            "PER": float(np.mean(pred != idx)),
            "disagreement_with_full": float(np.mean(pred != pred_full)),
            "candidate_CFO_scores": int(256 * 5 + K * 11),
        })

ctf_seed_ind = pd.DataFrame(ctf_seed_rows)
ctf_rows = []
for decoder, g in ctf_seed_ind.groupby("decoder", sort=False):
    p = g["PER"].to_numpy(float)
    d = g["disagreement_with_full"].to_numpy(float)
    count = int(g["candidate_CFO_scores"].iloc[0])
    K = None if decoder == "full search" else int(decoder.rsplit("-", 1)[1])
    reuse_count = count if K is None else int(256 * 5 + K * (11 - 5))
    ctf_rows.append({
        "decoder": decoder,
        "PER_mean": float(p.mean()),
        "PER_95CI_halfwidth": ind_t_halfwidth(p),
        "disagreement_with_full": float(d.mean()),
        "candidate_CFO_scores": count,
        "relative_score_count": count / (256 * 11),
        "scores_if_coarse_reused": reuse_count,
        "relative_if_coarse_reused": reuse_count / (256 * 11),
        "blocks": int(IND_CTF_BLOCKS_PER_SEED * len(g)),
    })

fig12_ind = pd.DataFrame(ctf_rows)
fig12_ind.to_csv(IND_DIR / "fig12_coarse_to_fine_equation_ind.csv", index=False)
ctf_seed_ind.to_csv(IND_DIR / "fig12_coarse_to_fine_equation_per_seed_ind.csv", index=False)
released_ctf = pd.read_csv(ROOT / "Data" / "fractional_decoder_complexity.csv")

print("independent validation independent coarse-to-fine results:")
display(fig12_ind)
print("Released Fig. 12 data:")
display(released_ctf)

print("Top-8 Eq. (36) count:", 256*5 + 8*11, "=", (256*5+8*11)/(256*11)*100, "% of full")
print("Top-8 with cached coarse scores:", 256*5 + 8*(11-5), "=", (256*5+8*(11-5))/(256*11)*100, "% of full")

