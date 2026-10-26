# Ported from supervisor V8 notebook; see source_map.json.
bank_inc_path = COMP_DIR / "ablation_bank_incidence_r15.npz"
bank_pre_path = COMP_DIR / "ablation_bank_preincidence_r15.npz"

if bank_inc_path.exists() and bank_pre_path.exists() and not COMP_FORCE:
    bank_inc = np.load(bank_inc_path)["bank"]
    bank_pre = np.load(bank_pre_path)["bank"]
    print("Loaded cached comparative study ablation basis banks.")
else:
    print("Building incidence-trained rank-15 bank ...")
    bank_inc = build_tdla_bank_for_codebook(X_inc, R_FULL)
    np.savez_compressed(bank_inc_path, bank=bank_inc)
    print("Building pre-incidence rank-15 bank ...")
    bank_pre = build_tdla_bank_for_codebook(X_pre, R_FULL)
    np.savez_compressed(bank_pre_path, bank=bank_pre)

ablation_ps_path = COMP_DIR / "sqm_factorial_ablation_per_seed.csv"

if ablation_ps_path.exists() and not COMP_FORCE:
    ablation_ps = pd.read_csv(ablation_ps_path)
    print("Loaded cached factorial ablation.")
else:
    rows = []
    configs = [
        ("SQM: incidence + rank 4", X_inc, bank_inc, R_SEL),
        ("Ablation: incidence + full rank", X_inc, bank_inc, R_FULL),
        ("Ablation: pre-incidence + rank 4", X_pre, bank_pre, R_SEL),
        ("Ablation: pre-incidence + full rank", X_pre, bank_pre, R_FULL),
    ]
    for snr in COMP_SNRS:
        for seed in COMP_SEEDS:
            nuisance = draw_common_tdla_nuisance(
                snr, COMP_ABLATION_BLOCKS, 20261025 + 1000*int(snr) + seed
            )
            idx = nuisance[0]
            for name, Xd, bank, rank in configs:
                Y = apply_tdla_nuisance(Xd, nuisance)
                pred = decode_tdla_bank(Y, bank, rank)
                rows.append({
                    "snr_db": snr,
                    "seed": seed,
                    "method": name,
                    "rank": rank,
                    "PER": float(np.mean(pred != idx)),
                    "blocks": COMP_ABLATION_BLOCKS,
                })
                print(f"Ablation SNR={snr:>4} seed={seed:>3} {name:38s} PER={rows[-1]['PER']:.5f}")
    ablation_ps = pd.DataFrame(rows)
    ablation_ps.to_csv(ablation_ps_path, index=False)

ablation_summary = summarize_per_seed(ablation_ps)
ablation_summary.to_csv(COMP_DIR / "sqm_factorial_ablation_summary.csv", index=False)
display(ablation_summary)

# Paired factorial effects: positive values mean the named SQM component reduces PER.
effect_rows = []
for snr, g in ablation_ps.groupby("snr_db"):
    p = g.pivot(index="seed", columns="method", values="PER")
    effects = {
        "incidence benefit at rank 4":
            p["Ablation: pre-incidence + rank 4"] - p["SQM: incidence + rank 4"],
        "incidence benefit at full rank":
            p["Ablation: pre-incidence + full rank"] - p["Ablation: incidence + full rank"],
        "selective-rank benefit with incidence":
            p["Ablation: incidence + full rank"] - p["SQM: incidence + rank 4"],
        "selective-rank benefit pre-incidence":
            p["Ablation: pre-incidence + full rank"] - p["Ablation: pre-incidence + rank 4"],
    }
    effects["interaction (incidence changes rank benefit)"] = (
        effects["selective-rank benefit with incidence"] -
        effects["selective-rank benefit pre-incidence"]
    )
    for name, x in effects.items():
        effect_rows.append({
            "snr_db": snr,
            "effect": name,
            "mean_delta_PER": float(x.mean()),
            "paired_95CI_halfwidth": float(t_halfwidth(x.to_numpy())),
            "seeds": len(x),
        })

ablation_effects = pd.DataFrame(effect_rows)
ablation_effects.to_csv(COMP_DIR / "sqm_factorial_ablation_effects.csv", index=False)
display(ablation_effects)

