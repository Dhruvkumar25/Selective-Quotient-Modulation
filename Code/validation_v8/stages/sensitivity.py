# Ported from supervisor V8 notebook; see source_map.json.

# Recompute Figure 15 sensitivity using the same frozen validation procedure,
# but use:
#   * t(df=2) 95% intervals, not 1.96;
#   * adaptation delay from the three true boundaries only.

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from evaluate_online_adaptive import (
    make_trace,
    all_rank_stats,
    rank_maps,
    thresholds_from_validation,
    simulate_controller,
    RANKS,
)
from online_adaptive_diagnostics import pooled_validation
from run_final_headtohead import precompute_dmm_bases

SNRS_ADAPT = [5.0, 10.0, 15.0, 7.0]
DELAYS_ADAPT = [30.0, 60.0, 100.0, 30.0]
SEEDS_ADAPT = [13, 59, 151]

# Build the same nominal decoder banks.
t0 = make_trace(10, 777, SNRS_ADAPT, DELAYS_ADAPT)
Xd0, powers0, D0 = t0[5], t0[6], t0[7]
bases_sens = {
    r: precompute_dmm_bases(
        Xd0, powers0, D0, r, 4.0
    )[1]
    for r in RANKS
}

# Validation-only parameter selection, identical to released diagnostics.
vstats, vidx, vsnr, vdelay, vseg = pooled_validation(
    bases_sens
)
best_fixed, snr_map, regime_map, rank_bias, segdf = rank_maps(
    vstats, vidx, vseg, vsnr, vdelay
)
th = thresholds_from_validation(
    vstats, vidx, vseg, W=8
)

def corrected_controller_metrics(
    res,
    segs,
    regime_map
):
    target_map = {
        int(seg): int(regime_map[int(seg)])
        for seg in sorted(np.unique(segs))
    }

    mean_delay, _ = adaptation_delay_boundaries_only(
        res["selected"],
        segs,
        target_map,
        window=10,
    )

    return {
        "PER": float(np.mean(res["errors"])),
        "switch_rate": float(
            res["switches"] / max(1, len(res["selected"]) - 1)
        ),
        "mean_adaptation_delay_packets": mean_delay,
    }


sensitivity_rows = []

# Precompute each final seed ONCE.  The released diagnostics script
# recalculates the same trace/rank statistics for every W/alpha setting;
# caching them here makes the corrected V2 rerun much faster without
# changing the experiment.
final_sensitivity_cache = {}

for seed in tqdm(
    SEEDS_ADAPT,
    desc="Precomputing V2 sensitivity seeds",
    unit="seed"
):
    tr = make_trace(
        300,
        120000 + seed,
        SNRS_ADAPT,
        DELAYS_ADAPT
    )
    st = all_rank_stats(tr[0], bases_sens)

    final_sensitivity_cache[int(seed)] = {
        "trace": tr,
        "stats": st,
    }

settings = (
    [("S4_window", f"W={W}", W, np.nan)
     for W in [2, 4, 8, 16, 32]]
    +
    [("S5_EWMA", f"alpha={a:g}", np.nan, a)
     for a in [0.05, 0.10, 0.20, 0.35, 0.50]]
)

for policy, setting, W, alpha in tqdm(
    settings,
    desc="V2 sensitivity settings",
    unit="setting"
):
    vals = []

    for seed in SEEDS_ADAPT:
        cached = final_sensitivity_cache[int(seed)]
        tr = cached["trace"]
        st = cached["stats"]

        if policy == "S4_window":
            res = simulate_controller(
                st,
                tr[1],
                tr[4],
                "S4",
                W=int(W),
                tau_g=th["tau_g"],
                tau_e_hi=th["tau_e_hi"],
                tau_e_lo=th["tau_e_lo"],
                min_dwell=2,
            )
        else:
            res = simulate_controller(
                st,
                tr[1],
                tr[4],
                "S5",
                alpha=float(alpha),
                tau_g=th["tau_g"],
                tau_e_hi=th["tau_e_hi"],
                tau_e_lo=th["tau_e_lo"],
                min_dwell=2,
            )

        vals.append(
            corrected_controller_metrics(
                res,
                tr[4],
                regime_map
            )
        )

    per = np.array(
        [x["PER"] for x in vals],
        dtype=float
    )

    sensitivity_rows.append({
        "policy": policy,
        "setting": setting,
        "W": W,
        "alpha": alpha,
        "PER_mean": float(per.mean()),
        "PER_ci95_t_df2": float(
            T_CRIT_DF2
            * per.std(ddof=1)
            / np.sqrt(len(per))
        ),
        "switch_rate_mean": float(
            np.mean(
                [x["switch_rate"] for x in vals]
            )
        ),
        "adaptation_delay_mean_v2": float(
            np.mean([
                x["mean_adaptation_delay_packets"]
                for x in vals
            ])
        ),
    })

sensitivity_v2 = pd.DataFrame(sensitivity_rows)

sensitivity_v2.to_csv(
    V2_DIR / "online_controller_sensitivity_v2.csv",
    index=False
)

released_sensitivity = pd.read_csv(
    ROOT / "Data" / "online_controller_sensitivity.csv"
)

sens_compare = released_sensitivity.merge(
    sensitivity_v2,
    on=["policy", "setting"],
    how="inner",
    suffixes=("_released", "_v2")
)

print("=== Corrected sensitivity results ===")
display(sensitivity_v2)

print("\n=== Released vs V2 sensitivity ===")
display(
    sens_compare[
        [
            "policy",
            "setting",
            "PER_mean_released",
            "PER_mean_v2",
            "PER_ci95",
            "PER_ci95_t_df2",
            "adaptation_delay_mean",
            "adaptation_delay_mean_v2",
        ]
    ]
)


