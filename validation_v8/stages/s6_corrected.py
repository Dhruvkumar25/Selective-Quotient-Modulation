# Ported from supervisor V8 notebook; see source_map.json.

# Correct S6 semantics:
# the adjacent-rank comparison now changes the CURRENT packet's decoded
# prediction, selected rank, residual/gap record, and rank-aware MAC count.

from evaluate_online_adaptive import (
    make_trace,
    all_rank_stats,
    RANKS,
    FULL_EVALS,
)
import evaluate_online_adaptive_revision as adaptive_revision
from tqdm.auto import tqdm

L_BLOCK = 16

def simulate_s6_same_packet(
    stats,
    idx,
    *,
    tau_g,
    rank_bias,
    initial_rank=4
):
    n = len(idx)
    rank_to_i = {r: i for i, r in enumerate(RANKS)}

    selected = np.empty(n, dtype=int)
    pred = np.empty(n, dtype=int)
    gaps = np.empty(n, dtype=float)
    residuals = np.empty(n, dtype=float)
    evals = np.full(n, FULL_EVALS, dtype=float)
    approx_macs = np.empty(n, dtype=float)

    active = int(initial_rank)
    switches = 0

    for t in range(n):
        base_rank = active

        # Trigger is evaluated from the current active rank.
        base_gap = float(stats[base_rank][1][t])

        candidates = [base_rank]

        if base_gap < tau_g:
            i = rank_to_i[base_rank]

            if i > 0:
                candidates.append(RANKS[i - 1])

            if i < len(RANKS) - 1:
                candidates.append(RANKS[i + 1])

            def util(r):
                fit = 1.0 - float(stats[r][2][t])
                return (
                    float(stats[r][1][t])
                    - (rank_bias or {}).get(r, 0.0)
                    + 0.25 * (
                        fit
                        - (rank_bias or {}).get(
                            ("fit", r), 0.0
                        )
                    )
                )

            chosen = max(candidates, key=util)
        else:
            chosen = base_rank

        # SAME-PACKET correction:
        # use the winning candidate to decode packet t.
        selected[t] = int(chosen)
        pred[t] = int(stats[chosen][0][t])
        gaps[t] = float(stats[chosen][1][t])
        residuals[t] = float(stats[chosen][2][t])

        if chosen != base_rank:
            switches += 1

        # Carry the chosen rank forward.
        active = int(chosen)

        # Every candidate was scored over the full candidate/CFO bank.
        evals[t] = FULL_EVALS * len(candidates)

        # Rank-aware approximate MAC count:
        # sum cost over the ranks actually evaluated.
        approx_macs[t] = (
            FULL_EVALS
            * sum(2 * L_BLOCK * int(r) for r in candidates)
        )

    return {
        "scheme": "S6_same_packet_v2",
        "errors": (pred != idx).astype(float),
        "pred": pred,
        "selected": selected,
        "gap": gaps,
        "residual": residuals,
        "evals": evals,
        "approx_macs": approx_macs,
        "switches": int(switches),
    }


print("=== Calibrating corrected S6 on validation traces ===")

bases_s6 = adaptive_revision.build_bases()

(
    th_s6,
    rank_bias_s6,
    vstats_s6,
    vidx_s6,
    vsegs_s6,
    vsnrs_s6,
    vdelays_s6,
) = adaptive_revision.thresholds_and_rank_bias(
    bases_s6
)

s6_grid = np.quantile(
    np.concatenate(
        [vstats_s6[r][1] for r in RANKS]
    ),
    [0.10, 0.20, 0.30, 0.40]
)

s6_validation_rows = []

for tau in s6_grid:
    rr = simulate_s6_same_packet(
        vstats_s6,
        vidx_s6,
        tau_g=float(tau),
        rank_bias=rank_bias_s6,
        initial_rank=4,
    )

    s6_validation_rows.append({
        "tau_gap": float(tau),
        "validation_PER": float(
            rr["errors"].mean()
        ),
    })

s6_validation = pd.DataFrame(s6_validation_rows)
best_idx = s6_validation["validation_PER"].idxmin()
s6_tau_v2 = float(
    s6_validation.loc[best_idx, "tau_gap"]
)

display(s6_validation)
print("Selected V2 S6 threshold:", s6_tau_v2)


# ------------------------------------------------------------
# Final canonical three-seed rerun
# ------------------------------------------------------------

# Oracle target ranks are derived from the independently reproduced S8 trace.
target_by_seed_segment = oracle_targets

s6_seed_rows = []
s6_packet_rows = []

for seed in tqdm(
    [13, 59, 151],
    desc="V2 S6 final seeds",
    unit="seed"
):
    tr = make_trace(
        300,
        120000 + seed,
        [5.0, 10.0, 15.0, 7.0],
        [30.0, 60.0, 100.0, 30.0],
    )

    Y, idx, snrs, delays, segs = tr[:5]
    stats = all_rank_stats(Y, bases_s6)

    res = simulate_s6_same_packet(
        stats,
        idx,
        tau_g=s6_tau_v2,
        rank_bias=rank_bias_s6,
        initial_rank=4,
    )

    target_map = {
        int(seg): int(
            target_by_seed_segment.loc[
                (seed, seg)
            ]
        )
        for seg in sorted(np.unique(segs))
    }

    delay_mean, delay_each = (
        adaptation_delay_boundaries_only(
            res["selected"],
            segs,
            target_map,
            window=10,
        )
    )

    transitions = np.r_[
        False,
        res["selected"][1:]
        != res["selected"][:-1]
    ]

    s6_seed_rows.append({
        "sequence": "canonical",
        "seed": int(seed),
        "method": "S6_confidence_triggered_refinement",
        "blocks": len(idx),
        "PER": float(res["errors"].mean()),
        "mean_selected_rank": float(
            res["selected"].mean()
        ),
        "controller_switches": int(
            res["switches"]
        ),
        "controller_switch_rate": float(
            res["switches"]
            / max(1, len(idx) - 1)
        ),
        "selected_rank_transitions": int(
            transitions.sum()
        ),
        "switch_rate": float(
            transitions.sum()
            / max(1, len(idx) - 1)
        ),
        "mean_adaptation_delay_packets":
            float(delay_mean),
        "mean_score_evals": float(
            res["evals"].mean()
        ),
        "p95_score_evals": float(
            np.quantile(res["evals"], 0.95)
        ),
        "mean_approx_macs": float(
            res["approx_macs"].mean()
        ),
        "p95_approx_macs": float(
            np.quantile(
                res["approx_macs"], 0.95
            )
        ),
    })

    for t in range(len(idx)):
        s6_packet_rows.append({
            "sequence": "canonical",
            "seed": int(seed),
            "method":
                "S6_confidence_triggered_refinement",
            "packet": int(t),
            "segment": int(segs[t]),
            "snr_db": float(snrs[t]),
            "delay_ns": float(delays[t]),
            "error": int(res["errors"][t]),
            "selected_rank":
                int(res["selected"][t]),
            "selected_rank_transition":
                int(transitions[t]),
            "score_evals":
                float(res["evals"][t]),
            "approx_macs":
                float(res["approx_macs"][t]),
        })

s6_seed_v2 = pd.DataFrame(s6_seed_rows)
s6_packet_v2 = pd.DataFrame(s6_packet_rows)

s6_seed_v2.to_csv(
    V2_DIR / "s6_same_packet_seed_metrics_v2.csv",
    index=False
)
s6_packet_v2.to_csv(
    V2_DIR / "s6_same_packet_packet_results_v2.csv",
    index=False
)

print("\n=== Corrected S6 seed metrics ===")
display(s6_seed_v2)


# ------------------------------------------------------------
# Build corrected canonical Table VII dataset:
# - use reproduced released metrics for unchanged schemes;
# - replace adaptation delay for all adaptive schemes;
# - replace all S6 metrics with same-packet V2 results.
# ------------------------------------------------------------

table7_seed_v2 = (
    released_seed_metrics[
        released_seed_metrics["sequence"].eq(
            "canonical"
        )
    ]
    .copy()
)

# Correct adaptation delay for unchanged adaptive methods.
delay_lookup = delay_v2.set_index(
    ["seed", "method"]
)["adaptation_delay_v2"]

for i, row in table7_seed_v2.iterrows():
    key = (int(row["seed"]), row["method"])

    if key in delay_lookup.index:
        table7_seed_v2.loc[
            i,
            "mean_adaptation_delay_packets"
        ] = float(delay_lookup.loc[key])

# Replace S6 rows fully.
s6_cols = [
    c for c in s6_seed_v2.columns
    if c in table7_seed_v2.columns
]

for _, r in s6_seed_v2.iterrows():
    mask = (
        table7_seed_v2["seed"].eq(r["seed"])
        &
        table7_seed_v2["method"].eq(r["method"])
    )

    for c in s6_cols:
        table7_seed_v2.loc[mask, c] = r[c]


# Three-seed t-based summary.
summary_rows = []

summary_metrics = [
    "PER",
    "mean_selected_rank",
    "controller_switch_rate",
    "switch_rate",
    "mean_adaptation_delay_packets",
    "mean_score_evals",
    "p95_score_evals",
    "mean_approx_macs",
    "p95_approx_macs",
]

for method, g in table7_seed_v2.groupby(
    "method",
    sort=False
):
    row = {
        "method": method,
        "seeds": len(g),
        "blocks_total": int(g["blocks"].sum()),
    }

    for col in summary_metrics:
        vals = (
            g[col]
            .dropna()
            .to_numpy(dtype=float)
        )

        row[col] = (
            float(vals.mean())
            if len(vals)
            else np.nan
        )

        row[col + "_95CI_halfwidth"] = (
            float(
                T_CRIT_DF2
                * vals.std(ddof=1)
                / np.sqrt(len(vals))
            )
            if len(vals) > 1
            else 0.0
        )

    summary_rows.append(row)

table7_summary_v2 = pd.DataFrame(
    summary_rows
)

table7_seed_v2.to_csv(
    V2_DIR / "table_vii_seed_metrics_v2.csv",
    index=False
)
table7_summary_v2.to_csv(
    V2_DIR / "table_vii_summary_v2.csv",
    index=False
)

print("\n=== Corrected Table VII summary ===")
display(table7_summary_v2)

old_s6 = released_seed_metrics[
    released_seed_metrics["sequence"].eq("canonical")
    &
    released_seed_metrics["method"].eq(
        "S6_confidence_triggered_refinement"
    )
]

print("\n=== Released S6 vs same-packet V2 ===")
display(
    pd.concat([
        old_s6.assign(version="released"),
        s6_seed_v2.assign(version="V2 same-packet"),
    ], ignore_index=True)
)

