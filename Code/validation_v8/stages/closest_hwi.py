# Ported from supervisor V8 notebook; see source_map.json.
# ---------- Decisive matched TDL-A + hardware-uncertainty comparison ----------

DECISIVE_DIR = COMP_DIR / "decisive_hwi_vs_sqm"
DECISIVE_DIR.mkdir(parents=True, exist_ok=True)

if COMP_MODE == "manuscript":
    DECISIVE_BLOCKS = 800
    DECISIVE_TUNE_BLOCKS = 220
    DECISIVE_TWO_TAP_BLOCKS = 700
else:
    DECISIVE_BLOCKS = 80
    DECISIVE_TUNE_BLOCKS = 40
    DECISIVE_TWO_TAP_BLOCKS = 70

DECISIVE_RANKS = [2, 3, 4, 5, 6, 8]
DECISIVE_TUNE_SEEDS = [271, 863]  # held out from COMP_SEEDS
DECISIVE_TUNE_SNR_DB = 10

def save_decisive_figure(stem):
    png = DECISIVE_DIR / f"{stem}.png"
    pdf = DECISIVE_DIR / f"{stem}.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print("Saved:", png.relative_to(ROOT))
    print("Saved:", pdf.relative_to(ROOT))

# Fairness audit.
fairness_audit = pd.DataFrame([
    ["payload bits", 8, 8],
    ["messages", 256, 256],
    ["complex samples", 16, 16],
    ["mean codeword energy", float(np.mean(np.sum(np.abs(X_inc)**2, axis=1))),
     float(np.mean(np.sum(np.abs(HWI16)**2, axis=1)))],
    ["multipath", "fractional TDL-A", "fractional TDL-A"],
    ["CFO support", "[-4,4] deg/sample", "[-4,4] deg/sample"],
    ["I/Q model", "static real 2x2, cond=1.5", "static real 2x2, cond=1.5"],
    ["DC model", "same random draw", "same random draw"],
    ["AWGN", "same random draw", "same random draw"],
], columns=["quantity", "SQM", "HWI-Grassmannian"])
display(fairness_audit)
fairness_audit.to_csv(DECISIVE_DIR / "fairness_audit.csv", index=False)

def build_centered_cfo_line_bank(C, grid_deg=GRID5_DEG):
    """Centered unit-norm line templates for a CFO-search Grassmannian receiver."""
    C = np.asarray(C, complex)
    grid_deg = np.asarray(grid_deg, float)
    t = np.arange(C.shape[1])
    B = np.empty((len(C), len(grid_deg), C.shape[1]), complex)
    for m, x in enumerate(C):
        for j, deg in enumerate(grid_deg):
            v = x * np.exp(1j*np.deg2rad(deg)*t)
            v = v - v.mean()
            nv = np.linalg.norm(v)
            B[m, j] = v / max(nv, 1e-15)
    return B

def decode_centered_cfo_line(Y_real, bank, batch=64):
    """Closest native HWI extension: center, search CFO, then noncoherent line ML."""
    yc = Y_real[:, 0] + 1j*Y_real[:, 1]
    yc = yc - yc.mean(axis=1, keepdims=True)
    pred = np.empty(len(yc), int)
    for s in range(0, len(yc), batch):
        e = min(len(yc), s + batch)
        z = np.einsum("bl,mwl->bmw", yc[s:e], bank.conj(), optimize=True)
        score = np.abs(z)**2
        pred[s:e] = np.argmax(score.max(axis=2), axis=1)
    return pred

HWI_NATIVE_CFO_BANK = build_centered_cfo_line_bank(HWI16, GRID5_DEG)

# Generous HWI comparator: HWI codebook, but with the same TDL-A selective
# explanation machinery as SQM. This is explicitly a stress-test adaptation,
# not the source-paper HWI receiver.
hwi_tdla_bank_path = DECISIVE_DIR / "hwi_codebook_tdla_selective_bank_r8.npz"
if hwi_tdla_bank_path.exists() and not COMP_FORCE:
    HWI_TDLA_BANK = np.load(hwi_tdla_bank_path)["bank"]
    print("Loaded cached HWI TDL-A selective bank.")
else:
    print("Building HWI-codebook TDL-A selective basis bank ...")
    HWI_TDLA_BANK = build_tdla_bank_for_codebook(HWI16, max(DECISIVE_RANKS))
    np.savez_compressed(hwi_tdla_bank_path, bank=HWI_TDLA_BANK)

# Held-out rank selection for the generous HWI multipath-aware receiver.
rank_tune_path = DECISIVE_DIR / "hwi_tdla_rank_tuning_heldout.csv"
if rank_tune_path.exists() and not COMP_FORCE:
    hwi_rank_tune = pd.read_csv(rank_tune_path)
else:
    tune_rows = []
    for seed in DECISIVE_TUNE_SEEDS:
        nuisance = draw_common_tdla_nuisance(
            DECISIVE_TUNE_SNR_DB,
            DECISIVE_TUNE_BLOCKS,
            20267100 + seed,
        )
        idx = nuisance[0]
        Yhwi = apply_tdla_nuisance(HWI16, nuisance)
        for rank in DECISIVE_RANKS:
            pred = decode_tdla_bank(Yhwi, HWI_TDLA_BANK, rank)
            tune_rows.append({
                "snr_db": DECISIVE_TUNE_SNR_DB,
                "seed": seed,
                "rank": rank,
                "PER": float(np.mean(pred != idx)),
                "blocks": DECISIVE_TUNE_BLOCKS,
            })
    hwi_rank_tune = pd.DataFrame(tune_rows)
    hwi_rank_tune.to_csv(rank_tune_path, index=False)

hwi_rank_tune_summary = (
    hwi_rank_tune.groupby("rank", as_index=False)
    .agg(PER_mean=("PER", "mean"), seeds=("seed", "nunique"), blocks=("blocks", "sum"))
    .sort_values(["PER_mean", "rank"])
)
HWI_TDLA_SELECTED_RANK = int(hwi_rank_tune_summary.iloc[0]["rank"])
hwi_rank_tune_summary.to_csv(
    DECISIVE_DIR / "hwi_tdla_rank_tuning_summary.csv", index=False
)
print("Held-out HWI selective rank:", HWI_TDLA_SELECTED_RANK)
display(hwi_rank_tune_summary)

# Main decisive full TDL-A + CFO + static I/Q + DC experiment.
decisive_ps_path = DECISIVE_DIR / "tdla_hwi_dc_matched_per_seed.csv"
if decisive_ps_path.exists() and not COMP_FORCE:
    decisive_ps = pd.read_csv(decisive_ps_path)
else:
    rows = []
    for snr in COMP_SNRS:
        for seed in COMP_SEEDS:
            nuisance = draw_common_tdla_nuisance(
                snr, DECISIVE_BLOCKS, 20267200 + 1000*int(snr) + seed
            )
            idx = nuisance[0]

            # Same nuisance and AWGN realization for both codebooks.
            Ysqm = apply_tdla_nuisance(X_inc, nuisance)
            Yhwi = apply_tdla_nuisance(HWI16, nuisance)

            p_sqm = decode_tdla_bank(Ysqm, bank_inc, R_SEL)
            p_hwi_native = decode_centered_cfo_line(Yhwi, HWI_NATIVE_CFO_BANK)
            p_hwi_sq = decode_tdla_bank(
                Yhwi, HWI_TDLA_BANK, HWI_TDLA_SELECTED_RANK
            )

            for method, pred, receiver in [
                ("SQM released codebook + selective quotient r=4", p_sqm,
                 "released SQM selective quotient"),
                ("HWI-Grassmannian + centered CFO-search line ML", p_hwi_native,
                 "centered line ML with CFO search; no multipath model"),
                (f"HWI-Grassmannian + matched selective quotient r={HWI_TDLA_SELECTED_RANK}", p_hwi_sq,
                 "HWI codebook with held-out-tuned TDL-A selective quotient"),
            ]:
                rows.append({
                    "snr_db": snr,
                    "seed": seed,
                    "method": method,
                    "PER": float(np.mean(pred != idx)),
                    "blocks": DECISIVE_BLOCKS,
                    "payload_bits": 8,
                    "blocklength": 16,
                    "multipath": "released fractional TDL-A",
                    "cfo_support_deg_per_sample": "[-4,4]",
                    "iq_model": "static invertible real 2x2, condition number 1.5",
                    "dc_model": "unit-magnitude random direction",
                    "receiver": receiver,
                })
                print(
                    f"Decisive TDL-A SNR={snr:>4} seed={seed:>3} "
                    f"{method:58s} PER={rows[-1]['PER']:.5f}"
                )
    decisive_ps = pd.DataFrame(rows)
    decisive_ps.to_csv(decisive_ps_path, index=False)

decisive_summary = summarize_per_seed(decisive_ps)
decisive_summary.to_csv(
    DECISIVE_DIR / "tdla_hwi_dc_matched_summary.csv", index=False
)
display(decisive_summary)

method_order = [
    "SQM released codebook + selective quotient r=4",
    "HWI-Grassmannian + centered CFO-search line ML",
    f"HWI-Grassmannian + matched selective quotient r={HWI_TDLA_SELECTED_RANK}",
]

delta_rows = []
for snr, g in decisive_ps.groupby("snr_db"):
    p = g.pivot(index="seed", columns="method", values="PER")
    sqm_name = method_order[0]
    for competitor in method_order[1:]:
        d = p[competitor] - p[sqm_name]
        delta_rows.append({
            "snr_db": snr,
            "competitor": competitor,
            "mean_delta_PER_competitor_minus_SQM": float(d.mean()),
            "paired_95CI_halfwidth": float(t_halfwidth(d.to_numpy())),
            "seeds": len(d),
        })

decisive_deltas = pd.DataFrame(delta_rows)
decisive_deltas.to_csv(
    DECISIVE_DIR / "tdla_hwi_dc_paired_deltas.csv", index=False
)
display(decisive_deltas)

