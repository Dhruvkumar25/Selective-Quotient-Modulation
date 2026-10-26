# Ported from supervisor V8 notebook; see source_map.json.
# Closest released operating-screen codebook to the legacy Fig. 11 experiment.
stage3 = np.load(ROOT / "Artifact" / "stage3_codebooks.npz")
X2 = stage3["X_dmm"]
XD2 = X2[:, 0] + 1j * X2[:, 1]
L2 = XD2.shape[1]
PDP2 = np.array([0.8, 0.2], dtype=float)


def ind_shift1(x):
    y = np.zeros_like(x)
    y[..., 1:] = x[..., :-1]
    return y


def ind_two_tap_basis_bank(cfo_range_deg, rank=4):
    if float(cfo_range_deg) == 0.0:
        grid_deg = np.array([0.0])
    else:
        grid_deg = np.linspace(
            -cfo_range_deg - 1.0,
            cfo_range_deg + 1.0,
            int(round(2 * cfo_range_deg + 2)) + 1,
        )
    grid = np.deg2rad(grid_deg)
    bank = np.empty((256, len(grid), L2, rank), np.float32)
    H2 = np.eye(L2) - np.ones((L2, L2)) / L2
    t = np.arange(L2)
    for m in range(256):
        for j, w in enumerate(grid):
            # Physical two-tap feature family. A constant phase factor on a delayed
            # tap is absorbed by the arbitrary real 2x2 coefficient block for that tap.
            u = XD2[m] * np.exp(1j * w * t)
            v = ind_shift1(u)
            F = np.stack([u.real, u.imag, v.real, v.imag]) @ H2
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            bank[m, j] = Vh[:rank].T.astype(np.float32)
    return grid_deg, bank


def ind_decode_variable_grid(Y, bank, rank=4, batch=48):
    Yc = Y - Y.mean(axis=2, keepdims=True)
    pred = np.empty(len(Y), int)
    B = bank[..., :rank]
    for s in range(0, len(Y), batch):
        e = min(len(Y), s + batch)
        Z = np.einsum("bal,mwlr->bmwar", Yc[s:e], B, optimize=True)
        sc = np.sum(Z * Z, axis=(3, 4))
        pred[s:e] = np.argmax(sc.max(axis=2), axis=1)
    return pred


def ind_two_tap_run(cfo_range_deg, blocks, seed, snr_db=10.0, kappa=2.0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(256, size=blocks)
    h = (
        (rng.normal(size=(blocks, 2)) + 1j * rng.normal(size=(blocks, 2)))
        / np.sqrt(2.0)
    ) * np.sqrt(PDP2)[None, :]

    x = XD2[idx]
    y = np.zeros_like(x, dtype=complex)
    y[:, 0] = h[:, 0] * x[:, 0]
    y[:, 1:] = h[:, 0, None] * x[:, 1:] + h[:, 1, None] * x[:, :-1]

    if cfo_range_deg == 0:
        omega = np.zeros(blocks)
    else:
        omega = rng.uniform(
            -np.deg2rad(cfo_range_deg),
            np.deg2rad(cfo_range_deg),
            blocks,
        )
    y *= np.exp(1j * omega[:, None] * np.arange(L2)[None, :])

    A = ind_random_A(rng, blocks, kappa=kappa)
    ang = rng.uniform(0.0, 2 * np.pi, blocks)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    N0 = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(N0 / 2.0) * rng.normal(size=(blocks, 2, L2))
    yr = np.stack([y.real, y.imag], axis=1)
    Y = np.einsum("nij,njk->nik", A, yr, optimize=True) + d[:, :, None] + noise

    _, bank = ind_two_tap_basis_bank(cfo_range_deg, rank=4)
    pred = ind_decode_variable_grid(Y, bank, rank=4)
    return float(np.mean(pred != idx))


cfo_rows = []
for cfo in [0.0, 2.0, 4.0, 8.0, 12.0]:
    vals = []
    for seed in IND_SEEDS:
        per = ind_two_tap_run(
            cfo,
            IND_CFO_BLOCKS_PER_SEED,
            2026102500 + int(100 * cfo) + seed,
        )
        vals.append(per)
    vals = np.asarray(vals)
    cfo_rows.append({
        "cfo_range_deg_per_sample": cfo,
        "method": "independent validation equation-based SQM reconstruction",
        "PER_mean": float(vals.mean()),
        "PER_95CI_halfwidth": ind_t_halfwidth(vals),
        "blocks": int(IND_CFO_BLOCKS_PER_SEED * len(vals)),
    })

fig11_ind = pd.DataFrame(cfo_rows)
fig11_ind.to_csv(IND_DIR / "fig11_cfo_sqm_equation_ind.csv", index=False)
released_cfo = pd.read_csv(ROOT / "Data" / "operating_cfo_sweep.csv")
released_sqm = (
    released_cfo[released_cfo["method"].eq("Incidence-DMM")]
    [["x", "PER_mean", "PER_95CI_halfwidth"]]
    .rename(columns={
        "x": "cfo_range_deg_per_sample",
        "PER_mean": "released_PER_mean",
        "PER_95CI_halfwidth": "released_95CI_halfwidth",
    })
)
fig11_compare_ind = fig11_ind.merge(released_sqm, on="cfo_range_deg_per_sample", how="left")
fig11_compare_ind["ind_minus_released"] = fig11_compare_ind["PER_mean"] - fig11_compare_ind["released_PER_mean"]
fig11_compare_ind["abs_difference"] = fig11_compare_ind["ind_minus_released"].abs()
fig11_compare_ind.to_csv(IND_DIR / "fig11_cfo_sqm_released_comparison_ind.csv", index=False)

print("Independent independent validation SQM Fig. 11 reconstruction:")
display(fig11_ind)
print("Direct comparison with the released SQM curve:")
display(fig11_compare_ind)
print("Released full Fig. 11 data:")
display(released_cfo)

mean_abs_gap = float(fig11_compare_ind["abs_difference"].mean())
print(f"Mean absolute independent validation-vs-released SQM PER gap: {mean_abs_gap:.6f}")
if mean_abs_gap > 0.01:
    print("PROVENANCE WARNING: the independent reconstruction does not reproduce the released Fig. 11 SQM levels closely.")
    print("This indicates that at least one legacy codebook/model/noise/RNG convention used for operating_cfo_sweep.csv is not fully specified by the current manuscript/repository.")
else:
    print("The independent reconstruction is numerically close to the released SQM curve.")
print("Status: equation-based reconstruction attempted and quantified; released OFDM curves remain reference data, not independently regenerated.")

