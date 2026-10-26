# Ported from supervisor V8 notebook; see source_map.json.

# ---------- Held-out radius validation ----------
# Radius selection is isolated from the evaluation seeds.
radius_path = COMP_DIR / "mocz_family_radius_validation.csv"

if radius_path.exists() and not COMP_FORCE:
    radius_df = pd.read_csv(radius_path)
else:
    radius_grid = np.round(np.arange(1.10, 1.71, 0.10), 2)
    rows = []
    snr_val = 10.0
    seed_val = 20269991
    B = COMP_RADIUS_BLOCKS

    for R in radius_grid:
        rng = np.random.default_rng(seed_val)
        idx = rng.integers(256, size=B)
        h = (rng.normal(size=(B,2)) + 1j*rng.normal(size=(B,2)))/np.sqrt(2)
        h *= np.sqrt(PDP2)[None,:]
        N0 = 10.0**(-snr_val/10.0)

        # BMOCZ
        Xb, _ = bmocz_codebook(8, R)
        bb = build_gaussian_marginal_bank(Xb, np.array([1.0]), PDP2, N0)
        yb = simulate_from_marginal_bank(idx, h, bb, N0, rng)
        pb = decode_gaussian_marginal(yb, bb)
        rows.append({"scheme":"BMOCZ marginal ML", "R":R, "PER":np.mean(pb!=idx)})

        # SE-BMOCZ source model
        Xs, _ = bmocz_codebook(8, R)
        bs = build_gaussian_marginal_bank(
            Xs, SE_G, PDP2, N0, tau_noise=SE_TAU, beta=SE_BETA
        )
        ys = simulate_from_marginal_bank(idx, h, bs, N0, rng)
        ps = decode_gaussian_marginal(ys, bs)
        rows.append({"scheme":"SE-BMOCZ source-model marginal ML", "R":R, "PER":np.mean(ps!=idx)})

        # IM-MOCZ DiZeT, N=8, K=6
        Xi, _ = im_mocz_codebook(8, 6, R)
        yi = conv_channel_batch(Xi[idx], h)
        ni = np.sqrt(N0/2)*(rng.normal(size=yi.shape)+1j*rng.normal(size=yi.shape))
        pi = im_dizet_decode(yi+ni, 8, 6, R)
        rows.append({"scheme":"IM-MOCZ N=8,K=6 DiZeT", "R":R, "PER":np.mean(pi!=idx)})

        print(f"Radius R={R:.2f} completed.")

    radius_df = pd.DataFrame(rows)
    radius_df.to_csv(radius_path, index=False)

best_radius = (
    radius_df.sort_values(["scheme","PER","R"])
    .groupby("scheme", as_index=False)
    .first()[["scheme","R","PER"]]
    .rename(columns={"PER":"validation_PER"})
)
display(best_radius)

R_BM = float(best_radius.loc[best_radius.scheme.eq("BMOCZ marginal ML"), "R"].iloc[0])
R_SE = float(best_radius.loc[best_radius.scheme.eq("SE-BMOCZ source-model marginal ML"), "R"].iloc[0])
R_IM = float(best_radius.loc[best_radius.scheme.eq("IM-MOCZ N=8,K=6 DiZeT"), "R"].iloc[0])

