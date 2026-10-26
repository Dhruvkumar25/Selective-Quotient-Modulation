# Ported from supervisor V8 notebook; see source_map.json.
# ---------- Run / load the complete 1+4 matrix ----------

v8_ps_path = V8_DIR / "complete_1plus4_per_seed.csv"

if v8_ps_path.exists() and not V8_FORCE:
    v8_ps = pd.read_csv(v8_ps_path)
    print("Loaded cached V8 complete 1+4 results.")
else:
    rows = []

    for snr in V8_SNRS:
        N0 = 10.0**(-float(snr)/10.0)

        # MOCZ receiver banks for the current Eb/N0.
        se_flat_bank = build_gaussian_marginal_bank(
            V8_X_SE, SE_G, np.array([1.0]), N0,
            tau_noise=SE_TAU, beta=SE_BETA,
        )
        se_two_bank = build_gaussian_marginal_bank(
            V8_X_SE, SE_G, PDP2, N0,
            tau_noise=SE_TAU, beta=SE_BETA,
        )

        for seed in V8_SEEDS:
            B = V8_BLOCKS

            # ================================================================
            # A — clean flat Rayleigh fading
            # ================================================================
            rng, idx, h, N0 = v8_draw_flat_common(
                snr, B, 20268000 + int(snr)*1000 + seed
            )

            def cnoise(shape):
                return np.sqrt(N0/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))

            y = h[:,None]*V8_X_SQM[idx] + cnoise((B, V8_X_SQM.shape[1]))
            p = decode_sqm_flat(y, SQM_FLAT0_BANK)
            v8_append(rows, "A: flat clean", snr, seed, "SQM", p, idx,
                      "quotient decoder", "none", V8_X_SQM.shape[1])

            y = h[:,None]*V8_X_HWI[idx] + cnoise((B, V8_X_HWI.shape[1]))
            p = cube_ml_decode(y, V8_X_HWI)
            v8_append(rows, "A: flat clean", snr, seed, "HWI-Grassmannian", p, idx,
                      "noncoherent line ML", "none", V8_X_HWI.shape[1])

            y = h[:,None]*V8_X_CUBE[idx] + cnoise((B, V8_X_CUBE.shape[1]))
            p = cube_ml_decode(y, V8_X_CUBE)
            v8_append(rows, "A: flat clean", snr, seed, "Cube-Split", p, idx,
                      "noncoherent line ML", "none", V8_X_CUBE.shape[1])

            y = simulate_from_marginal_bank(idx, h[:,None], se_flat_bank, N0, rng)
            p = decode_gaussian_marginal(y, se_flat_bank)
            v8_append(rows, "A: flat clean", snr, seed, "SE-MOCZ", p, idx,
                      "source-model Gaussian marginal ML", "none", y.shape[1])

            y = h[:,None]*V8_X_IM[idx] + cnoise((B, V8_X_IM.shape[1]))
            p = im_dizet_decode(y, 8, 6, R_IM)
            v8_append(rows, "A: flat clean", snr, seed, "IM-MOCZ", p, idx,
                      "DiZeT + majority vote", "none", y.shape[1])

            # ================================================================
            # B — flat fading + CFO + I/Q + DC
            # ================================================================
            rng = np.random.default_rng(20268100 + int(snr)*1000 + seed)
            idx = rng.integers(256, size=B)
            h = (rng.normal(size=B)+1j*rng.normal(size=B))/np.sqrt(2)
            omega = rng.uniform(-np.deg2rad(V8_CFO_MAX_DEG), np.deg2rad(V8_CFO_MAX_DEG), B)
            A = comp_random_A(rng, B, 1.5)
            dcang = rng.uniform(0, 2*np.pi, B)
            d = np.stack([np.cos(dcang), np.sin(dcang)], axis=1)

            s = h[:,None]*V8_X_SQM[idx]
            Y = v8_real_affine_observation(V8_X_SQM, idx, s, omega, A, d, N0, rng)
            p = decode_sqm_flat(Y[:,0] + 1j*Y[:,1], V8_SQM_FLAT_HWI_BANK)
            v8_append(rows, "B: flat + HWI", snr, seed, "SQM", p, idx,
                      "centered quotient + CFO search", "none", V8_X_SQM.shape[1])

            s = h[:,None]*V8_X_HWI[idx]
            Y = v8_real_affine_observation(V8_X_HWI, idx, s, omega, A, d, N0, rng)
            p = decode_centered_cfo_line(Y, V8_HWI_LINE_BANK)
            v8_append(rows, "B: flat + HWI", snr, seed, "HWI-Grassmannian", p, idx,
                      "centered CFO-search line ML", "none", V8_X_HWI.shape[1])

            s = h[:,None]*V8_X_CUBE[idx]
            Y = v8_real_affine_observation(V8_X_CUBE, idx, s, omega, A, d, N0, rng)
            p = decode_centered_cfo_line(Y, V8_CUBE_LINE_BANK)
            v8_append(rows, "B: flat + HWI", snr, seed, "Cube-Split", p, idx,
                      "centered CFO-search line ML", "none", V8_X_CUBE.shape[1])

            # MOCZ family gets an intentionally generous oracle front-end cleanup:
            # same payload/fading/EbN0, but hardware nuisance is perfectly removed.
            y = simulate_from_marginal_bank(idx, h[:,None], se_flat_bank, N0, rng)
            p = decode_gaussian_marginal(y, se_flat_bank)
            v8_append(rows, "B: flat + HWI", snr, seed, "SE-MOCZ", p, idx,
                      "Gaussian marginal ML after oracle HWI cleanup",
                      "oracle front-end cleanup", y.shape[1])

            y = h[:,None]*V8_X_IM[idx] + cnoise((B, V8_X_IM.shape[1]))
            p = im_dizet_decode(y, 8, 6, R_IM)
            v8_append(rows, "B: flat + HWI", snr, seed, "IM-MOCZ", p, idx,
                      "DiZeT after oracle HWI cleanup",
                      "oracle front-end cleanup", y.shape[1])

            # ================================================================
            # C — clean two-tap multipath
            # ================================================================
            rng = np.random.default_rng(20268200 + int(snr)*1000 + seed)
            idx = rng.integers(256, size=B)
            h2 = (rng.normal(size=(B,2))+1j*rng.normal(size=(B,2)))/np.sqrt(2)
            h2 *= np.sqrt(PDP2)[None,:]

            def cnoise2(shape):
                return np.sqrt(N0/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))

            y = v8_two_tap_signal(V8_X_SQM, idx, h2) + cnoise2((B,16))
            p = decode_tdla_bank(v8_complex_to_real(y), V8_SQM_2T_BANK, 4)
            v8_append(rows, "C: two-tap clean", snr, seed, "SQM", p, idx,
                      "matched rank-4 two-tap quotient", "none", 16)

            y = v8_two_tap_signal(V8_X_HWI, idx, h2) + cnoise2((B,16))
            p = decode_tdla_bank(v8_complex_to_real(y), V8_HWI_2T_BANK, 4)
            v8_append(rows, "C: two-tap clean", snr, seed, "HWI-Grassmannian", p, idx,
                      "matched rank-4 two-tap quotient", "none", 16)

            y = v8_two_tap_signal(V8_X_CUBE, idx, h2) + cnoise2((B,16))
            p = decode_tdla_bank(v8_complex_to_real(y), V8_CUBE_2T_BANK, 4)
            v8_append(rows, "C: two-tap clean", snr, seed, "Cube-Split", p, idx,
                      "matched rank-4 two-tap quotient", "none", 16)

            y = simulate_from_marginal_bank(idx, h2, se_two_bank, N0, rng)
            p = decode_gaussian_marginal(y, se_two_bank)
            v8_append(rows, "C: two-tap clean", snr, seed, "SE-MOCZ", p, idx,
                      "two-tap Gaussian marginal ML", "none", y.shape[1])

            y = conv_channel_batch(V8_X_IM[idx], h2) + cnoise2((B, V8_X_IM.shape[1]+1))
            p = im_dizet_decode(y, 8, 6, R_IM)
            v8_append(rows, "C: two-tap clean", snr, seed, "IM-MOCZ", p, idx,
                      "two-tap DiZeT + majority vote", "none", y.shape[1])

            # ================================================================
            # D — two-tap multipath + CFO + I/Q + DC
            # ================================================================
            rng = np.random.default_rng(20268300 + int(snr)*1000 + seed)
            idx = rng.integers(256, size=B)
            h2 = (rng.normal(size=(B,2))+1j*rng.normal(size=(B,2)))/np.sqrt(2)
            h2 *= np.sqrt(PDP2)[None,:]
            omega = rng.uniform(-np.deg2rad(V8_CFO_MAX_DEG), np.deg2rad(V8_CFO_MAX_DEG), B)
            A = comp_random_A(rng, B, 1.5)
            dcang = rng.uniform(0, 2*np.pi, B)
            d = np.stack([np.cos(dcang), np.sin(dcang)], axis=1)

            s = v8_two_tap_signal(V8_X_SQM, idx, h2)
            Y = v8_real_affine_observation(V8_X_SQM, idx, s, omega, A, d, N0, rng)
            p = decode_tdla_bank(Y, V8_SQM_2T_BANK, 4)
            v8_append(rows, "D: two-tap + HWI", snr, seed, "SQM", p, idx,
                      "matched rank-4 two-tap quotient + CFO search", "none", 16)

            s = v8_two_tap_signal(V8_X_HWI, idx, h2)
            Y = v8_real_affine_observation(V8_X_HWI, idx, s, omega, A, d, N0, rng)
            p = decode_tdla_bank(Y, V8_HWI_2T_BANK, 4)
            v8_append(rows, "D: two-tap + HWI", snr, seed, "HWI-Grassmannian", p, idx,
                      "matched rank-4 two-tap quotient + CFO search", "none", 16)

            s = v8_two_tap_signal(V8_X_CUBE, idx, h2)
            Y = v8_real_affine_observation(V8_X_CUBE, idx, s, omega, A, d, N0, rng)
            p = decode_tdla_bank(Y, V8_CUBE_2T_BANK, 4)
            v8_append(rows, "D: two-tap + HWI", snr, seed, "Cube-Split", p, idx,
                      "matched rank-4 two-tap quotient + CFO search", "none", 16)

            # Again, give MOCZ an oracle HWI-cleaned upper reference.
            y = simulate_from_marginal_bank(idx, h2, se_two_bank, N0, rng)
            p = decode_gaussian_marginal(y, se_two_bank)
            v8_append(rows, "D: two-tap + HWI", snr, seed, "SE-MOCZ", p, idx,
                      "two-tap marginal ML after oracle HWI cleanup",
                      "oracle front-end cleanup", y.shape[1])

            y = conv_channel_batch(V8_X_IM[idx], h2) + cnoise2((B, V8_X_IM.shape[1]+1))
            p = im_dizet_decode(y, 8, 6, R_IM)
            v8_append(rows, "D: two-tap + HWI", snr, seed, "IM-MOCZ", p, idx,
                      "two-tap DiZeT after oracle HWI cleanup",
                      "oracle front-end cleanup", y.shape[1])

            print(f"V8 SNR={snr:>4} seed={seed:>3}: four panels complete")

    v8_ps = pd.DataFrame(rows)
    v8_ps.to_csv(v8_ps_path, index=False)

v8_summary_df = v8_summary(v8_ps)
meta_cols = ["panel","method","payload_bits","observation_samples",
             "bits_per_observation_sample","receiver","calibration_assumption"]
v8_meta = v8_ps.groupby(["panel","method"], as_index=False)[meta_cols[2:]].first()
v8_summary_df = v8_summary_df.merge(v8_meta, on=["panel","method"], how="left")
v8_summary_df.to_csv(V8_DIR / "complete_1plus4_summary.csv", index=False)

display(v8_summary_df)

