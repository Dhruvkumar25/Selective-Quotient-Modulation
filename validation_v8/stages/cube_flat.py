# Ported from supervisor V8 notebook; see source_map.json.
cube_ps_path = COMP_DIR / "cube_split_flat_comparison_per_seed.csv"
if cube_ps_path.exists() and not COMP_FORCE:
    cube_ps = pd.read_csv(cube_ps_path)
else:
    rows = []
    Xsqm = X_inc.copy()  # already energy 16
    for snr in COMP_SNRS:
        for seed in COMP_SEEDS:
            rng = np.random.default_rng(20262000 + int(snr)*1000 + seed)
            idx = rng.integers(256, size=COMP_FLAT_BLOCKS)
            h = (rng.normal(size=COMP_FLAT_BLOCKS) + 1j*rng.normal(size=COMP_FLAT_BLOCKS))/np.sqrt(2)
            N0 = 10.0**(-snr/10.0)

            # Equal 16-sample / equal-energy diagnostic.
            noise16a = np.sqrt(N0/2)*(rng.normal(size=(COMP_FLAT_BLOCKS,16)) + 1j*rng.normal(size=(COMP_FLAT_BLOCKS,16)))
            noise16b = np.sqrt(N0/2)*(rng.normal(size=(COMP_FLAT_BLOCKS,16)) + 1j*rng.normal(size=(COMP_FLAT_BLOCKS,16)))
            y_sqm = h[:,None]*Xsqm[idx] + noise16a
            y_cube16 = h[:,None]*CUBE16[idx] + noise16b

            p_sqm = decode_sqm_flat(y_sqm, SQM_FLAT0_BANK)
            p_cube16 = cube_ml_decode(y_cube16, CUBE16)

            # Native CS(4,1): four channel uses, unit-energy source codeword.
            noise4 = np.sqrt(N0/2)*(rng.normal(size=(COMP_FLAT_BLOCKS,4)) + 1j*rng.normal(size=(COMP_FLAT_BLOCKS,4)))
            y_cube4 = h[:,None]*(2.0*CUBE4[idx]) + noise4  # energy T=4, matching unit average power per channel use
            p_cube4 = cube_ml_decode(y_cube4, CUBE4)

            for method, per, uses in [
                ("SQM flat quotient, L=16", np.mean(p_sqm != idx), 16),
                ("Cube-Split CS(4,1), 16-sample isometric embedding", np.mean(p_cube16 != idx), 16),
                ("Cube-Split CS(4,1), native T=4", np.mean(p_cube4 != idx), 4),
            ]:
                rows.append({
                    "snr_db": snr, "seed": seed, "method": method,
                    "PER": float(per), "blocks": COMP_FLAT_BLOCKS,
                    "channel_uses": uses, "payload_bits": 8,
                    "bits_per_channel_use": 8/uses,
                })
    cube_ps = pd.DataFrame(rows)
    cube_ps.to_csv(cube_ps_path, index=False)

cube_summary = summarize_per_seed(cube_ps)
cube_meta = cube_ps.groupby("method", as_index=False)[["channel_uses","payload_bits","bits_per_channel_use"]].first()
cube_summary = cube_summary.merge(cube_meta, on="method", how="left")
cube_summary.to_csv(COMP_DIR / "cube_split_flat_comparison_summary.csv", index=False)
display(cube_summary)

