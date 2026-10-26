# Ported from supervisor V8 notebook; see source_map.json.
hwi_ps_path = COMP_DIR / "flat_hwi_comparison_per_seed.csv"
if hwi_ps_path.exists() and not COMP_FORCE:
    hwi_ps = pd.read_csv(hwi_ps_path)
else:
    rows = []
    for snr in COMP_SNRS:
        for seed in COMP_SEEDS:
            rng = np.random.default_rng(20263000 + int(snr)*1000 + seed)
            B = COMP_FLAT_BLOCKS
            idx = rng.integers(256, size=B)
            h = (rng.normal(size=B)+1j*rng.normal(size=B))/np.sqrt(2)
            omega = np.deg2rad(rng.uniform(-10.0, 10.0, B))
            gdb = rng.uniform(0.0, 3.0, B)
            theta = np.deg2rad(rng.uniform(0.0, 15.0, B))
            N0 = 10.0**(-snr/10.0)

            # Independent receiver noises, same nuisance draws.
            noises = [
                np.sqrt(N0/2)*(rng.normal(size=(B,16))+1j*rng.normal(size=(B,16)))
                for _ in range(3)
            ]

            Ysqm = apply_hwi_channel(X_inc, idx, h, omega, gdb, theta, noises[0])
            Yhwi = apply_hwi_channel(HWI16, idx, h, omega, gdb, theta, noises[1])
            Ycube = apply_hwi_channel(CUBE16, idx, h, omega, gdb, theta, noises[2])

            p_sqm = decode_sqm_flat(Ysqm, SQM_HWI_BANK)
            p_hwi = cube_ml_decode(Yhwi, HWI16)
            p_cube = cube_ml_decode(Ycube, CUBE16)

            for method, pred in [
                ("SQM quotient + CFO search", p_sqm),
                ("HWI-aware Grassmannian, single-stream adaptation", p_hwi),
                ("HWI-unaware Cube-Split embedding", p_cube),
            ]:
                rows.append({
                    "snr_db": snr, "seed": seed, "method": method,
                    "PER": float(np.mean(pred != idx)),
                    "blocks": B,
                    "payload_bits": 8,
                    "blocklength": 16,
                    "cfo_max_deg_per_sample": 10.0,
                    "iq_amp_max_db": 3.0,
                    "iq_phase_max_deg": 15.0,
                })
                print(f"HWI SNR={snr:>4} seed={seed:>3} {method:48s} PER={rows[-1]['PER']:.5f}")
    hwi_ps = pd.DataFrame(rows)
    hwi_ps.to_csv(hwi_ps_path, index=False)

hwi_summary = summarize_per_seed(hwi_ps)
hwi_summary.to_csv(COMP_DIR / "flat_hwi_comparison_summary.csv", index=False)
display(hwi_summary)

