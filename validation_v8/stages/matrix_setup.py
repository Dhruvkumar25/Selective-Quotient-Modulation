# Ported from supervisor V8 notebook; see source_map.json.
# ---------- V8 complete 1+4 comparison matrix: configuration and helpers ----------

V8_DIR = COMP_DIR / "complete_1plus4_matrix"
V8_DIR.mkdir(parents=True, exist_ok=True)

V8_SCALE = os.environ.get("SQM_V8_SCALE", "manuscript").strip().lower()
V8_FORCE = False

if V8_SCALE == "publication":
    # High-confidence final run: 10 independent seeds x 1000 packets/SNR.
    V8_SEEDS = [13, 59, 151, 233, 347, 431, 587, 719, 863, 997]
    V8_BLOCKS = 1000
elif V8_SCALE == "manuscript":
    # Practical manuscript run with more independent seeds than V7.
    V8_SEEDS = [13, 59, 151, 233, 347]
    V8_BLOCKS = 600
else:
    V8_SEEDS = [13]
    V8_BLOCKS = 60

V8_SNRS = list(COMP_SNRS)
V8_METHODS = ["SQM", "HWI-Grassmannian", "Cube-Split", "SE-MOCZ", "IM-MOCZ"]
V8_MARKERS = ["o", "s", "^", "D", "v"]
V8_BITS = 8
V8_ENERGY = float(V8_BITS)  # Eb = 1.
V8_CFO_MAX_DEG = 4.0

print("V8 scale:", V8_SCALE)
print("V8 seeds:", V8_SEEDS)
print("V8 packets per seed/SNR:", V8_BLOCKS)

# Equal-energy waveform representatives. Native lengths are retained for MOCZ.
V8_X_SQM = normalize_rows_energy(X_inc, V8_ENERGY)
V8_X_HWI = normalize_rows_energy(HWI16, V8_ENERGY)
V8_X_CUBE = normalize_rows_energy(CUBE16, V8_ENERGY)

# Reuse held-out MOCZ radii from the earlier validation.
V8_X_SE_POLY, _ = bmocz_codebook(8, R_SE, total_energy=1.0)
V8_SE_SHAPED = np.stack([np.convolve(x, SE_G) for x in V8_X_SE_POLY])
V8_SE_SCALE = np.sqrt(V8_ENERGY) / np.maximum(np.linalg.norm(V8_SE_SHAPED, axis=1), 1e-15)
V8_X_SE = V8_X_SE_POLY * V8_SE_SCALE[:, None]
V8_X_IM, _ = im_mocz_codebook(8, 6, R_IM, total_energy=V8_ENERGY)
V8_X_IM = normalize_rows_energy(V8_X_IM, V8_ENERGY)

# CFO-search banks for the flat HWI panel.
V8_CFO_GRID_DEG = GRID5_DEG.copy()
V8_SQM_FLAT_HWI_BANK = flat_sqm_basis_bank(V8_X_SQM, V8_CFO_GRID_DEG)
V8_HWI_LINE_BANK = build_centered_cfo_line_bank(V8_X_HWI, V8_CFO_GRID_DEG)
V8_CUBE_LINE_BANK = build_centered_cfo_line_bank(V8_X_CUBE, V8_CFO_GRID_DEG)

# Matched 16-sample two-tap rank-4 banks for SQM, HWI and Cube.
def v8_load_or_build_two_tap_bank(name, C):
    p = V8_DIR / f"two_tap_bank_{name}_r4.npz"
    if p.exists() and not V8_FORCE:
        return np.load(p)["bank"]
    b = build_two_tap_selective_bank(C, 4)
    np.savez_compressed(p, bank=b)
    return b

V8_SQM_2T_BANK = v8_load_or_build_two_tap_bank("sqm", V8_X_SQM)
V8_HWI_2T_BANK = v8_load_or_build_two_tap_bank("hwi", V8_X_HWI)
V8_CUBE_2T_BANK = v8_load_or_build_two_tap_bank("cube", V8_X_CUBE)


def v8_summary(df):
    return summarize_per_seed(df, group_cols=("panel", "snr_db", "method"))


def v8_draw_flat_common(snr_db, blocks, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(256, size=blocks)
    h = (rng.normal(size=blocks) + 1j*rng.normal(size=blocks))/np.sqrt(2)
    N0 = 10.0**(-float(snr_db)/10.0)
    return rng, idx, h, N0


def v8_real_affine_observation(C, idx, s_complex, omega, A, d, N0, rng):
    """Apply CFO -> static real 2x2 I/Q mixing -> DC -> post-front-end AWGN."""
    L = s_complex.shape[1]
    z = s_complex * np.exp(1j*omega[:,None]*np.arange(L)[None,:])
    zr = np.stack([z.real, z.imag], axis=1)
    noise = np.sqrt(N0/2.0) * rng.normal(size=(len(idx), 2, L))
    return np.einsum("nij,njk->nik", A, zr, optimize=True) + d[:,:,None] + noise


def v8_two_tap_signal(C, idx, h2):
    x = C[idx]
    y = np.zeros_like(x, complex)
    y[:,0] = h2[:,0] * x[:,0]
    y[:,1:] = h2[:,0,None]*x[:,1:] + h2[:,1,None]*x[:,:-1]
    return y


def v8_complex_to_real(Y):
    return np.stack([Y.real, Y.imag], axis=1)


def v8_append(rows, panel, snr, seed, method, pred, idx, receiver, calibration, obs_len):
    rows.append({
        "panel": panel,
        "snr_db": float(snr),
        "seed": int(seed),
        "method": method,
        "PER": float(np.mean(pred != idx)),
        "blocks": int(len(idx)),
        "payload_bits": V8_BITS,
        "observation_samples": int(obs_len),
        "bits_per_observation_sample": float(V8_BITS/obs_len),
        "receiver": receiver,
        "calibration_assumption": calibration,
    })

comparison_rules = pd.DataFrame([
    ["A: flat clean", "SQM", "quotient decoder", "none"],
    ["A: flat clean", "HWI-Grassmannian", "noncoherent line ML", "none"],
    ["A: flat clean", "Cube-Split", "noncoherent line ML", "none"],
    ["A: flat clean", "SE-MOCZ", "source-model Gaussian marginal ML", "none"],
    ["A: flat clean", "IM-MOCZ", "DiZeT + majority vote", "none"],
    ["B: flat + HWI", "SQM", "centered quotient + CFO search", "none"],
    ["B: flat + HWI", "HWI-Grassmannian", "centered CFO-search line ML", "none"],
    ["B: flat + HWI", "Cube-Split", "centered CFO-search line ML", "none"],
    ["B: flat + HWI", "SE-MOCZ", "source-model Gaussian marginal ML", "oracle front-end cleanup"],
    ["B: flat + HWI", "IM-MOCZ", "DiZeT + majority vote", "oracle front-end cleanup"],
    ["C: two-tap clean", "SQM", "matched rank-4 two-tap quotient", "none"],
    ["C: two-tap clean", "HWI-Grassmannian", "matched rank-4 two-tap quotient", "none"],
    ["C: two-tap clean", "Cube-Split", "matched rank-4 two-tap quotient", "none"],
    ["C: two-tap clean", "SE-MOCZ", "two-tap Gaussian marginal ML", "none"],
    ["C: two-tap clean", "IM-MOCZ", "two-tap DiZeT + majority vote", "none"],
    ["D: two-tap + HWI", "SQM", "matched rank-4 two-tap quotient + CFO search", "none"],
    ["D: two-tap + HWI", "HWI-Grassmannian", "matched rank-4 two-tap quotient + CFO search", "none"],
    ["D: two-tap + HWI", "Cube-Split", "matched rank-4 two-tap quotient + CFO search", "none"],
    ["D: two-tap + HWI", "SE-MOCZ", "two-tap Gaussian marginal ML", "oracle front-end cleanup"],
    ["D: two-tap + HWI", "IM-MOCZ", "two-tap DiZeT + majority vote", "oracle front-end cleanup"],
], columns=["panel", "scheme", "receiver", "hardware_calibration"])
comparison_rules.to_csv(V8_DIR / "comparison_rules.csv", index=False)
display(comparison_rules)

