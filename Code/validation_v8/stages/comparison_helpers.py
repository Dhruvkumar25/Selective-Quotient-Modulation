# Ported from supervisor V8 notebook; see source_map.json.

# ---------- Shared TDL-A ablation helpers ----------

M5, L5 = 256, 16
GRID5_DEG = np.arange(-5.0, 6.0, 1.0)
GRID5_RAD = np.deg2rad(GRID5_DEG)
R_SEL, R_FULL = 4, 15

X_inc_r = np.load(ROOT / "Artifact" / "fractional_incidence_codebook.npz")["X_opt"]
X_pre_r = np.load(ROOT / "Artifact" / "stage3_codebooks.npz")["X_dmm"]
X_inc = X_inc_r[:, 0] + 1j * X_inc_r[:, 1]
X_pre = X_pre_r[:, 0] + 1j * X_pre_r[:, 1]

delays5, powers5, D5 = tdla_operators(L5)

def comp_rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)

def comp_random_A(rng, N, kappa=1.5):
    slo = np.sqrt(2.0 / (1.0 + kappa*kappa))
    shi = kappa * slo
    a = rng.uniform(0, 2*np.pi, N)
    b = rng.uniform(0, 2*np.pi, N)
    A = np.empty((N, 2, 2), float)
    for i in range(N):
        A[i] = comp_rot(a[i]) @ np.diag([shi, slo]) @ comp_rot(b[i])
    return A

def build_tdla_bank_for_codebook(Xd, rmax=15):
    bank = np.empty((M5, len(GRID5_RAD), L5, rmax), dtype=np.float32)
    for m in range(M5):
        for j, omega in enumerate(GRID5_RAD):
            F = weighted_nuisance_dictionary(Xd[m], omega, powers5, D5)
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            rr = min(rmax, Vh.shape[0])
            bank[m, j, :, :rr] = Vh[:rr].T.astype(np.float32)
            if rr < rmax:
                bank[m, j, :, rr:] = 0.0
    return bank

def draw_common_tdla_nuisance(snr_db, blocks, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(M5, size=blocks)
    h = random_tdla_coefficients(rng, blocks, powers5)
    omega = rng.uniform(-np.deg2rad(4.0), np.deg2rad(4.0), blocks)
    A = comp_random_A(rng, blocks, 1.5)
    dcang = rng.uniform(0, 2*np.pi, blocks)
    d = np.stack([np.cos(dcang), np.sin(dcang)], axis=1)
    N0 = 10.0**(-snr_db/10.0)
    noise = np.sqrt(N0/2.0) * rng.normal(size=(blocks, 2, L5))
    return idx, h, omega, A, d, noise

def apply_tdla_nuisance(Xd, nuisance):
    idx, h, omega, A, d, noise = nuisance
    s = exact_fractional_channel(Xd[idx], h, D5)
    s *= np.exp(1j * omega[:, None] * np.arange(L5)[None, :])
    sr = np.stack([s.real, s.imag], axis=1)
    return np.einsum("nij,njk->nik", A, sr, optimize=True) + d[:, :, None] + noise

def decode_tdla_bank(Y, bank, rank, batch=32):
    Yc = Y - Y.mean(axis=2, keepdims=True)
    B = bank[..., :rank]
    pred = np.empty(len(Y), int)
    for s in range(0, len(Y), batch):
        e = min(len(Y), s+batch)
        Z = np.einsum("bal,mwlr->bmwar", Yc[s:e], B, optimize=True)
        score = np.sum(Z*Z, axis=(3,4))
        pred[s:e] = np.argmax(score.max(axis=2), axis=1)
    return pred


