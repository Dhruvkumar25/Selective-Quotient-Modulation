# Ported from supervisor V8 notebook; see source_map.json.
# Self-contained implementation of Eqs. (21), (22), (34), and (35) on the
# released frozen codebook. The core fractional-delay operator is imported,
# but rank selection, basis bank construction, channel draws, and decoding are
# explicitly implemented here rather than delegated to a publication CSV.

M_IND = 256
L_IND = 16
IND_GRID_DEG = np.arange(-5.0, 6.0, 1.0)
IND_GRID_RAD = np.deg2rad(IND_GRID_DEG)
IND_RANKS = [2, 3, 4, 5, 6, 8]

X_ind = np.load(ROOT / "Artifact" / "fractional_incidence_codebook.npz")["X_opt"]
Xd_ind = X_ind[:, 0] + 1j * X_ind[:, 1]
delays_ind, powers_ind, D_ind = tdla_operators(L_IND)


def ind_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def ind_random_A(rng, N, kappa=1.5):
    slo = np.sqrt(2.0 / (1.0 + kappa * kappa))
    shi = kappa * slo
    out = np.empty((N, 2, 2), dtype=float)
    a = rng.uniform(0.0, 2 * np.pi, N)
    b = rng.uniform(0.0, 2 * np.pi, N)
    for i in range(N):
        out[i] = ind_rot(a[i]) @ np.diag([shi, slo]) @ ind_rot(b[i])
    return out


def ind_build_tdla_basis_bank(rmax=8):
    bank = np.empty((M_IND, len(IND_GRID_RAD), L_IND, rmax), dtype=np.float32)
    for m in range(M_IND):
        for j, omega in enumerate(IND_GRID_RAD):
            F = weighted_nuisance_dictionary(Xd_ind[m], omega, powers_ind, D_ind)
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            bank[m, j] = Vh[:rmax].T.astype(np.float32)
    return bank


def ind_generate_tdla_trace(snr_db, blocks, seed, kappa=1.5):
    rng = np.random.default_rng(seed)
    idx = rng.integers(M_IND, size=blocks)
    h = random_tdla_coefficients(rng, blocks, powers_ind)
    omega = rng.uniform(-np.deg2rad(4.0), np.deg2rad(4.0), blocks)
    A = ind_random_A(rng, blocks, kappa)
    dc_angle = rng.uniform(0.0, 2 * np.pi, blocks)
    d = np.stack([np.cos(dc_angle), np.sin(dc_angle)], axis=1)
    N0 = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(N0 / 2.0) * rng.normal(size=(blocks, 2, L_IND))
    s = exact_fractional_channel(Xd_ind[idx], h, D_ind)
    s *= np.exp(1j * omega[:, None] * np.arange(L_IND)[None, :])
    sr = np.stack([s.real, s.imag], axis=1)
    Y = np.einsum("nij,njk->nik", A, sr, optimize=True) + d[:, :, None] + noise
    return Y, idx


def ind_score_cube(Y, bank, rank=4, batch=32):
    Yc = Y - Y.mean(axis=2, keepdims=True)
    out = np.empty((len(Y), M_IND, len(IND_GRID_RAD)), dtype=np.float32)
    B = bank[..., :rank]
    for s in range(0, len(Y), batch):
        e = min(len(Y), s + batch)
        Z = np.einsum("bal,mwlr->bmwar", Yc[s:e], B, optimize=True)
        out[s:e] = np.sum(Z * Z, axis=(3, 4), dtype=np.float64).astype(np.float32)
    return out


def ind_decode_from_scores(scores):
    return np.argmax(scores.max(axis=2), axis=1)


