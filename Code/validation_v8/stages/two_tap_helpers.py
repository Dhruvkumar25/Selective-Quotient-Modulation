# Ported from supervisor V8 notebook; see source_map.json.
def draw_common_two_tap_hwi_nuisance(snr_db, blocks, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(256, size=blocks)
    z = (
        rng.normal(size=(blocks,2)) + 1j*rng.normal(size=(blocks,2))
    ) / np.sqrt(2)
    h = z * np.sqrt(np.array([0.8, 0.2]))[None,:]
    omega = rng.uniform(-np.deg2rad(4.0), np.deg2rad(4.0), blocks)
    A = comp_random_A(rng, blocks, 1.5)
    dcang = rng.uniform(0, 2*np.pi, blocks)
    d = np.stack([np.cos(dcang), np.sin(dcang)], axis=1)
    N0 = 10.0**(-snr_db/10.0)
    noise = np.sqrt(N0/2.0) * rng.normal(size=(blocks, 2, 16))
    return idx, h, omega, A, d, noise

def apply_two_tap_hwi_nuisance(C, nuisance):
    idx, h, omega, A, d, noise = nuisance
    x = C[idx]
    s = np.empty_like(x, dtype=complex)
    s[:,0] = h[:,0] * x[:,0]
    s[:,1:] = h[:,0,None]*x[:,1:] + h[:,1,None]*x[:,:-1]
    s *= np.exp(1j*omega[:,None]*np.arange(16)[None,:])
    sr = np.stack([s.real, s.imag], axis=1)
    return (
        np.einsum("nij,njk->nik", A, sr, optimize=True)
        + d[:,:,None] + noise
    )

def build_two_tap_selective_bank(C, rmax=4):
    C = np.asarray(C, complex)
    bank = np.zeros((256, len(GRID5_RAD), 16, rmax), np.float32)
    t = np.arange(16)
    H = np.eye(16) - np.ones((16,16))/16
    for m, x in enumerate(C):
        for j, omega in enumerate(GRID5_RAD):
            rot = np.exp(1j*omega*t)
            v0 = x * rot
            v1 = np.zeros(16, complex)
            v1[1:] = x[:-1] * rot[1:]
            F = np.stack([v0.real, v0.imag, v1.real, v1.imag]) @ H
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            rr = min(rmax, Vh.shape[0])
            bank[m,j,:,:rr] = Vh[:rr].T.astype(np.float32)
    return bank

