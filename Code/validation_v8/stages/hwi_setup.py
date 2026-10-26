# Ported from supervisor V8 notebook; see source_map.json.

def line_repel_step(C, mu=0.1):
    C = C.copy()
    Gram = C @ C.conj().T
    np.fill_diagonal(Gram, 0.0)
    partner = np.argmax(np.abs(Gram)**2, axis=1)
    inner = np.sum(C[partner].conj()*C, axis=1)  # c_j^H c_k
    g = C[partner] * inner[:,None]
    # Riemannian projection to the tangent space of the complex unit sphere.
    g -= C * np.sum(C.conj()*g, axis=1)[:,None]
    C -= mu*g
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def iq_perturb_lines(C, g_db, theta_rad):
    G = 10.0**(g_db/20.0)
    a = (1 + G*np.exp(1j*theta_rad))/2
    b = (1 - G*np.exp(-1j*theta_rad))/2
    Z = a*C + b*C.conj()
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)

def cfo_centroid_update(C, rng, cfo_max_deg=10.0, n_cfo=20):
    T = C.shape[1]
    t = np.arange(T)
    out = np.empty_like(C)
    for m, x in enumerate(C):
        om = np.deg2rad(rng.uniform(-cfo_max_deg, cfo_max_deg, n_cfo))
        F = np.exp(1j*om[:,None]*t[None,:]) * x[None,:]
        Pbar = (F.T @ F.conj()) / n_cfo
        _, V = np.linalg.eigh(Pbar)
        out[m] = V[:, -1]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out

def train_hwi_aware_lines(
    C0, seed=20261025, cfo_max_deg=10.0, gmax_db=3.0, theta_max_deg=15.0,
    mu=0.1, niter_cfo=50, ncfo=20, niter_iqi=20, q=5
):
    rng = np.random.default_rng(seed)
    C = C0.copy()
    for outer in range(niter_cfo):
        C = cfo_centroid_update(C, rng, cfo_max_deg, ncfo)
        for it in range(1, niter_iqi+1):
            if it % q == 0:
                gd = float(rng.uniform(0.0, gmax_db))
                th = float(np.deg2rad(rng.uniform(0.0, theta_max_deg)))
                C = iq_perturb_lines(C, gd, th)
            else:
                C = line_repel_step(C, mu)
        if (outer+1) % max(1, niter_cfo//5) == 0:
            Gm = np.abs(C @ C.conj().T)**2
            np.fill_diagonal(Gm, 0.0)
            print(f"HWI design {outer+1:3d}/{niter_cfo}: worst coherence={np.sqrt(Gm.max()):.4f}")
    return C

hwi_code_path = COMP_DIR / "hwi_aware_grassmann_single_stream_codebook.npz"
if hwi_code_path.exists() and not COMP_FORCE:
    HWI16 = np.load(hwi_code_path)["codebook"]
    print("Loaded cached single-stream HWI-aware Grassmannian codebook.")
else:
    print("Training single-stream HWI-aware Grassmannian codebook ...")
    t0 = time.time()
    # Source paper starts from an HWI-unaware Grassmannian codebook.
    # CS(4,1) embedded in C^16 is a deterministic, structured HWI-unaware initializer.
    HWI16 = train_hwi_aware_lines(
        CUBE16 / 4.0,
        niter_cfo=HWI_NITER_CFO,
        ncfo=HWI_NCFO,
        niter_iqi=HWI_NITER_IQI,
        q=5,
    )
    HWI16 *= 4.0
    np.savez_compressed(hwi_code_path, codebook=HWI16)
    print(f"Training completed in {time.time()-t0:.1f} s")

# SQM bank covers the same guarded CFO support used for the matched HWI experiment.
HWI_GRID_DEG = np.arange(-11.0, 12.0, 1.0)
SQM_HWI_BANK = flat_sqm_basis_bank(X_inc, HWI_GRID_DEG)

def apply_hwi_channel(C, idx, h, omega, gdb, theta, noise):
    x = C[idx]
    G = 10.0**(gdb/20.0)
    a = (1 + G*np.exp(1j*theta))/2
    b = (1 - G*np.exp(-1j*theta))/2
    z = a[:,None]*x + b[:,None]*x.conj()
    z *= np.exp(1j*omega[:,None]*np.arange(C.shape[1])[None,:])
    return h[:,None]*z + noise


