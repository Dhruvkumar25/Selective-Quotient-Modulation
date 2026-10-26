# Ported from supervisor V8 notebook; see source_map.json.

def cube_split_cs4_b1():
    T = 4
    # Eq. (9), B_j = 1 -> {1/4, 3/4} for each of 2(T-1)=6 real coordinates.
    vals = np.array([0.25, 0.75])
    coords = np.array(np.meshgrid(*([vals]*6), indexing="ij")).reshape(6, -1).T
    code = []
    labels = []
    for cell in range(T):
        for q, a in enumerate(coords):
            tcoords = []
            for j in range(T-1):
                w = norm.ppf(a[2*j]) + 1j*norm.ppf(a[2*j+1])
                r2 = abs(w)**2
                # Eq. (12)
                amp = np.sqrt((1-np.exp(-r2/2)) / (1+np.exp(-r2/2))) if r2 > 1e-14 else 0.0
                tj = amp * w/abs(w) if abs(w) > 1e-14 else 0.0j
                tcoords.append(tj)
            # Eq. (11): insert "1" in the cell coordinate and normalize.
            x = np.empty(T, complex)
            x[cell] = 1.0
            k = 0
            for i in range(T):
                if i != cell:
                    x[i] = tcoords[k]
                    k += 1
            x /= np.linalg.norm(x)
            code.append(x)
            labels.append((cell, q))
    return np.asarray(code), labels

CUBE4, cube_labels = cube_split_cs4_b1()
assert CUBE4.shape == (256, 4)
assert np.allclose(np.linalg.norm(CUBE4, axis=1), 1.0)

# Pairwise-distance sanity check.
cube_gram = np.abs(CUBE4 @ CUBE4.conj().T)**2
np.fill_diagonal(cube_gram, 0.0)
cube_max_coherence = float(cube_gram.max())
cube_min_chordal = float(np.sqrt(1.0 - cube_max_coherence))

# Fixed semi-unitary 4->16 DFT embedding: preserves all pairwise inner products.
n = np.arange(16)[:, None]
k = np.arange(4)[None, :]
U16x4 = np.exp(2j*np.pi*n*k/16) / 4.0
assert np.allclose(U16x4.conj().T @ U16x4, np.eye(4), atol=1e-12)
CUBE16 = CUBE4 @ U16x4.conj().T
CUBE16 *= 4.0  # equal energy 16 for the 16-sample diagnostic

np.savez_compressed(
    COMP_DIR / "cube_split_cs4_b1_and_embedding.npz",
    cube4=CUBE4,
    cube16=CUBE16,
    max_coherence=cube_max_coherence,
    min_chordal=cube_min_chordal,
)

print("Cube-Split CS(4,1) size:", CUBE4.shape)
print("Minimum chordal distance:", cube_min_chordal)
print("16-sample embedding energy:", np.mean(np.sum(np.abs(CUBE16)**2, axis=1)))

def cube_ml_decode(y, C):
    # Noncoherent line ML: maximize |y^H c|^2.
    return np.argmax(np.abs(y @ C.conj().T)**2, axis=1)

def flat_sqm_basis_bank(Xd, cfo_grid_deg=(0.0,)):
    Xd = np.asarray(Xd)
    L = Xd.shape[1]
    H = np.eye(L) - np.ones((L,L))/L
    grid = np.asarray(cfo_grid_deg, float)
    bank = np.empty((len(Xd), len(grid), L, 2), np.float32)
    t = np.arange(L)
    for m, x in enumerate(Xd):
        for j, deg in enumerate(grid):
            v = x * np.exp(1j*np.deg2rad(deg)*t)
            F = np.stack([v.real, v.imag]) @ H
            _, _, Vh = np.linalg.svd(F, full_matrices=False)
            bank[m,j] = Vh[:2].T.astype(np.float32)
    return bank

SQM_FLAT0_BANK = flat_sqm_basis_bank(X_inc, [0.0])

def decode_sqm_flat(y_complex, bank, batch=64):
    Y = np.stack([y_complex.real, y_complex.imag], axis=1)
    Y -= Y.mean(axis=2, keepdims=True)
    pred = np.empty(len(Y), int)
    for s in range(0, len(Y), batch):
        e = min(len(Y), s+batch)
        Z = np.einsum("bal,mwlr->bmwar", Y[s:e], bank, optimize=True)
        sc = np.sum(Z*Z, axis=(3,4))
        pred[s:e] = np.argmax(sc.max(axis=2), axis=1)
    return pred


