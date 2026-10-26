# Ported from supervisor V8 notebook; see source_map.json.
def build_centered_cfo_line_bank(C, grid_deg=GRID5_DEG):
    """Centered unit-norm line templates for a CFO-search Grassmannian receiver."""
    C = np.asarray(C, complex)
    grid_deg = np.asarray(grid_deg, float)
    t = np.arange(C.shape[1])
    B = np.empty((len(C), len(grid_deg), C.shape[1]), complex)
    for m, x in enumerate(C):
        for j, deg in enumerate(grid_deg):
            v = x * np.exp(1j*np.deg2rad(deg)*t)
            v = v - v.mean()
            nv = np.linalg.norm(v)
            B[m, j] = v / max(nv, 1e-15)
    return B

def decode_centered_cfo_line(Y_real, bank, batch=64):
    """Closest native HWI extension: center, search CFO, then noncoherent line ML."""
    yc = Y_real[:, 0] + 1j*Y_real[:, 1]
    yc = yc - yc.mean(axis=1, keepdims=True)
    pred = np.empty(len(yc), int)
    for s in range(0, len(yc), batch):
        e = min(len(yc), s + batch)
        z = np.einsum("bl,mwl->bmw", yc[s:e], bank.conj(), optimize=True)
        score = np.abs(z)**2
        pred[s:e] = np.argmax(score.max(axis=2), axis=1)
    return pred

