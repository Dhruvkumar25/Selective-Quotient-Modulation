# Ported from supervisor V8 notebook; see source_map.json.
def normalize_rows_energy(X, energy=8.0):
    X = np.asarray(X, complex).copy()
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X * (np.sqrt(energy) / np.maximum(nrm, 1e-15))

