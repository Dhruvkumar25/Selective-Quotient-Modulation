# Ported from supervisor V8 notebook; see source_map.json.

# ---------- MOCZ family helpers ----------

PDP2 = np.array([0.8, 0.2], float)
E_MATCH = 10.0

# Released 10-sample SQM comparison artifact.
X_2TAP_R = np.load(ROOT / "Artifact" / "two_tap_incidence_codebook.npz")["X_inc"]
X_2TAP = X_2TAP_R[:,0] + 1j*X_2TAP_R[:,1]

def bits_lsb(M=256, K=8):
    return np.array([[(m>>k)&1 for k in range(K)] for m in range(M)], dtype=np.int8)

BITS8_LSB = bits_lsb()

def bmocz_codebook(K=8, R=None, total_energy=E_MATCH):
    if R is None:
        R = np.sqrt(1 + np.sin(np.pi/K))
    M = 2**K
    B = bits_lsb(M, K)
    ang = 2*np.pi*np.arange(K)/K
    X = np.empty((M, K+1), complex)
    for m in range(M):
        roots = np.where(B[m] == 1, R, 1/R) * np.exp(1j*ang)
        x = np.poly(roots)[::-1]
        X[m] = x / np.linalg.norm(x) * np.sqrt(total_energy)
    return X, R

def im_mocz_codebook(N=8, K=6, R=None, total_energy=E_MATCH):
    if R is None:
        R = np.sqrt(1 + np.sin(np.pi/K))
    M = 2**N
    nc = 2**(N-K)
    X = np.empty((M, K+1), complex)
    for m in range(M):
        bits = np.array([(m>>(N-1-j))&1 for j in range(N)], dtype=int)  # MSB first
        ci = 0
        for b in bits[:N-K]:
            ci = (ci << 1) | int(b)
        theta_i = 2*np.pi*ci/(K*nc)
        radial = bits[N-K:]
        roots = np.where(radial == 1, R, 1/R) * np.exp(
            1j*(2*np.pi*np.arange(K)/K + theta_i)
        )
        x = np.poly(roots)[::-1]
        X[m] = x / np.linalg.norm(x) * np.sqrt(total_energy)
    return X, R

def conv_design(x, g, Lh):
    s = np.convolve(x, g)
    U = np.zeros((len(s)+Lh-1, Lh), complex)
    for p in range(Lh):
        U[p:p+len(s), p] = s
    return U

def raised_cosine_samples(integer_lags, tau=0.9, beta=0.3):
    # Matched SRRC Tx/Rx gives a raised-cosine correlation pulse.
    t = np.asarray(integer_lags, float) * tau
    den = 1 - (2*beta*t)**2
    out = np.empty_like(t)
    regular = np.abs(den) > 1e-7
    out[regular] = np.sinc(t[regular]) * np.cos(np.pi*beta*t[regular]) / den[regular]
    # Stable numerical limit at the removable singularity.
    for q in np.where(~regular)[0]:
        tt = t[q] + 1e-6
        out[q] = np.sinc(tt) * np.cos(np.pi*beta*tt) / (1-(2*beta*tt)**2)
    return out

def build_gaussian_marginal_bank(X, g, pdp, N0, tau_noise=None, beta=0.3):
    pdp = np.asarray(pdp, float)
    Lh = len(pdp)
    U = np.stack([conv_design(x, g, Lh) for x in X])  # M x N x Lh
    N = U.shape[1]

    if tau_noise is None:
        Rw = np.eye(N)
    else:
        first = raised_cosine_samples(np.arange(N), tau=tau_noise, beta=beta)
        Rw = toeplitz(first)
        lam_min = np.linalg.eigvalsh(Rw).min()
        if lam_min < 1e-8:
            Rw += np.eye(N)*(1e-8-lam_min+1e-10)

    Winv = np.linalg.inv(N0*Rw)
    WU = np.einsum("ab,mbj->maj", Winv, U, optimize=True)
    G = np.einsum("mai,maj->mij", U.conj(), WU, optimize=True)

    Dinv = np.diag(1/pdp)
    Binv = np.linalg.inv(Dinv[None,:,:] + G)
    sqrtD = np.diag(np.sqrt(pdp))
    DG = np.einsum("ij,mjk,kl->mil", sqrtD, G, sqrtD, optimize=True)
    logterm = np.array([
        np.linalg.slogdet(np.eye(Lh) + DG[m])[1].real for m in range(len(X))
    ])
    return dict(U=U, Rw=Rw, Winv=Winv, Binv=Binv, logterm=logterm)

def decode_gaussian_marginal(y, bank, batch=96):
    U, Winv, Binv, logterm = bank["U"], bank["Winv"], bank["Binv"], bank["logterm"]
    pred = np.empty(len(y), int)
    for s in range(0, len(y), batch):
        yy = y[s:s+batch]
        Wy = yy @ Winv.T
        z = np.einsum("mni,bn->bmi", U.conj(), Wy, optimize=True)
        gain = np.einsum("bmi,mij,bmj->bm", z.conj(), Binv, z, optimize=True).real
        metric = logterm[None,:] - gain
        pred[s:s+len(yy)] = np.argmin(metric, axis=1)
    return pred

def simulate_from_marginal_bank(idx, h, bank, N0, rng):
    U = bank["U"]
    y = np.einsum("bni,bi->bn", U[idx], h, optimize=True)
    Lc = np.linalg.cholesky(bank["Rw"])
    z = (rng.normal(size=y.shape) + 1j*rng.normal(size=y.shape))/np.sqrt(2)
    y += np.sqrt(N0) * (z @ Lc.T)
    return y

def conv_channel_batch(Xsel, h):
    B, Lx = Xsel.shape
    Lh = h.shape[1]
    y = np.zeros((B, Lx+Lh-1), complex)
    for p in range(Lh):
        y[:, p:p+Lx] += h[:,p,None]*Xsel
    return y

def im_dizet_decode(y, N=8, K=6, R=1.2):
    nc = 2**(N-K)
    theta_i = 2*np.pi*np.arange(nc)/(K*nc)
    base = 2*np.pi*np.arange(K)/K
    outer = R*np.exp(1j*(theta_i[:,None] + base[None,:]))
    inner = (1/R)*np.exp(1j*(theta_i[:,None] + base[None,:]))
    pred = np.empty(len(y), int)

    for b, yy in enumerate(y):
        roots = np.roots(yy[::-1])
        Mroot = len(roots)
        pout = np.empty((nc,K), float)
        pin = np.empty((nc,K), float)
        for i in range(nc):
            for k in range(K):
                pout[i,k] = np.prod(np.abs(outer[i,k] - roots))
                pin[i,k] = (R**Mroot)*np.prod(np.abs(inner[i,k] - roots))
        p = np.minimum(pout, pin)
        sector_vote = np.argmin(p, axis=0)
        counts = np.bincount(sector_vote, minlength=nc)
        ci = int(np.argmax(counts))  # deterministic tie handling
        radial = (pout[ci] < pin[ci]).astype(int)  # outer=1, inner=0

        bits = [(ci>>(N-K-1-j))&1 for j in range(N-K)] + radial.tolist()
        m = 0
        for bit in bits:
            m = (m << 1) | int(bit)
        pred[b] = m
    return pred

# Released SQM two-tap rank-4 projector decoder, written locally for comparative study.
def two_tap_projector_bank():
    M, L = 256, 10
    arr = np.empty((M, L, L), np.float32)
    for m in range(M):
        U = X_2TAP_R[m].copy()
        V = np.zeros_like(U)
        V[:,1:] = U[:,:-1]
        F = np.concatenate([U, V], axis=0)
        F -= F.mean(axis=-1, keepdims=True)
        Q = np.linalg.qr(F.T)[0][:,:4]
        arr[m] = (Q @ Q.T).astype(np.float32)
    return arr.reshape(256, 100)

P2T = two_tap_projector_bank()


def finite_block_two_tap(Xsel, h):
    """Released SQM two-tap convention: fixed 10-sample observation block."""
    y = np.zeros_like(Xsel, dtype=complex)
    y[:,0] = h[:,0] * Xsel[:,0]
    y[:,1:] = h[:,0,None]*Xsel[:,1:] + h[:,1,None]*Xsel[:,:-1]
    return y

def decode_sqm_two_tap(y):
    Y = np.stack([y.real, y.imag], axis=1)
    Y -= Y.mean(axis=2, keepdims=True)
    C = np.einsum("bil,bik->blk", Y, Y).reshape(len(Y), -1).astype(np.float32)
    return np.argmax(C @ P2T.T, axis=1)

# SE-MOCZ paper source parameters used in the matched adaptation.
SE_TAU = 0.9
SE_BETA = 0.3
SE_LG = 8  # source reports 2*Lg+1 = 17 at tau=0.9
SE_G = raised_cosine_samples(np.arange(-SE_LG, SE_LG+1), tau=SE_TAU, beta=SE_BETA)

print("SE pulse-model taps:", len(SE_G), "center=", SE_G[SE_LG])

