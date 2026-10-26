# Ported from supervisor V8 notebook; see source_map.json.
# Equation-level numerical checks.
# These tests do not replace proofs; they catch dimension, rank, covariance,
# and algebra inconsistencies in the actual experimental regime.

from fractional_tdla_core import (
    centering_matrix,
    tdla_operators,
    weighted_nuisance_dictionary,
    exact_fractional_channel,
    random_tdla_coefficients,
)

rng = np.random.default_rng(20261024)
L = 16
H = centering_matrix(L)

# ---------- Eq. (3): centering matrix ----------
err_H_sym = np.linalg.norm(H - H.T)
err_H_idem = np.linalg.norm(H @ H - H)
err_H1 = np.linalg.norm(H @ np.ones(L))

# ---------- Eq. (7) and Theorem 1: exact static affine row-space quotient ----------
def row_projector_fullrow(Z):
    G = Z @ Z.T
    return Z.T @ np.linalg.inv(G) @ Z

Z = rng.normal(size=(2, L))
Z = Z @ H
# Normalize to the paper's centered/equal-row-energy convention.
Gz = Z @ Z.T
w, U = np.linalg.eigh(Gz)
Z = np.sqrt(L / 2.0) * (U @ np.diag(1.0 / np.sqrt(w)) @ U.T) @ Z

A = rng.normal(size=(2, 2))
while abs(np.linalg.det(A)) < 0.2:
    A = rng.normal(size=(2, 2))
d = rng.normal(size=2)
Y_aff = A @ Z + d[:, None]
Qx = row_projector_fullrow(Z @ H)
Qy = row_projector_fullrow(Y_aff @ H)
affine_projector_error = np.max(np.abs(Qx - Qy))

# ---------- Eq. (15): actual 23-path TDL-A full dictionary ----------
X_art = np.load(ROOT / "Artifact" / "fractional_incidence_codebook.npz")["X_opt"]
Xd = X_art[:, 0] + 1j * X_art[:, 1]
delays_tdla, powers_tdla, D_tdla = tdla_operators(L)
Phi = weighted_nuisance_dictionary(Xd[0], 0.0, powers_tdla, D_tdla)
Gram = Phi @ Phi.T
rank_phi = int(np.linalg.matrix_rank(Phi))
rank_gram = int(np.linalg.matrix_rank(Gram))

P_pinv = Phi.T @ np.linalg.pinv(Gram) @ Phi
_, s_phi, Vh_phi = np.linalg.svd(Phi, full_matrices=False)
tol_phi = np.finfo(float).eps * max(Phi.shape) * s_phi[0]
rho = int(np.sum(s_phi > tol_phi))
Vr = Vh_phi[:rho].T
P_svd = Vr @ Vr.T
projector_pinv_svd_error = np.max(np.abs(P_pinv - P_svd))

# ---------- Eq. (17): residual/projection identity ----------
r = 4
Qr = Vh_phi[:r].T
Pr = Qr @ Qr.T
S = rng.normal(size=(2, L)) @ H
rho_fit = np.sum((S @ Pr) ** 2) / np.sum(S ** 2)
residual_ratio = np.linalg.norm(S @ (np.eye(L) - Pr), "fro") / np.linalg.norm(S, "fro")
eq17_error = abs(residual_ratio - np.sqrt(max(0.0, 1.0 - rho_fit)))

# ---------- Theorem 2: rank-saturated lower bound ----------
n = L - 1
r_full = rank_phi
intersection_lower = max(0, 2 * r_full - n)

# ---------- Theorem 3: deterministic triangle-inequality radius sanity ----------
# Construct a true subspace and a separated wrong subspace, then verify that
# random perturbations strictly below Delta/2 do not reverse nearest-subspace order.
Qt = np.linalg.qr(rng.normal(size=(L, 4)))[0]
Qw = np.linalg.qr(rng.normal(size=(L, 4)))[0]
Pt = Qt @ Qt.T
Pw = Qw @ Qw.T
S0 = rng.normal(size=(2, 4)) @ Qt.T

def dist_to_proj(Y, P):
    return float(np.linalg.norm(Y @ (np.eye(L) - P), "fro"))

eps0 = dist_to_proj(S0, Pt)
del0 = dist_to_proj(S0, Pw)
Delta0 = del0 - eps0
radius_check_pass = None
if Delta0 > 1e-10:
    ok = []
    for _ in range(200):
        N = rng.normal(size=S0.shape)
        N /= np.linalg.norm(N, "fro")
        N *= 0.49 * Delta0
        Y = S0 + N
        ok.append(dist_to_proj(Y, Pt) < dist_to_proj(Y, Pw))
    radius_check_pass = bool(np.all(ok))

# ---------- Noise covariance convention after A / after A^{-1} ----------
def normalized_A(kappa=1.5, angle1=0.7, angle2=-0.4):
    c1, s1a = np.cos(angle1), np.sin(angle1)
    c2, s2a = np.cos(angle2), np.sin(angle2)
    R1 = np.array([[c1, -s1a], [s1a, c1]])
    R2 = np.array([[c2, -s2a], [s2a, c2]])
    slo = np.sqrt(2.0 / (1.0 + kappa * kappa))
    shi = kappa * slo
    return R1 @ np.diag([shi, slo]) @ R2

A0 = normalized_A()
sigma2 = 0.05
N = np.sqrt(sigma2) * rng.normal(size=(300000, 2))
cov_received = np.cov(N.T, bias=True)
N_eq = N @ np.linalg.inv(A0).T
cov_equalized = np.cov(N_eq.T, bias=True)
cov_equalized_theory = sigma2 * np.linalg.inv(A0) @ np.linalg.inv(A0).T
noise_received_error = np.max(np.abs(cov_received - sigma2 * np.eye(2)))
noise_equalized_error = np.max(np.abs(cov_equalized - cov_equalized_theory))

# ---------- Eq. (47): widely-linear determinant ----------
alpha = 0.9 + 0.4j
beta = 0.2 - 0.1j
Awl = np.array([
    [np.real(alpha + beta), -np.imag(alpha - beta)],
    [np.imag(alpha + beta),  np.real(alpha - beta)],
])
eq47_error = abs(np.linalg.det(Awl) - (abs(alpha) ** 2 - abs(beta) ** 2))

# ---------- Eqs. (48)-(52): GLRT / projection-energy equivalence ----------
r_glrt = 5
Phi_g = rng.normal(size=(r_glrt, L)) @ H
while np.linalg.matrix_rank(Phi_g) < r_glrt:
    Phi_g = rng.normal(size=(r_glrt, L)) @ H
Yg = rng.normal(size=(2, L)) @ H
Bg = Yg @ Phi_g.T @ np.linalg.inv(Phi_g @ Phi_g.T)
resid_direct = np.linalg.norm(Yg - Bg @ Phi_g, "fro") ** 2
Pg = Phi_g.T @ np.linalg.inv(Phi_g @ Phi_g.T) @ Phi_g
resid_proj = np.linalg.norm(Yg, "fro") ** 2 - np.trace(Yg @ Pg @ Yg.T)
glrt_error = abs(resid_direct - resid_proj)

# ---------- Eq. (36): arithmetic and optional score reuse ----------
M = 256
Omega_full = 11
Omega_coarse = 5
K = 8
scores_eq36 = M * Omega_coarse + K * Omega_full
scores_reuse = M * Omega_coarse + K * (Omega_full - Omega_coarse)

math_audit_ind = pd.DataFrame([
    ["Eq. (3)", "PASS", f"sym={err_H_sym:.2e}, idem={err_H_idem:.2e}, H1={err_H1:.2e}", "No change."],
    ["Eqs. (7)-(10) / Theorem 1", "PASS WITH STATED RANK-2 ASSUMPTION", f"affine projector max error={affine_projector_error:.2e}", "Ordinary inverse is valid for normalized rank-2 codewords; pinv is a safe generic form."],
    ["Eq. (15) full multipath projector", "MANUSCRIPT CORRECTION REQUIRED", f"Phi={Phi.shape}, rank={rank_phi}; Gram rank={rank_gram}/{Gram.shape[0]}; pinv-vs-SVD={projector_pinv_svd_error:.2e}", "Replace inverse by Moore-Penrose pseudoinverse or compact-SVD projector."],
    ["Eqs. (18)-(20) over-invariance barrier", "PASS AFTER RANK-SATURATION QUALIFICATION", f"n={n}, actual r_full={r_full}, lower bound=max(0,2r-n)={intersection_lower}", "State r_full=rank(Phi_c)<=min(2P,L-1). L>=4P is the necessary unsaturated/full-row-rank condition; saturation is even more over-invariant."],
    ["Eq. (17) projection-distance identity", "PASS", f"absolute error={eq17_error:.2e}", "No change."],
    ["Eq. (26) deterministic radius", "PASS", f"random below-radius trials pass={radius_check_pass}", "The theorem is correct for exact true/wrong explanation sets in the stated metric."],
    ["Eq. (27) chi-square tail", "PASS WITH NOISE-CONVENTION QUALIFICATION", "2(L-1) centered real degrees of freedom is consistent with isotropic Gaussian noise on the centered subspace.", "Keep sigma convention explicit (per-real-dimension variance)."],
    ["Eq. (28) diagnostic + following covariance paragraph", "FORMULA DIAGNOSTIC OK; COVARIANCE WORDING REQUIRES CORRECTION", f"post-A isotropic cov error={noise_received_error:.2e}; equalized cov error={noise_equalized_error:.2e}", "With Y=A S0+d1^T+N and N added after A: Cov(N)=sigma^2 I. After A^{-1}: sigma^2 A^{-1}A^{-T}."],
    ["Eqs. (29)-(35)", "PASS", "Projection/incidence and nested-rank decoder identities are dimensionally consistent.", "Rename overloaded Q symbols for readability, not correctness."],
    ["Eq. (36) complexity", "PASS AS IMPLEMENTATION COUNT", f"reported={scores_eq36}; if coarse scores are cached/reused={scores_reuse}", "1368 is correct if all 11 shortlisted scores are recomputed; it is not the minimum possible count."],
    ["Eqs. (37)-(39) adaptive proxies/controller", "PASS AS DEFINITIONS", "gap/residual and hysteresis equations are internally consistent.", "Implementation semantics still require same-packet vs next-packet wording to match code."],
    ["Eq. (41) Grassmann manifold", "QUALIFICATION REQUIRED", "Broad B can be rank-deficient.", "Restrict E_m to regular points with rank(B Phi_m(theta))=2, or describe a stratified union rather than all of E_m subset Gr(2,L-1)."],
    ["Eq. (42) local-overlap heuristic", "PASS AS HEURISTIC", "dim Gr(2,15)=26 and d_i+d_j-26 is the generic transverse-intersection dimension.", "Keep the existing regularity/transversality caveat."],
    ["Eq. (44) constructive B_i/B_j", "PASS WHEN Phi_i,Phi_j HAVE FULL ROW RANK", "P=3 screened construction has rank 6 in the independent independent validation test below.", "State the full-row-rank assumption or use pinv."],
    ["Eqs. (45)-(47) widely-linear real form", "PASS", f"determinant identity error={eq47_error:.2e}", "No change."],
    ["Eqs. (48)-(52) GLRT", "PASS UNDER EXPLICIT FULL-ROW-RANK ASSUMPTION", f"residual/projection identity error={glrt_error:.2e}", "Appendix D already states full row rank; pinv generalization is optional."],
    ["Fig. 5 / Appendix B empirical 'certified radius' wording", "TERMINOLOGY CORRECTION REQUIRED", "Original Appendix B searches only 64 wrong messages.", "Call it a sampled/diagnostic radius; the V2 all-255 discrete-grid screen is stricter but still grid-relative."],
], columns=["item", "status", "numerical_check", "recommended_manuscript_action"])

display(math_audit_ind)
math_audit_ind.to_csv(IND_DIR / "mathematical_audit_ind.csv", index=False)

assert err_H_sym < 1e-12 and err_H_idem < 1e-12 and err_H1 < 1e-12
assert affine_projector_error < 1e-10
assert rank_phi == 15 and rank_gram == 15
assert projector_pinv_svd_error < 1e-5
assert eq17_error < 1e-10
assert radius_check_pass is True
assert noise_received_error < 2e-3
assert noise_equalized_error < 2e-3
assert eq47_error < 1e-12
assert glrt_error < 1e-10

print("\nIND mathematical audit: executable checks completed successfully.")

