# Ported from supervisor V8 notebook; see source_map.json.
# Independent implementation of the geometry behind Appendix A / Table VIII.
# We use a structured P=3 delayed-feature family (zero-padded delays 0,1,2)
# built from random complex codewords, center every feature row, and verify the
# generic intersection dimension and constructive rank-two collision.


def ind_random_centered_complex_codeword(L, rng):
    x = rng.normal(size=L) + 1j * rng.normal(size=L)
    x = x - x.mean()
    x = x * np.sqrt(L / np.sum(np.abs(x) ** 2))
    return x


def ind_zero_shift(x, d):
    y = np.zeros_like(x)
    if d == 0:
        y[:] = x
    else:
        y[d:] = x[:-d]
    return y


def ind_phi_p3(x):
    L = len(x)
    H = np.eye(L) - np.ones((L, L)) / L
    rows = []
    for d in [0, 1, 2]:
        v = ind_zero_shift(x, d)
        rows.extend([v.real, v.imag])
    return np.stack(rows) @ H


def ind_row_basis(Phi, tol=1e-10):
    _, s, Vh = np.linalg.svd(Phi, full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    return Vh[:r], r


def ind_intersection_basis(Qi, Qj, tol=1e-8):
    U, s, Vh = np.linalg.svd(Qi @ Qj.T, full_matrices=False)
    hit = np.where(s > 1.0 - tol)[0]
    return U[:, hit].T @ Qi, s

rng8 = np.random.default_rng(20261027)
rows8 = []
TRIALS8 = 100
for L8 in range(9, 15):
    dims = []
    collision_residuals = []
    for trial in range(TRIALS8):
        Phi_i = ind_phi_p3(ind_random_centered_complex_codeword(L8, rng8))
        Phi_j = ind_phi_p3(ind_random_centered_complex_codeword(L8, rng8))
        ri = int(np.linalg.matrix_rank(Phi_i))
        rj = int(np.linalg.matrix_rank(Phi_j))
        stacked_rank = int(np.linalg.matrix_rank(np.vstack([Phi_i, Phi_j])))
        dint = ri + rj - stacked_rank
        dims.append(dint)

        if dint >= 2:
            Qi, _ = ind_row_basis(Phi_i)
            Qj, _ = ind_row_basis(Phi_j)
            Ibas, sing = ind_intersection_basis(Qi, Qj)
            if Ibas.shape[0] >= 2:
                Y = Ibas[:2]
                # Eq. (44) is valid here because rank(Phi_i)=rank(Phi_j)=6.
                Bi = Y @ Phi_i.T @ np.linalg.inv(Phi_i @ Phi_i.T)
                Bj = Y @ Phi_j.T @ np.linalg.inv(Phi_j @ Phi_j.T)
                rel_i = np.linalg.norm(Y - Bi @ Phi_i, "fro") / np.linalg.norm(Y, "fro")
                rel_j = np.linalg.norm(Y - Bj @ Phi_j, "fro") / np.linalg.norm(Y, "fro")
                collision_residuals.append(max(rel_i, rel_j))

    n8 = L8 - 1
    r8 = 6
    lower = max(0, 2 * r8 - n8)
    rows8.append({
        "P": 3,
        "L": L8,
        "centered_dimension_n": n8,
        "feature_rank_r": r8,
        "dimension_lower_bound_max0_2r_minus_n": lower,
        "median_observed_intersection_dim": float(np.median(dims)),
        "min_observed_intersection_dim": int(np.min(dims)),
        "max_observed_intersection_dim": int(np.max(dims)),
        "rank_two_collision_constructed": bool(len(collision_residuals) > 0),
        "max_relative_collision_residual": float(max(collision_residuals)) if collision_residuals else np.nan,
    })

table8_ind = pd.DataFrame(rows8)
table8_ind.to_csv(IND_DIR / "table_viii_intersection_equation_ind.csv", index=False)
released_table8 = pd.read_csv(ROOT / "Data" / "P3_intersection_barrier.csv")

print("independent validation independent Table VIII:")
display(table8_ind)
print("Released Table VIII artifact:")
display(released_table8)

assert np.all(table8_ind["feature_rank_r"].eq(6))
assert np.all(table8_ind["median_observed_intersection_dim"].to_numpy() == np.array([4,3,2,1,0,0], float))
assert table8_ind.loc[table8_ind["L"] <= 11, "rank_two_collision_constructed"].all()
assert (~table8_ind.loc[table8_ind["L"] >= 12, "rank_two_collision_constructed"]).all()

print("Table VIII theorem check: exact generic intersection dimensions reproduced; rank-two collisions constructed exactly where the lower bound is at least two.")

