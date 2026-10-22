"""Local manifold diagnostic for the over-invariance argument.

Theorem 2 is a global linear-subspace result.  This script studies the local
received-plane manifolds induced by channel coefficients and CFO.  It computes
finite-difference tangent dimensions and the Grassmann-manifold dimension bound
max(0, d_i+d_j-q(n-q)).  The result is deliberately diagnostic: it is not a
replacement for a global intersection theorem.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, weighted_nuisance_dictionary
from experiment_config import M, L
ROOT = Path(__file__).resolve().parents[1]


def projector(S):
    S = S - S.mean(axis=1, keepdims=True)
    Q = np.linalg.qr(S.T, mode='reduced')[0]
    return Q @ Q.T


def manifold_point(x, B, omega, powers, D):
    F = weighted_nuisance_dictionary(x, omega, powers, D)
    S = B @ F
    return projector(S)


def tangent_jacobian(x, B, omega, powers, D, eps=1e-5):
    P0 = manifold_point(x, B, omega, powers, D)
    cols = []
    # Channel-coordinate perturbations.
    for a in range(B.shape[0]):
        for b in range(B.shape[1]):
            B1 = B.copy(); B1[a,b] += eps
            B2 = B.copy(); B2[a,b] -= eps
            cols.append(((manifold_point(x,B1,omega,powers,D)-manifold_point(x,B2,omega,powers,D))/(2*eps)).ravel())
    # CFO coordinate perturbation.
    cols.append(((manifold_point(x,B,omega+eps,powers,D)-manifold_point(x,B,omega-eps,powers,D))/(2*eps)).ravel())
    J = np.stack(cols, axis=1)
    s = np.linalg.svd(J, compute_uv=False)
    tol = max(J.shape)*max(s[0], 1e-12)*1e-7
    d = int(np.sum(s > tol))
    U = np.linalg.svd(J, full_matrices=False)[0][:,:d]
    return P0, d, U, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--output', default='Data/manifold_geometry.csv')
    a = ap.parse_args(); rng=np.random.default_rng(a.seed)
    X = np.load(ROOT/'Artifact'/'fractional_incidence_codebook.npz')['X_opt']; Xd=X[:,0]+1j*X[:,1]
    _, powers, D = tdla_operators(L)
    q, n = 2, L-1
    ambient = q*(n-q)
    rows=[]
    for case in range(a.pairs):
        i,j = rng.choice(M, size=2, replace=False)
        omega=float(rng.uniform(-math.radians(4),math.radians(4)))
        B=rng.normal(size=(2,2*len(powers)))
        _,di,Ui,si=tangent_jacobian(Xd[i],B,omega,powers,D)
        _,dj,Uj,sj=tangent_jacobian(Xd[j],B,omega,powers,D)
        sv=np.linalg.svd(Ui.T@Uj,compute_uv=False) if di and dj else np.array([])
        principal_angle_min=float(np.arccos(np.clip(sv.max(),-1,1))) if len(sv) else float('nan')
        rows.append({'case':case,'message_i':int(i),'message_j':int(j),'ambient_grassmann_dimension':ambient,
                     'tangent_dim_i':di,'tangent_dim_j':dj,
                     'generic_intersection_lower_bound':max(0,di+dj-ambient),
                     'smallest_principal_angle_rad':principal_angle_min,
                     'max_singular_value_tangent_overlap':float(sv.max()) if len(sv) else np.nan})
    df=pd.DataFrame(rows); print(df.describe().to_string()); out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print('saved',out)

if __name__=='__main__': main()
