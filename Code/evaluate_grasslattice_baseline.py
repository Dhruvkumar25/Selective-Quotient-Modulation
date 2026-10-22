"""Reproduce the single-antenna Grass-Lattice flat-fading sanity baseline.

The encoder is an independent Python translation of the M=1 measure-preserving
mapping distributed in the BSD-licensed grassbox toolbox / IEEE TWC 2024 paper.
Detection here is exhaustive unknown-scalar noncoherent GLRT, intentionally
stronger than the structured demapper.  T=5, B=1 gives 8 payload bits.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "Data"
A = ROOT / "Artifact"
X_DMM = np.load(A / "two_tap_incidence_codebook.npz")["X_inc"]
M,_,L = X_DMM.shape
T = 5
BITS = np.array([[(m>>k)&1 for k in range(8)] for m in range(M)], dtype=int)
XD = X_DMM[:,0] + 1j*X_DMM[:,1]


def codebook(alpha=0.19):
    lattice = np.array([alpha,1-alpha])
    X = np.empty((M,T),complex)
    for m in range(M):
        vals = lattice[BITS[m]]
        z = norm.ppf(vals[0::2], scale=1/np.sqrt(2)) + 1j*norm.ppf(vals[1::2], scale=1/np.sqrt(2))
        nz = np.linalg.norm(z)
        fz = chi2.cdf(2*nz*nz, df=2*(T-1))**(1/(2*(T-1))) / nz
        w = z*fz
        X[m] = np.r_[np.sqrt(max(0.0,1-np.vdot(w,w).real)), w]
    Xp = np.zeros((M,L),complex); Xp[:,:T] = X*np.sqrt(L)
    return Xp


def grass_decode(y,X):
    return np.argmax(np.abs(y @ X.conj().T)**2, 1)


def dmm_projectors():
    F = X_DMM-X_DMM.mean(2,keepdims=True)
    Q = np.linalg.qr(F.swapaxes(-2,-1))[0][...,:2]
    return np.einsum("...ik,...jk->...ij",Q,Q).astype(np.float32).reshape(M,L*L)


def dmm_decode(Y,P):
    Y -= Y.mean(2,keepdims=True)
    C = np.einsum("bil,bik->blk",Y,Y).reshape(len(Y),-1).astype(np.float32)
    return np.argmax(C @ P.T,1)


def run(blocks=1800,seeds=(17,73,157,229)):
    Xg=codebook(); P=dmm_projectors(); rows=[]
    for snr in [5,7,10,12,15]:
        vals=[]
        for sd in seeds:
            rng=np.random.default_rng(20260919+snr*1000+sd)
            idx=rng.integers(M,size=blocks)
            h=(rng.normal(size=blocks)+1j*rng.normal(size=blocks))/np.sqrt(2)
            N0=10**(-snr/10)
            n=np.sqrt(N0/2)*(rng.normal(size=(blocks,L))+1j*rng.normal(size=(blocks,L)))
            yg=h[:,None]*Xg[idx]+n
            yd=h[:,None]*XD[idx]+n
            vals.append([np.mean(dmm_decode(np.stack([yd.real,yd.imag],1),P)!=idx),
                         np.mean(grass_decode(yg,Xg)!=idx)])
        vals=np.asarray(vals)
        for j,name in enumerate(["Incidence-trained SQM","Grass-Lattice exhaustive GLRT"]):
            x=vals[:,j]
            rows.append({"snr_db":snr,"method":name,"PER_mean":x.mean(),
                         "PER_95CI_halfwidth":1.96*x.std(ddof=1)/np.sqrt(len(x)),
                         "blocks":blocks*len(x)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--blocks",type=int,default=1800)
    ap.add_argument("--out",default=str(D/"dmm_vs_grasslattice_flat_rerun.csv")); args=ap.parse_args()
    df=run(args.blocks); df.to_csv(args.out,index=False); print(df.to_string(index=False))
