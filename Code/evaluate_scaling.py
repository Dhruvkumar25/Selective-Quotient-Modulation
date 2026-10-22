"""Preliminary SQM scaling stress test beyond 8 bits.

The shipped incidence-trained artifact contains 256 messages.  For 6 and 8 bits
we use nested subsets of that artifact; for 10 bits we generate a reproducible
random centered orthonormal codebook.  The 10-bit result is therefore a
complexity/identifiability stress test, not evidence of an optimized 10-bit
codebook.  A publication-grade 10-bit claim requires incidence retraining.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, weighted_nuisance_dictionary, exact_fractional_channel, random_tdla_coefficients
from run_final_headtohead import randA, apply_iq_dc_noise, decode_dmm
from experiment_config import L, physical_cfo_sample, cfo_grid_rad
ROOT=Path(__file__).resolve().parents[1]


def random_codebook(M, rng):
    H=np.eye(L)-np.ones((L,L))/L
    X=np.empty((M,2,L))
    for m in range(M):
        Z=rng.normal(size=(2,L))@H
        Q=np.linalg.qr(Z.T, mode='reduced')[0].T
        X[m]=np.sqrt(L/2)*Q
    return X


def make_bases(Xd,powers,D,rank):
    grid=cfo_grid_rad(); M=Xd.shape[0]; B=np.empty((M,len(grid),L,rank),np.float32)
    for m in range(M):
        for j,w in enumerate(grid):
            F=weighted_nuisance_dictionary(Xd[m],w,powers,D); _,_,Vh=np.linalg.svd(F,full_matrices=False); B[m,j]=Vh[:rank].T
    return B


def run(bits,snr,blocks,seed,rank=4):
    rng=np.random.default_rng(seed); M=2**bits
    artifact=np.load(ROOT/'Artifact'/'fractional_incidence_codebook.npz')['X_opt']
    source='incidence-trained subset' if M<=artifact.shape[0] else 'random stress-test codebook'
    X=artifact[:M] if M<=artifact.shape[0] else random_codebook(M,rng)
    Xd=X[:,0]+1j*X[:,1]
    delays,powers,D=tdla_operators(L); B=make_bases(Xd,powers,D,rank)
    idx=rng.integers(M,size=blocks); h=random_tdla_coefficients(rng,blocks,powers); om=physical_cfo_sample(rng,blocks)
    A=randA(rng,blocks,1.5); ang=rng.uniform(0,2*np.pi,blocks); d=np.stack([np.cos(ang),np.sin(ang)],1)
    noise=np.sqrt(10**(-snr/10)/2)*rng.normal(size=(blocks,2,L))
    s=exact_fractional_channel(Xd[idx],h,D)*np.exp(1j*om[:,None]*np.arange(L)); Y=apply_iq_dc_noise(s,om,A,d,noise)
    return {'bits':bits,'messages':M,'source':source,'snr_db':snr,'rank':rank,'PER':float(np.mean(decode_dmm(Y,B)!=idx)),
            'cfo_hypotheses':len(cfo_grid_rad()),'score_evaluations':M*len(cfo_grid_rad()),'blocks':blocks}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bits',default='6,8,10'); ap.add_argument('--snr',type=float,default=10); ap.add_argument('--blocks',type=int,default=200); ap.add_argument('--seed',type=int,default=20260929); ap.add_argument('--output',default='Data/scaling_smoke.csv'); a=ap.parse_args()
    rows=[run(int(b),a.snr,a.blocks,a.seed+int(b)) for b in a.bits.split(',')]
    df=pd.DataFrame(rows); print(df.to_string(index=False)); out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print('saved',out)
if __name__=='__main__': main()
