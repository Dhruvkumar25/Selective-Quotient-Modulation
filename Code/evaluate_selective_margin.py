"""Sample the selective-margin decomposition epsilon_r, delta_r, Delta_r.
The manuscript CSV uses a larger case count and sampled wrong-message set.
"""
from pathlib import Path
import argparse, math, sys
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, weighted_nuisance_dictionary, exact_fractional_channel, random_tdla_coefficients
ROOT=Path(__file__).resolve().parents[1]; M=256; L=16

def banks(Xd,powers,D,ranks,cfo=4):
    grid=np.deg2rad(np.linspace(-cfo-1, cfo+1, int(round(2*cfo+2))+1)); out={}
    for r in ranks:
        A=np.empty((M,len(grid),L,L),np.float32)
        for m in range(M):
            for j,w in enumerate(grid):
                _,_,Vh=np.linalg.svd(weighted_nuisance_dictionary(Xd[m],w,powers,D),full_matrices=False); Q=Vh[:r].T; A[m,j]=Q@Q.T
        out[r]=A
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,default=20); ap.add_argument('--wrong',type=int,default=16)
    ap.add_argument('--ranks',default='2,3,4,5,6,8'); ap.add_argument('--seed',type=int,default=20260918); ap.add_argument('--output',default='')
    a=ap.parse_args(); ranks=[int(x) for x in a.ranks.split(',')]; X=np.load(ROOT/'Artifact/fractional_incidence_codebook.npz')['X_opt']; Xd=X[:,0]+1j*X[:,1]
    _,powers,D=tdla_operators(L); P=banks(Xd,powers,D,ranks); rng=np.random.default_rng(a.seed); rows=[]
    for case in range(a.cases):
        i=int(rng.integers(M)); h=random_tdla_coefficients(rng,1,powers); om=float(rng.uniform(-math.radians(4),math.radians(4)))
        S=exact_fractional_channel(Xd[i:i+1],h,D)[0]*np.exp(1j*om*np.arange(L)); S=np.stack([S.real,S.imag]); S-=S.mean(1,keepdims=True)
        e0=float(np.sum(S*S)); norm=np.sqrt(e0); wrong=np.delete(np.arange(M),i); wrong=rng.choice(wrong,size=min(a.wrong,len(wrong)),replace=False)
        for r in ranks:
            fit_t=np.einsum('al,klm,am->k',S,P[r][i],S); eps=float(np.sqrt(np.maximum(0,e0-fit_t)).min())
            Pw=P[r][wrong].reshape(-1,L,L); fit_w=np.einsum('al,klm,am->k',S,Pw,S); delta=float(np.sqrt(np.maximum(0,e0-fit_w)).min())
            rows.append({'case':case,'rank':r,'true_mismatch_norm':eps/norm,'nearest_wrong_dist_norm':delta/norm,'selective_margin_norm':(delta-eps)/norm})
    df=pd.DataFrame(rows); print(df.groupby('rank').agg({'true_mismatch_norm':'mean','nearest_wrong_dist_norm':'mean','selective_margin_norm':'mean'}).to_string())
    if a.output: df.to_csv(ROOT/a.output,index=False)
if __name__=='__main__': main()
