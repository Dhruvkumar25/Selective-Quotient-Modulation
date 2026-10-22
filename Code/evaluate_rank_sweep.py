"""Evaluate selective nuisance-rank PER on the exact fractional TDL-A model.
Default is a short smoke run; use more blocks/seeds for manuscript-scale estimates.
"""
from pathlib import Path
import argparse, math, sys
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, weighted_nuisance_dictionary, exact_fractional_channel, random_tdla_coefficients

ROOT=Path(__file__).resolve().parents[1]; M=256; L=16

def rot(th):
    c,s=np.cos(th),np.sin(th); return np.stack([np.stack([c,-s],-1),np.stack([s,c],-1)],-2)

def randA(rng,N,kappa=1.5):
    a=rng.uniform(0,2*np.pi,N); b=rng.uniform(0,2*np.pi,N); s2=np.sqrt(2/(1+kappa*kappa)); s1=kappa*s2
    A=np.empty((N,2,2))
    for n in range(N): A[n]=rot(a[n])@np.diag([s1,s2])@rot(b[n])
    return A

def bases(Xd,powers,D,r,cfo=4):
    deg=np.linspace(-cfo-1, cfo+1, int(round(2*cfo+2))+1); grid=np.deg2rad(deg); B=np.empty((M,len(grid),L,r),np.float32)
    for m in range(M):
        for j,w in enumerate(grid):
            _,_,Vh=np.linalg.svd(weighted_nuisance_dictionary(Xd[m],w,powers,D),full_matrices=False); B[m,j]=Vh[:r].T
    return B

def decode(Y,B):
    Y=Y-Y.mean(2,keepdims=True); out=np.empty(len(Y),int)
    for b in range(len(Y)):
        Z=np.einsum('al,mklr->mkar',Y[b],B,optimize=True); sc=np.sum(Z*Z,axis=(2,3)); out[b]=np.unravel_index(np.argmax(sc),sc.shape)[0]
    return out

def eval_rank(r,snr,N,seed,Xd,powers,D,B):
    rng=np.random.default_rng(seed); idx=rng.integers(M,size=N); h=random_tdla_coefficients(rng,N,powers)
    om=rng.uniform(-math.radians(4),math.radians(4),N); A=randA(rng,N); ang=rng.uniform(0,2*np.pi,N); dc=np.stack([np.cos(ang),np.sin(ang)],1)
    noise=np.sqrt(10**(-snr/10)/2)*rng.normal(size=(N,2,L)); s=exact_fractional_channel(Xd[idx],h,D)*np.exp(1j*om[:,None]*np.arange(L))
    Y=np.einsum('nij,njk->nik',A,np.stack([s.real,s.imag],1))+dc[:,:,None]+noise; return np.mean(decode(Y,B)!=idx)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ranks',default='3,4,5,6,8'); ap.add_argument('--snr',type=float,default=10)
    ap.add_argument('--blocks',type=int,default=100); ap.add_argument('--seeds',default='17'); ap.add_argument('--output',default='')
    a=ap.parse_args(); ranks=[int(x) for x in a.ranks.split(',')]; seeds=[int(x) for x in a.seeds.split(',')]
    X=np.load(ROOT/'Artifact/fractional_incidence_codebook.npz')['X_opt']; Xd=X[:,0]+1j*X[:,1]; _,powers,D=tdla_operators(L)
    rows=[]
    for r in ranks:
        B=bases(Xd,powers,D,r); x=np.array([eval_rank(r,a.snr,a.blocks,20260918+100*r+s,Xd,powers,D,B) for s in seeds])
        ci=0.0 if len(x)<2 else 1.96*x.std(ddof=1)/np.sqrt(len(x)); rows.append({'rank':r,'snr_db':a.snr,'PER_mean':x.mean(),'PER_95CI_halfwidth':ci,'blocks':a.blocks*len(seeds)})
    df=pd.DataFrame(rows); print(df.to_string(index=False))
    if a.output: df.to_csv(ROOT/a.output,index=False)
if __name__=='__main__': main()
