"""Condition-number control with covariance-aware OFDM detection.

The old plot used a scalar complex-noise metric after applying A^{-1}.  This
script keeps ||A||_F^2=2 and evaluates the same known-A pilot receiver with an
augmented improper-noise pilot update and a 2-D Mahalanobis data metric.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, random_tdla_coefficients
from run_final_headtohead import M,L,NFFT,CP,PILOT_BINS,DATA_BINS,make_ofdm_codebooks,randA,apply_iq_dc_noise,ofdm_received,decode_practical
ROOT=Path(__file__).resolve().parents[1]

def A_norm(k):
    return np.diag([math.sqrt(2)*k/math.sqrt(1+k*k), math.sqrt(2)/math.sqrt(1+k*k)])

def inv2(A):
    det=A[...,0,0]*A[...,1,1]-A[...,0,1]*A[...,1,0]; R=np.empty_like(A)
    R[...,0,0]=A[...,1,1]/det; R[...,1,1]=A[...,0,0]/det; R[...,0,1]=-A[...,0,1]/det; R[...,1,0]=-A[...,1,0]/det
    return R

def decode_covaware(Y,A,d,N0,Fscaled,powers,delays):
    IA=inv2(A); z=np.einsum('nij,njk->nik',IA,Y-d[:,:,None]); z=z[:,0]+1j*z[:,1]
    corr=np.sum(np.conj(z[:,:CP])*z[:,NFFT:NFFT+CP],axis=1); omhat=np.angle(corr)/NFFT
    z*=np.exp(-1j*omhat[:,None]*np.arange(L)); R=np.fft.fft(z[:,CP:],axis=1,norm='ortho')
    E=np.exp(-1j*2*np.pi*np.arange(NFFT)[:,None]*delays[None,:]/NFFT); RH=(E*powers[None,:])@E.conj().T
    pilot_amp2=float(np.mean(np.abs(Fscaled[0,PILOT_BINS])**2))
    Rpp=RH[np.ix_(PILOT_BINS,PILOT_BINS)]; Rap=RH[:,PILOT_BINS]
    Hhat=np.empty((len(Y),NFFT),complex); invsig=[]
    for n in range(len(Y)):
        Sigma=(N0/2)*IA[n]@IA[n].T
        c=np.trace(Sigma); # After the unitary FFT, non-DC subcarriers have proper complex noise;
        # the relevant variance is the trace of the real I/Q covariance.
        p=0.0
        Cpp=Rpp+(c/pilot_amp2)*np.eye(len(PILOT_BINS))
        P=(p/pilot_amp2)*np.eye(len(PILOT_BINS))
        Caug=np.block([[Cpp,P],[P.conj(),Cpp.conj()]])
        zaug=np.r_[R[n,PILOT_BINS]/Fscaled[0,PILOT_BINS],(R[n,PILOT_BINS]/Fscaled[0,PILOT_BINS]).conj()]
        Haug=np.concatenate((Rap,np.zeros_like(Rap)),axis=1)
        htmp=Haug@np.linalg.solve(Caug,zaug)
        Hhat[n]=htmp
        invsig.append(np.linalg.inv(Sigma))
    invsig=np.asarray(invsig)
    pred=np.empty(len(Y),int)
    Cand=Fscaled[:,DATA_BINS]
    for s in range(0,len(Y),200):
        e=min(len(Y),s+200); resid=R[s:e,None,DATA_BINS]-Hhat[s:e,None,DATA_BINS]*Cand[None]
        variance=np.asarray([np.trace((N0/2)*IA[n]@IA[n].T) for n in range(s,e)])
        dist=np.sum(np.abs(resid)**2,axis=2)/variance[:,None]
        pred[s:e]=np.argmin(dist,axis=1)
    return pred

def run(kappa,blocks,seed):
    rng=np.random.default_rng(seed); delays,powers,D=tdla_operators(L); Fp,_,_,_,_=make_ofdm_codebooks()
    idx=rng.integers(M,size=blocks); h=random_tdla_coefficients(rng,blocks,powers); om=rng.uniform(-math.radians(8),math.radians(8),blocks)
    A0=A_norm(kappa); A=np.repeat(A0[None],blocks,axis=0); ang=rng.uniform(0,2*np.pi,blocks); d=np.stack([np.cos(ang),np.sin(ang)],1); N0=10**(-10/10)
    noise=np.sqrt(N0/2)*rng.normal(size=(blocks,2,L)); Y=ofdm_received(Fp[idx],h,delays,om,A,d,noise)
    old=np.mean(decode_practical(Y,A,d,N0,Fp,powers,delays)!=idx)
    new=np.mean(decode_covaware(Y,A,d,N0,Fp,powers,delays)!=idx)
    return old,new

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--blocks',type=int,default=1000); ap.add_argument('--seeds',default='13,59,151'); ap.add_argument('--output',default='Data/operating_iq_sweep_covaware.csv'); a=ap.parse_args(); rows=[]
    for k in [1,1.25,1.5,2,3]:
        for seed in [int(x) for x in a.seeds.split(',')]:
            old,new=run(k,a.blocks,20261003+seed+int(100*k)); rows += [{'condition_number':k,'seed':seed,'receiver':'legacy scalar metric','PER':old,'blocks':a.blocks},{'condition_number':k,'seed':seed,'receiver':'covariance-aware metric','PER':new,'blocks':a.blocks}]
    df=pd.DataFrame(rows); out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print(df.groupby(['condition_number','receiver']).PER.agg(['mean','std']).to_string()); print('saved',out)
if __name__=='__main__': main()
