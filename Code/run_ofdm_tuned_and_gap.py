"""Stronger CP-OFDM controls and oracle-gap decomposition.

The tuned baseline keeps 8 information bits and 16 total samples while varying
pilot/data modulation and CP allocation.  The gap screen additionally evaluates
full-tone coded OFDM with selected nuisance coordinates revealed, including an
unknown-static-I/Q/DC oracle subcase.  Results are intentionally labeled by the
side information each receiver receives.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractional_tdla_core import tdla_operators, random_tdla_coefficients
from run_final_headtohead import randA, apply_iq_dc_noise
from experiment_config import M, L, physical_cfo_sample
ROOT=Path(__file__).resolve().parents[1]


def messages8(): return np.array([[(m>>k)&1 for k in range(8)] for m in range(M)], dtype=np.int8)


def make_family(nfft, cp, pilot_bins, data_bins, modulation='bpsk'):
    bits=messages8(); F=np.zeros((M,nfft),complex); F[:,pilot_bins]=1
    if modulation=='bpsk':
        assert len(data_bins)==8; F[:,data_bins]=(1-2*bits)
    elif modulation=='qpsk':
        assert len(data_bins)==4
        for q,k in enumerate(data_bins):
            b0,b1=bits[:,2*q],bits[:,2*q+1]
            F[:,k]=((1-2*b0)+1j*(1-2*b1))/np.sqrt(2)
    else: raise ValueError(modulation)
    x=np.fft.ifft(F,axis=1,norm='ortho'); x=np.concatenate([x[:,-cp:],x],axis=1)
    e=np.mean(np.sum(np.abs(x)**2,axis=1)); scale=np.sqrt(L/e)
    return F*scale, x*scale


def channel_freq(h, delays, nfft):
    k=np.arange(nfft); E=np.exp(-1j*2*np.pi*k[:,None]*delays[None,:]/nfft); return h@E.T


def apply_real(y, omega, A, d, noise):
    y=y*np.exp(1j*omega[:,None]*np.arange(L)); sr=np.stack([y.real,y.imag],axis=1)
    return np.einsum('nij,njk->nik',A,sr)+d[:,:,None]+noise


def ofdm_received(Fsel,h,delays,omega,A,d,noise,nfft,cp):
    H=channel_freq(h,delays,nfft); yf=H*Fsel; yu=np.fft.ifft(yf,axis=1,norm='ortho'); y=np.concatenate([yu[:,-cp:],yu],axis=1)
    return apply_real(y,omega,A,d,noise)


def invA(A):
    det=A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]; R=np.empty_like(A)
    R[:,0,0]=A[:,1,1]/det; R[:,1,1]=A[:,0,0]/det; R[:,0,1]=-A[:,0,1]/det; R[:,1,0]=-A[:,1,0]/det; return R


def decode_pilot(Y,A,d,N0,Fscaled,powers,delays,nfft,cp,pilot_bins,data_bins,modulation,cfo_mode='estimate'):
    IA=invA(A); z=np.einsum('nij,njk->nik',IA,Y-d[:,:,None]); z=z[:,0]+1j*z[:,1]
    if cfo_mode=='estimate':
        corr=np.sum(np.conj(z[:,:cp])*z[:,nfft:nfft+cp],axis=1); omhat=np.angle(corr)/nfft
    else: omhat=np.zeros(len(Y))
    z=z*np.exp(-1j*omhat[:,None]*np.arange(L)); R=np.fft.fft(z[:,cp:],axis=1,norm='ortho')
    E=np.exp(-1j*2*np.pi*np.arange(nfft)[:,None]*delays[None,:]/nfft); RH=(E*powers[None,:])@E.conj().T
    Hobs=R[:,pilot_bins]/Fscaled[0,pilot_bins][None,:]
    noise_var=(N0/2)*np.einsum('nij,nij->n',IA,IA)
    Hhat=np.empty((len(Y),nfft),complex)
    for n in range(len(Y)):
        pilot_amp2=np.mean(np.abs(Fscaled[0,pilot_bins])**2)
        Hhat[n]=RH[:,pilot_bins]@np.linalg.solve(RH[np.ix_(pilot_bins,pilot_bins)]+(noise_var[n]/pilot_amp2)*np.eye(len(pilot_bins)),Hobs[n])
    cand=Fscaled[:,data_bins]; rd=R[:,data_bins]; pred=np.empty(len(Y),int)
    for s in range(0,len(Y),100):
        e=min(len(Y),s+100); expct=Hhat[s:e,None,data_bins]*cand[None,:,:]
        dist=np.sum(np.abs(rd[s:e,None,:]-expct)**2,axis=2); pred[s:e]=np.argmin(dist,axis=1)
    return pred


def make_full_code(nfft,cp):
    bits=messages8(); rs=np.random.default_rng(20260919); bestd=-1; bestP=None
    for _ in range(20000):
        P=rs.integers(0,2,(8,4),dtype=np.int8); G=np.concatenate([np.eye(8,dtype=np.int8),P],1); C=(bits@G)%2; d=int(C[1:].sum(1).min())
        if d>bestd: bestd=d; bestP=P.copy()
        if d>=3: break
    G=np.concatenate([np.eye(8,dtype=np.int8),bestP],1); coded=(bits@G)%2; F=(1-2*coded).astype(complex)
    x=np.fft.ifft(F,axis=1,norm='ortho'); x=np.concatenate([x[:,-cp:],x],axis=1); scale=np.sqrt(L/np.mean(np.sum(np.abs(x)**2,axis=1))); return F*scale,x*scale


def decode_full_oracle(Y,h,delays,omega,A,d,Fscaled,nfft,cp,know_omega=True,know_A=True):
    H=channel_freq(h,delays,nfft); pred=np.empty(len(Y),int)
    for b in range(len(Y)):
        scores=[]
        w=omega[b] if know_omega else 0.0
        for m in range(M):
            yf=H[b]*Fscaled[m]; yu=np.fft.ifft(yf,norm='ortho'); x=np.concatenate([yu[-cp:],yu])*np.exp(1j*w*np.arange(L))
            R=np.stack([x.real,x.imag])
            if know_A:
                exp=np.einsum('ij,jl->il',A[b],R)+d[b,:,None]; scores.append(np.sum((Y[b]-exp)**2))
            else:
                Z=np.vstack([R,np.ones(L)]); C=Y[b]@np.linalg.pinv(Z); exp=C@Z; scores.append(np.sum((Y[b]-exp)**2))
        pred[b]=int(np.argmin(scores))
    return pred


def run(snr,blocks,seed):
    rng=np.random.default_rng(seed); delays,powers,D=tdla_operators(L); idx=rng.integers(M,size=blocks); h=random_tdla_coefficients(rng,blocks,powers); om=physical_cfo_sample(rng,blocks); A=randA(rng,blocks,1.5); ang=rng.uniform(0,2*np.pi,blocks); d=np.stack([np.cos(ang),np.sin(ang)],1); noise=np.sqrt(10**(-snr/10)/2)*rng.normal(size=(blocks,2,L))
    rows=[]
    # Tuned pilot/CP families.
    families=[('CP4-4pilot-8BPSK',12,4,[0,3,6,9],[1,2,4,5,7,8,10,11],'bpsk'),('CP4-8pilot-4QPSK',12,4,list(range(0,8)),[8,9,10,11],'qpsk'),('CP2-4pilot-8BPSK',14,2,[0,3,6,9],[1,2,4,5,7,8,10,11],'bpsk')]
    for name,nfft,cp,pilots,data,mod in families:
        F,x=make_family(nfft,cp,pilots,data,mod); Y=ofdm_received(F[idx],h,delays,om,A,d,noise,nfft,cp)
        for cfo_mode in ['estimate','oracle']:
            pred=decode_pilot(Y,A,d,10**(-snr/10),F,powers,delays,nfft,cp,np.array(pilots),np.array(data),mod,cfo_mode); rows.append({'experiment':'tuned_ofdm','method':name+' '+cfo_mode+' CFO','PER':float(np.mean(pred!=idx)),'snr_db':snr,'blocks':blocks})
    # Oracle decomposition with full coded OFDM.
    F,x=make_full_code(12,4); Y=ofdm_received(F[idx],h,delays,om,A,d,noise,12,4)
    for label,know_omega,know_A in [('full oracle',True,True),('oracle CFO + unknown I/Q/DC',True,False),('oracle I/Q/DC + no CFO oracle',False,True)]:
        pred=decode_full_oracle(Y,h,delays,om,A,d,F,12,4,know_omega,know_A); rows.append({'experiment':'oracle_gap','method':label,'PER':float(np.mean(pred!=idx)),'snr_db':snr,'blocks':blocks})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snr',type=float,default=10); ap.add_argument('--blocks',type=int,default=200); ap.add_argument('--seeds',default='13,59'); ap.add_argument('--output',default='Data/ofdm_tuned_and_gap.csv'); a=ap.parse_args()
    rows=[]
    for sd in [int(x) for x in a.seeds.split(',')]: rows += run(a.snr,a.blocks,20260929+sd)
    df=pd.DataFrame(rows); print(df.groupby(['experiment','method']).PER.mean().to_string()); out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print('saved',out)
if __name__=='__main__': main()
