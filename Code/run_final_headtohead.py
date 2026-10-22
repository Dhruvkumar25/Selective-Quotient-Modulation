"""Re-run the final exact fractional-delay TDL-A head-to-head.

The default command is a small smoke run. For the manuscript-scale run use, e.g.:
  python Code/run_final_headtohead.py --blocks 1400 --seeds 13,59,151 \
      --snrs 5,7,10,12,15 --output Data/rerun_fractional_tdla_headtohead.csv

The script implements the same three main curves used in the final comparison:
Selective SQM, practical TDL-aware CP-OFDM, and full-oracle CP-OFDM. The DMM
codebook is loaded from the shipped artifact; the OFDM short linear code is
reconstructed deterministically from the fixed search seed.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd
from fractional_tdla_core import tdla_operators, weighted_nuisance_dictionary, exact_fractional_channel, random_tdla_coefficients

ROOT=Path(__file__).resolve().parents[1]
M=256; L=16; NFFT=12; CP=4
PILOT_BINS=np.array([0,3,6,9]); DATA_BINS=np.array([1,2,4,5,7,8,10,11])
T_CRIT_DF2=4.302652729911275


def rot(th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([np.stack([c,-s],-1),np.stack([s,c],-1)],-2)


def randA(rng,N,kappa=1.5):
    a=rng.uniform(0,2*np.pi,N); b=rng.uniform(0,2*np.pi,N)
    s2=np.sqrt(2/(1+kappa*kappa)); s1=kappa*s2
    A=np.empty((N,2,2))
    for n in range(N): A[n]=rot(a[n])@np.diag([s1,s2])@rot(b[n])
    return A


def invA(A):
    det=A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]
    R=np.empty_like(A)
    R[:,0,0]= A[:,1,1]/det; R[:,1,1]= A[:,0,0]/det
    R[:,0,1]=-A[:,0,1]/det; R[:,1,0]=-A[:,1,0]/det
    return R


def messages8():
    return np.array([[(m>>k)&1 for k in range(8)] for m in range(M)],dtype=np.int8)


def make_ofdm_codebooks():
    messages=messages8()
    Fp=np.zeros((M,NFFT),complex); Fp[:,PILOT_BINS]=1
    Fp[:,DATA_BINS]=(1-2*messages)
    rs=np.random.default_rng(20260919)
    bestd=-1; bestP=None
    for _ in range(20000):
        Pm=rs.integers(0,2,(8,4),dtype=np.int8)
        G=np.concatenate([np.eye(8,dtype=np.int8),Pm],1)
        C=(messages@G)%2; d=int(C[1:].sum(1).min())
        if d>bestd: bestd=d; bestP=Pm.copy()
        if d>=3: break
    G=np.concatenate([np.eye(8,dtype=np.int8),bestP],1)
    coded=(messages@G)%2
    Fo=(1-2*coded).astype(float).astype(complex)
    def mod(F):
        x=np.fft.ifft(F,axis=-1,norm='ortho')
        return np.concatenate([x[...,-CP:],x],axis=-1)
    def normalize(X):
        e=np.mean(np.sum(np.abs(X)**2,axis=-1)); scale=np.sqrt(L/e)
        return X*scale,scale
    Xp,sp=normalize(mod(Fp)); Xo,so=normalize(mod(Fo))
    return Fp*sp,Fo*so,Xp,Xo,bestd


def precompute_dmm_bases(Xd,powers,D,rank=4,cfo_max=4):
    deg=np.linspace(-cfo_max-1, cfo_max+1, int(round(2*cfo_max+2))+1); grid=np.deg2rad(deg)
    B=np.empty((M,len(grid),L,rank),np.float32)
    for m in range(M):
        for j,w in enumerate(grid):
            F=weighted_nuisance_dictionary(Xd[m],w,powers,D)
            _,_,Vh=np.linalg.svd(F,full_matrices=False)
            B[m,j]=Vh[:rank].T.astype(np.float32)
    return grid,B


def decode_dmm(Y,bases):
    Y=Y-Y.mean(axis=2,keepdims=True); pred=np.empty(len(Y),int)
    for b in range(len(Y)):
        Z=np.einsum('al,mklr->mkar',Y[b],bases,optimize=True)
        score=np.sum(Z*Z,axis=(2,3)); pred[b]=np.unravel_index(np.argmax(score),score.shape)[0]
    return pred


def freq_response(h,delays):
    k=np.arange(NFFT)
    E=np.exp(-1j*2*np.pi*k[:,None]*delays[None,:]/NFFT)
    return h@E.T,E


def apply_iq_dc_noise(s,omega,A,d,noise):
    s=s*np.exp(1j*omega[:,None]*np.arange(L)); sr=np.stack([s.real,s.imag],axis=1)
    return np.einsum('nij,njk->nik',A,sr)+d[:,:,None]+noise


def ofdm_received(Fsel,h,delays,omega,A,d,noise):
    Hf,_=freq_response(h,delays); Yf=Hf*Fsel
    yu=np.fft.ifft(Yf,axis=1,norm='ortho'); ycp=np.concatenate([yu[:,-CP:],yu],axis=1)
    return apply_iq_dc_noise(ycp,omega,A,d,noise)


def decode_practical(Y,A,d,N0,Fp_scaled,powers,delays):
    IA=invA(A); zr=np.einsum('nij,njk->nik',IA,Y-d[:,:,None]); z=zr[:,0]+1j*zr[:,1]
    corr=np.sum(np.conj(z[:,:CP])*z[:,NFFT:NFFT+CP],axis=1); omhat=np.angle(corr)/NFFT
    z=z*np.exp(-1j*omhat[:,None]*np.arange(L)); R=np.fft.fft(z[:,CP:],axis=1,norm='ortho')
    k=np.arange(NFFT); E=np.exp(-1j*2*np.pi*k[:,None]*delays[None,:]/NFFT)
    R_H=(E*powers[None,:])@E.conj().T; R_ap=R_H[:,PILOT_BINS]; R_pp=R_H[np.ix_(PILOT_BINS,PILOT_BINS)]
    Hobs=R[:,PILOT_BINS]/Fp_scaled[0,PILOT_BINS][None,:]
    noise_var=(N0/2)*np.einsum('nij,nij->n',IA,IA)
    Hhat=np.empty((len(Y),NFFT),complex)
    for n in range(len(Y)):
        Cpp=R_pp+(noise_var[n]/(np.abs(Fp_scaled[0,PILOT_BINS])**2).mean())*np.eye(len(PILOT_BINS))
        Hhat[n]=R_ap@np.linalg.solve(Cpp,Hobs[n])
    Cand=Fp_scaled[:,DATA_BINS]; rd=R[:,DATA_BINS]; pred=np.empty(len(Y),int)
    for s in range(0,len(Y),250):
        expct=Hhat[s:s+250,None,DATA_BINS]*Cand[None,:,:]
        dist=np.sum(np.abs(rd[s:s+250,None,:]-expct)**2,axis=2)
        pred[s:s+len(expct)]=np.argmin(dist,axis=1)
    return pred


def decode_oracle(Y,h,delays,omega,A,d,Fo_scaled):
    pred=np.empty(len(Y),int); Hf,_=freq_response(h,delays)
    for s in range(0,len(Y),50):
        e=min(len(Y),s+50); expected=np.empty((e-s,M,2,L),float)
        for b in range(e-s):
            Yf=Hf[s+b][None,:]*Fo_scaled; yu=np.fft.ifft(Yf,axis=1,norm='ortho')
            ycp=np.concatenate([yu[:,-CP:],yu],axis=1)*np.exp(1j*omega[s+b]*np.arange(L))[None,:]
            rr=np.stack([ycp.real,ycp.imag],axis=1)
            expected[b]=np.einsum('ij,mjl->mil',A[s+b],rr)+d[s+b][None,:,None]
        dist=np.sum((Y[s:e,None]-expected)**2,axis=(2,3)); pred[s:e]=np.argmin(dist,axis=1)
    return pred


def run_point(snr,cfo_max,blocks,seed,Xd,bases,Fp,Fo,powers,delays,D):
    rng=np.random.default_rng(seed); idx=rng.integers(M,size=blocks)
    h=random_tdla_coefficients(rng,blocks,powers); omega=rng.uniform(-math.radians(cfo_max),math.radians(cfo_max),blocks)
    A=randA(rng,blocks,1.5); ang=rng.uniform(0,2*np.pi,blocks); d=np.stack([np.cos(ang),np.sin(ang)],1)
    N0=10**(-snr/10); noise=np.sqrt(N0/2)*rng.normal(size=(blocks,2,L))
    sd=exact_fractional_channel(Xd[idx],h,D); Yd=apply_iq_dc_noise(sd,omega,A,d,noise)
    Yp=ofdm_received(Fp[idx],h,delays,omega,A,d,noise); Yo=ofdm_received(Fo[idx],h,delays,omega,A,d,noise)
    return np.mean(decode_dmm(Yd,bases)!=idx),np.mean(decode_practical(Yp,A,d,N0,Fp,powers,delays)!=idx),np.mean(decode_oracle(Yo,h,delays,omega,A,d,Fo)!=idx)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--blocks',type=int,default=100)
    ap.add_argument('--seeds',default='13'); ap.add_argument('--snrs',default='10'); ap.add_argument('--cfo',type=float,default=4.0)
    ap.add_argument('--rank',type=int,default=4); ap.add_argument('--output',default='')
    ap.add_argument('--per-seed-output',default=''); ap.add_argument('--paired-output',default='')
    args=ap.parse_args(); seeds=[int(x) for x in args.seeds.split(',')]; snrs=[float(x) for x in args.snrs.split(',')]
    X=np.load(ROOT/'Artifact/fractional_incidence_codebook.npz')['X_opt']; Xd=X[:,0]+1j*X[:,1]
    delays,powers,D=tdla_operators(L); Fp,Fo,_,_,dmin=make_ofdm_codebooks(); _,bases=precompute_dmm_bases(Xd,powers,D,args.rank,args.cfo)
    rows=[]; per_seed=[]
    for snr in snrs:
        names=['Selective SQM','CP-OFDM TDL-LMMSE','CP-OFDM full oracle ML']
        for sd in seeds:
            vals=run_point(snr,args.cfo,args.blocks,20260918+int(1000*snr)+sd,Xd,bases,Fp,Fo,powers,delays,D)
            for name, value in zip(names, vals):
                per_seed.append({'snr_db':snr,'seed':sd,'method':name,'PER':float(value),'blocks':args.blocks})
        g=pd.DataFrame(per_seed)
        for name in names:
            x=g[(g.snr_db==snr)&(g.method==name)].PER.to_numpy()
            ci=0.0 if len(x)<2 else T_CRIT_DF2*x.std(ddof=1)/np.sqrt(len(x))
            rows.append({'snr_db':snr,'method':name,'PER_mean':x.mean(),'PER_95CI_halfwidth':ci,'blocks':args.blocks*len(seeds)})
    df=pd.DataFrame(rows); print('linear code dmin:',dmin); print(df.to_string(index=False))
    if args.output:
        path=ROOT/args.output; path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); print('saved',path)
    ps=pd.DataFrame(per_seed)
    if args.per_seed_output:
        path=ROOT/args.per_seed_output; path.parent.mkdir(parents=True,exist_ok=True); ps.to_csv(path,index=False); print('saved',path)
    if args.paired_output:
        rows_p=[]
        for snr,g in ps.groupby('snr_db'):
            p=g.pivot(index='seed',columns='method',values='PER')
            for name in names[1:]:
                x=(p['Selective SQM']-p[name]).to_numpy()
                ci=0.0 if len(x)<2 else T_CRIT_DF2*x.std(ddof=1)/np.sqrt(len(x))
                rows_p.append({'snr_db':snr,'comparison':'SQM - '+name,
                               'paired_delta_mean':x.mean(),
                               'paired_95CI_halfwidth':ci,'seeds':len(x)})
        path=ROOT/args.paired_output; path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows_p).to_csv(path,index=False); print('saved',path)

if __name__=='__main__': main()
