"""Reference implementation of the offline incidence objective.

The shipped optimized codebook is in Artifact/fractional_incidence_codebook.npz.
This script documents the training mechanism used in the final stage. The default
iteration count is intentionally modest; increase --steps for a full rerun.
"""
from pathlib import Path
import argparse, math
import numpy as np
import torch
from fractional_tdla_core import tdla_operators, centering_matrix


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=180)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--rank',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260918)
    ap.add_argument('--hard-negative-k',type=int,default=0,
                    help='If positive, backpropagate only through the top-k wrong-message incidences.')
    args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    root=Path(__file__).resolve().parents[1]
    init=np.load(root/'Artifact/stage3_codebooks.npz')['X_dmm'].astype(np.float32)
    M,_,L=init.shape
    _,powers,D=tdla_operators(L)
    D_t=torch.tensor(D,dtype=torch.float32)
    pow_t=torch.tensor(np.sqrt(powers),dtype=torch.float32)

    H=centering_matrix(L)
    ew,EV=np.linalg.eigh(H)
    B=EV[:,ew>0.5].astype(np.float32)
    Bt=torch.tensor(B)
    Z0=np.empty((M,L-1,2),np.float32)
    for m in range(M): Z0[m]=B.T@init[m].T
    Z=torch.tensor(Z0,requires_grad=True)
    opt=torch.optim.Adam([Z],lr=.006)
    tidx=torch.arange(L,dtype=torch.float32)

    def codewords(Zs):
        Q,_=torch.linalg.qr(Zs)
        return math.sqrt(L/2)*(Q.transpose(1,2)@Bt.T)

    def delayed(X):
        return torch.einsum('plk,bck->bpcl',D_t,X)

    def rotate(U,omega):
        th=omega[:,None]*tidx[None,:]
        c,s=torch.cos(th),torch.sin(th)
        I,Q=U[:,:,0],U[:,:,1]
        return torch.stack([c[:,None,:]*I-s[:,None,:]*Q,
                            s[:,None,:]*I+c[:,None,:]*Q],dim=2)

    for it in range(args.steps):
        opt.zero_grad()
        ids=torch.randperm(M)[:args.batch]
        X=codewords(Z[ids]); U=delayed(X)
        a=torch.randn(args.batch,len(powers))*pow_t[None,:]/math.sqrt(2)
        b=torch.randn(args.batch,len(powers))*pow_t[None,:]/math.sqrt(2)
        I,Q=U[:,:,0],U[:,:,1]
        Y=torch.stack([(a[:,:,None]*I-b[:,:,None]*Q).sum(1),
                       (b[:,:,None]*I+a[:,:,None]*Q).sum(1)],dim=1)
        omega=(2*torch.rand(args.batch)-1)*math.radians(4)
        # rotate received plane
        th=omega[:,None]*tidx[None,:]
        c,s=torch.cos(th),torch.sin(th)
        Y=torch.stack([c*Y[:,0]-s*Y[:,1],s*Y[:,0]+c*Y[:,1]],dim=1)
        A=torch.randn(args.batch,2,2)+2*torch.eye(2)[None,:,:]
        Y=A@Y; Y=Y-Y.mean(-1,keepdim=True)
        Y=Y/torch.sqrt((Y*Y).sum((1,2),keepdim=True).clamp_min(1e-8))

        omega_w=(2*torch.rand(args.batch)-1)*math.radians(4)
        Uw=rotate(U,omega_w)
        F=torch.cat([Uw[:,:,0],Uw[:,:,1]],dim=1)
        F=F*torch.cat([pow_t,pow_t])[None,:,None]
        F=F-F.mean(-1,keepdim=True)
        _,_,Vh=torch.linalg.svd(F,full_matrices=False)
        Qr=Vh[:,:args.rank,:].transpose(1,2)
        Pj=Qr@Qr.transpose(1,2)
        Cy=Y.transpose(1,2)@Y
        score=Cy.reshape(args.batch,-1)@Pj.reshape(args.batch,-1).T
        off=score[~torch.eye(args.batch,dtype=torch.bool)].clamp(0,1)
        if args.hard_negative_k > 0:
            k=min(args.hard_negative_k, off.numel())
            off=torch.topk(off.reshape(-1), k=k).values
        loss=(off**8).mean()+0.12*(off**3).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_([Z],5.0); opt.step()
        if it%30==0: print(it,float(loss.detach()))

    with torch.no_grad(): X=codewords(Z).cpu().numpy()
    np.savez_compressed(root/'Artifact/retrained_from_reference.npz',X_opt=X)

if __name__=='__main__': main()
