from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'Manuscript'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':180})
# delay/PDP combined
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for i,(f,xcol,xlabel,title) in enumerate([('delay_spread_sweep.csv','delay_spread_ns','True delay scale (ns)','Delay-spread sweep'),('pdp_mismatch_sweep.csv','pdp_power_exponent','PDP power exponent','PDP mismatch')]):
 df=pd.read_csv(ROOT/'Data'/f)
 for method,grp in df.groupby('method'):
  g=grp.groupby(xcol).PER.agg(['mean','std']).reset_index(); ax[i].plot(g[xcol],g['mean'],marker='o',label='SQM' if method.startswith('Incidence-trained') else method)
 ax[i].set_xlabel(xlabel); ax[i].set_ylabel('PER at 10 dB'); ax[i].set_title(title); ax[i].grid(alpha=.25)
 ax[i].set_ylim(bottom=0)
 ax[i].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'fig_delay_pdp_revision.png',bbox_inches='tight'); plt.close(fig)
# theorem 3
fig,ax=plt.subplots(figsize=(4.0,2.8)); df=pd.read_csv(ROOT/'Data'/'theorem3_validation.csv')
for r,g in df.groupby('rank'):
 h=g.groupby('noise_fraction_of_certified_radius').empirical_error_rate.mean().reset_index(); ax.plot(h.iloc[:,0],h.iloc[:,1],marker='o',label=f'r={r}')
ax.set_xlabel('Noise norm / certified radius'); ax.set_ylabel('Empirical error rate'); ax.set_title('Theorem 3 validation'); ax.set_ylim(-.002,.05); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'fig_theorem3_revision.png',bbox_inches='tight'); plt.close(fig)
# scaling
fig,ax=plt.subplots(1,2,figsize=(6.7,2.7)); df=pd.read_csv(ROOT/'Data'/'scaling_10dB.csv')
ax[0].bar(df.bits.astype(str),df.PER,color='#1f77b4'); ax[0].set_xlabel('Payload bits'); ax[0].set_ylabel('PER at 10 dB'); ax[0].set_title('Preliminary scaling stress test'); ax[0].grid(axis='y',alpha=.25)
ax[1].bar(df.bits.astype(str),df.score_evaluations,color='#ff7f0e'); ax[1].set_xlabel('Payload bits'); ax[1].set_ylabel('Full score evaluations'); ax[1].set_title('Decoder search growth'); ax[1].grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'fig_scaling_revision.png',bbox_inches='tight'); plt.close(fig)
# ofdm baseline
fig,ax=plt.subplots(figsize=(6.5,2.8)); df=pd.read_csv(ROOT/'Data'/'ofdm_tuned_and_gap.csv'); g=df.groupby('method').PER.mean().sort_values(); ax.barh(g.index,g.values,color='#2ca02c'); ax.set_xlabel('PER at 10 dB'); ax.set_title('Strengthened OFDM and oracle-gap controls'); ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'fig_ofdm_revision.png',bbox_inches='tight'); plt.close(fig)
