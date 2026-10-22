from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'Figure'; MF=ROOT/'Manuscript'/'figures'; F.mkdir(exist_ok=True); MF.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7})
packet=pd.read_csv(ROOT/'Data/online_adaptive_packet_results.csv')
# Packet-by-packet selected rank, representative final seed.
d=packet[(packet.sequence=='canonical')&(packet.seed==13)]
methods=['S0_fixed_r4','S2_SNR_class_lookup','S4_window_confidence_residual','S5_EWMA_confidence_residual','S8_regime_oracle_rank']
labels={'S0_fixed_r4':'fixed $r=4$','S2_SNR_class_lookup':'SNR lookup','S4_window_confidence_residual':'window gap+residual','S5_EWMA_confidence_residual':'EWMA gap+residual','S8_regime_oracle_rank':'state oracle'}
colors=['#777777','#1f77b4','#d95f02','#2ca02c','#9467bd']
fig,ax=plt.subplots(figsize=(7.0,3.1))
for m,c in zip(methods,colors):
    q=d[d.method==m].sort_values('packet')
    ax.step(q.packet,q.selected_rank,where='post',lw=1.2,color=c,label=labels[m],alpha=.95)
for x in [300,600,900]: ax.axvline(x,color='k',ls='--',lw=.8,alpha=.65)
for x,txt in [(0,'5 dB / 30 ns'),(300,'10 dB / 60 ns'),(600,'15 dB / 100 ns'),(900,'7 dB / 30 ns')]:
    ax.text(x+8,6.08,txt,fontsize=7,va='top')
ax.set_xlim(0,1200); ax.set_ylim(2.8,6.25); ax.set_xlabel('Packet index'); ax.set_ylabel('Selected rank'); ax.set_yticks([3,4,5,6]); ax.grid(alpha=.22,axis='y'); ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,.02),frameon=True,framealpha=.85,facecolor='white',edgecolor='none'); fig.tight_layout(); fig.savefig(F/'fig_online_rank_trace.png',dpi=300,bbox_inches='tight'); fig.savefig(MF/'fig_online_rank_trace.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Transient PER after transitions vs steady-state.
t=pd.read_csv(ROOT/'Data/online_transient_seed_metrics.csv')
methods2=['S2_SNR_class_lookup','S4_window_confidence_residual','S5_EWMA_confidence_residual','S8_regime_oracle_rank']
labels2={'S2_SNR_class_lookup':'SNR lookup','S4_window_confidence_residual':'window gap+residual','S5_EWMA_confidence_residual':'EWMA gap+residual','S8_regime_oracle_rank':'state oracle'}
order=['first_25','first_50','first_100','steady_after_100']; xpos=np.arange(4)
fig,ax=plt.subplots(figsize=(6.4,3.0))
for m,c in zip(methods2,['#1f77b4','#d95f02','#2ca02c','#9467bd']):
    q=t[t.method==m].groupby('window').PER.agg(['mean','std','count']).reindex(order)
    err=4.30265273*q['std']/np.sqrt(q['count'])
    ax.errorbar(xpos,q['mean'],yerr=err,marker='o',lw=1.1,capsize=2.5,label=labels2[m],color=c)
ax.set_xticks(xpos,['first 25','first 50','first 100','packets 100+']); ax.set_ylabel('PER'); ax.set_xlabel('Packets after each regime transition'); ax.set_ylim(0,0.45); ax.grid(alpha=.22); ax.legend(ncol=2,loc='upper left',frameon=False); fig.tight_layout(); fig.savefig(F/'fig_online_transient.png',dpi=300); fig.savefig(MF/'fig_online_transient.png',dpi=300); plt.close(fig)
# W/alpha sensitivity.
s=pd.read_csv(ROOT/'Data/online_controller_sensitivity.csv')
fig,axs=plt.subplots(1,2,figsize=(6.6,2.65))
q=s[s.policy=='S4_window'].sort_values('W'); axs[0].errorbar(q.W,q.PER_mean,yerr=q.PER_ci95,marker='o',capsize=2.5,color='#d95f02'); axs[0].set_xlabel('Window length $W$'); axs[0].set_ylabel('PER'); axs[0].set_xticks(q.W); axs[0].grid(alpha=.22); axs[0].set_title('Sliding window')
q=s[s.policy=='S5_EWMA'].sort_values('alpha'); axs[1].errorbar(q.alpha,q.PER_mean,yerr=q.PER_ci95,marker='o',capsize=2.5,color='#2ca02c'); axs[1].set_xlabel('EWMA factor $\\alpha$'); axs[1].set_ylabel('PER'); axs[1].grid(alpha=.22); axs[1].set_title('EWMA')
fig.tight_layout(); fig.savefig(F/'fig_online_sensitivity.png',dpi=300); fig.savefig(MF/'fig_online_sensitivity.png',dpi=300); plt.close(fig)
