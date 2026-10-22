from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'Figure'; MF=ROOT/'Manuscript'/'figures'
F.mkdir(exist_ok=True); MF.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7.5,
                     'xtick.labelsize':8,'ytick.labelsize':8,'figure.dpi':180})

# Main rerun: three independent seeds, 1200 blocks/seed/SNR.
df=pd.read_csv(ROOT/'Data/final_rerun_fractional_tdla_headtohead.csv')
fig,ax=plt.subplots(figsize=(5.9,3.6))
colors={'Selective SQM':'#1f77b4','CP-OFDM TDL-LMMSE':'#d95f02','CP-OFDM full oracle ML':'#2ca25f'}
labels={'Selective SQM':'SQM, rank 4','CP-OFDM TDL-LMMSE':'Practical CP-OFDM','CP-OFDM full oracle ML':'Full-oracle CP-OFDM'}
for method,g in df.groupby('method'):
    g=g.sort_values('snr_db')
    ax.errorbar(g.snr_db,g.PER_mean,yerr=g.PER_95CI_halfwidth,marker='o',lw=1.4,ms=4,
                capsize=2,color=colors[method],label=labels[method])
ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Packet error rate')
ax.set_title('Exact fractional-delay TDL-A: independent rerun')
ax.grid(True,which='both',alpha=.22); ax.legend(frameon=True,loc='lower left')
fig.tight_layout(); fig.savefig(F/'fig_main_rerun.png',dpi=300); fig.savefig(MF/'fig_main_rerun.png',dpi=300); plt.close(fig)

# Adaptive extension summary and regime response.
s=pd.read_csv(ROOT/'Data/online_adaptive_summary.csv')
order=['S0_fixed_r4','S1_best_fixed','S2_SNR_class_lookup','S3_window_confidence',
       'S4_window_confidence_residual','S5_EWMA_confidence_residual','S6_confidence_triggered_refinement',
       'S7_ACK_UCB_optional','S8_oracle_rank']
short=['S0 fixed $r=4$','S1 best fixed','S2 SNR/class','S3 window gap','S4 window gap+resid','S5 EWMA','S6 refine','S7 ACK-UCB','S8 oracle']
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),gridspec_kw={'width_ratios':[1.15,1]})
ss=s.set_index('method').loc[order].reset_index()
x=np.arange(len(ss))
axs[0].errorbar(x,ss.PER,yerr=ss.PER_95CI_halfwidth,fmt='o',color='#2774ae',ecolor='#2774ae',capsize=2)
axs[0].set_xticks(x); axs[0].set_xticklabels(short,rotation=65,ha='right'); axs[0].set_ylabel('PER on time-varying trace'); axs[0].set_title('Scheme comparison')
axs[0].grid(axis='y',alpha=.22); axs[0].set_ylim(0.14,0.235)
by=pd.read_csv(ROOT/'Data/online_adaptive_summary_by_segment.csv')
keep=['S0_fixed_r4','S2_SNR_class_lookup','S4_window_confidence_residual','S5_EWMA_confidence_residual','S6_confidence_triggered_refinement','S8_oracle_rank']
means=by[by.method.isin(keep)].groupby(['method','segment']).mean(numeric_only=True).reset_index()
for method in keep:
    g=means[means.method==method].sort_values('segment')
    axs[1].plot(g.segment,g.mean_selected_rank,marker='o',lw=1.2,label=short[order.index(method)])
axs[1].set_xticks([0,1,2,3]); axs[1].set_xticklabels(['5/30','10/60','15/100','7/30'])
axs[1].set_xlabel('Regime: SNR dB / delay ns'); axs[1].set_ylabel('Mean selected rank'); axs[1].set_title('Rank response across regimes')
axs[1].set_ylim(3.5,6.3); axs[1].grid(alpha=.22); axs[1].legend(fontsize=6.5,frameon=True,loc='upper left',bbox_to_anchor=(0.0,0.98))
fig.tight_layout(); fig.savefig(F/'fig_online_adaptive.png',dpi=300); fig.savefig(MF/'fig_online_adaptive.png',dpi=300); plt.close(fig)
print('wrote',F/'fig_main_rerun.png',F/'fig_online_adaptive.png')
