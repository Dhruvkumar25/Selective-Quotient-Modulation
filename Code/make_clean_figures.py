from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; F=ROOT/'Figure'; MF=ROOT/'Manuscript'/'figures'; F.mkdir(exist_ok=True); MF.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':10,'axes.labelsize':8.5,'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8})
# Fig. 3: new coded controls at 10 dB, with t-based seed intervals.
df=pd.read_csv(ROOT/'Data/reviewer_baselines_summary.csv'); g=df[df.snr_db==10].copy()
labels={'SQM_rank4':'SQM, rank 4','OFDM_pilot_free_marginal':'Pilot-free coded OFDM\n(TDL-A marginal ML)','OFDM_rate_matched_pilot_coded':'Pilot-aided coded\n(8-bit/16-sample)','OFDM_practical_unknown_IQ_DC':'Practical OFDM\n(unknown I/Q/DC)'}
g['label']=[labels[x] for x in g.method]; g=g.sort_values('PER_mean')
fig,ax=plt.subplots(figsize=(5.5,2.9)); y=np.arange(len(g)); ax.errorbar(g.PER_mean,y,xerr=g.PER_95CI_halfwidth_t_df2,fmt='none',ecolor='#222',capsize=3,lw=1); ax.barh(y,g.PER_mean,color=['#2878b5' if x=='SQM_rank4' else '#62a0ca' if x=='OFDM_pilot_free_marginal' else '#a6cee3' if x=='OFDM_rate_matched_pilot_coded' else '#d95f02' for x in g.method],height=.55); ax.set_yticks(y); ax.set_yticklabels(g.label); ax.set_xlabel('PER at 10 dB'); ax.set_title('Coded OFDM controls on shared three-seed draws'); ax.grid(axis='x',alpha=.25); ax.set_xlim(0,.36); fig.tight_layout(); fig.savefig(F/'fig_ofdm_revision.png',dpi=300); fig.savefig(MF/'fig_ofdm_revision.png',dpi=300); plt.close(fig)
# Fig. 10: corrected normalized-I/Q sweep, no duplicated legacy curves.
df=pd.read_csv(ROOT/'Data/operating_iq_sweep_covaware.csv'); g=df[df.receiver=='covariance-aware metric'].groupby('condition_number').PER.agg(['mean','std']).reset_index();
fig,ax=plt.subplots(figsize=(4.8,2.9)); ax.errorbar(g.condition_number,g['mean'],yerr=4.30265273*g['std']/np.sqrt(3),marker='o',capsize=3,color='#2878b5'); ax.set_xlabel('Static I/Q condition number'); ax.set_ylabel('PER at 10 dB'); ax.set_title('Known-I/Q OFDM with colored-noise metric'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/'fig_iq_sweep_covaware.png',dpi=300); fig.savefig(MF/'fig_iq_sweep_covaware.png',dpi=300); plt.close(fig)
# Fig. 12: clean complexity plot with no overlapping point labels.
df=pd.read_csv(ROOT/'Data/fractional_decoder_complexity.csv'); fig,ax=plt.subplots(figsize=(4.7,2.8));
for _,r in df.iterrows():
    label={'coarse-to-fine top-4':'top-4 shortlist','coarse-to-fine top-8':'top-8 shortlist','coarse-to-fine top-16':'top-16 shortlist','full search':'full search'}[r.decoder]
    ax.errorbar(r.relative_score_count,r.PER_mean,yerr=r.PER_95CI_halfwidth,fmt='o',capsize=3,label=label)
ax.set_xlabel('Relative candidate--CFO score count'); ax.set_ylabel('PER at 10 dB'); ax.set_title('Coarse-to-fine SQM decoding'); ax.grid(alpha=.25); ax.legend(frameon=True,fontsize=7,loc='lower right'); fig.tight_layout(); fig.savefig(F/'fig_coarsefine_clean.png',dpi=300); fig.savefig(MF/'fig_coarsefine_clean.png',dpi=300); fig.savefig(MF/'fig-016.png',dpi=300); plt.close(fig)
print('wrote clean figures')
