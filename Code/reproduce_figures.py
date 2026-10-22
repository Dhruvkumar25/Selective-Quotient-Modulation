from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'Data'
F = ROOT / 'Figure'
F.mkdir(exist_ok=True)

# 1. Main exact fractional TDL-A result
opt = pd.read_csv(D/'fractional_optimized_snr_sweep.csv')
base = pd.read_csv(D/'fractional_tdla_headtohead.csv')
plot = pd.concat([
    base[base.method.isin(['CP-OFDM TDL-LMMSE','CP-OFDM full oracle ML'])],
    opt
], ignore_index=True)
fig = plt.figure(figsize=(7.8,5.0)); ax = fig.add_subplot(111)
for name,g in plot.groupby('method', sort=False):
    ax.errorbar(g.snr_db, g.PER_mean, yerr=g.PER_95CI_halfwidth, marker='o', capsize=3, label=name)
ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Packet error rate')
ax.set_title('Exact fractional TDL-A after incidence retraining')
ax.grid(True, which='both', alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(F/'fig_main_per.png', dpi=180); plt.close(fig)

# 2. Nuisance spectrum
spec = pd.read_csv(D/'mean_singular_spectrum.csv')
fig = plt.figure(figsize=(7.2,4.8)); ax = fig.add_subplot(111)
ax.semilogy(spec['mode'], np.maximum(spec['mean_energy_fraction'],1e-12), marker='o')
ax.set_xlabel('Fractional-delay nuisance mode'); ax.set_ylabel('Mean normalized singular energy')
ax.set_title('TDL-A exact fractional-delay nuisance spectrum, L=16')
ax.grid(True, which='both', alpha=.25); fig.tight_layout(); fig.savefig(F/'fig_nuisance_spectrum.png', dpi=180); plt.close(fig)

# 3. Rank vs SNR
rs = pd.read_csv(D/'rank_vs_snr.csv')
fig = plt.figure(figsize=(7.3,4.8)); ax = fig.add_subplot(111)
for snr,g in rs.groupby('snr_db'):
    ax.plot(g['rank'], g['PER_mean'], marker='o', label=f'{snr} dB')
ax.set_yscale('log'); ax.set_xlabel('Retained nuisance modes'); ax.set_ylabel('PER')
ax.set_title('Exact fractional TDL-A: optimal nuisance rank depends on SNR')
ax.grid(True, which='both', alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(F/'fig_rank_snr.png', dpi=180); plt.close(fig)

# 4. Margin decomposition
m = pd.read_csv(D/'selective_margin_summary.csv')
fig = plt.figure(figsize=(7.3,4.8)); ax = fig.add_subplot(111)
ax.plot(m['rank'],m['mean_true_mismatch'],marker='o',label='True-model mismatch')
ax.plot(m['rank'],m['mean_wrong_distance'],marker='o',label='Nearest wrong distance')
ax.plot(m['rank'],m['mean_selective_margin'],marker='o',label='Selective margin')
ax.set_xlabel('Retained nuisance rank'); ax.set_ylabel('Normalized Frobenius distance')
ax.set_title('Selective-quotient margin decomposition'); ax.grid(True, alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(F/'fig_margin.png', dpi=180); plt.close(fig)

# 5. Complexity
c = pd.read_csv(D/'fractional_decoder_complexity.csv')
fig = plt.figure(figsize=(7.0,4.6)); ax = fig.add_subplot(111)
ax.scatter(c.relative_score_count,c.PER_mean)
for _,r in c.iterrows():
    ax.annotate(r.decoder,(r.relative_score_count,r.PER_mean),xytext=(5,5),textcoords='offset points')
ax.set_xlabel('Relative candidate-CFO score count'); ax.set_ylabel('PER at 10 dB')
ax.set_title('Selective SQM coarse-to-fine decoder'); ax.grid(True,alpha=.25)
fig.tight_layout(); fig.savefig(F/'fig_complexity.png', dpi=180); plt.close(fig)

# 6. Hardware condition sweep
iq = pd.read_csv(D/'operating_iq_sweep.csv')
fig = plt.figure(figsize=(7.4,4.8)); ax = fig.add_subplot(111)
for name,g in iq.groupby('method',sort=False):
    ax.errorbar(g.x,g.PER_mean,yerr=g.PER_95CI_halfwidth,marker='o',capsize=3,label=name)
ax.set_yscale('log'); ax.set_xlabel('Static I/Q transform condition number'); ax.set_ylabel('PER at 10 dB')
ax.set_title('Operating regime vs I/Q distortion'); ax.grid(True,which='both',alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(F/'fig_iq_sweep.png', dpi=180); plt.close(fig)

# 7. CFO sweep
cfo = pd.read_csv(D/'operating_cfo_sweep.csv')
fig = plt.figure(figsize=(7.4,4.8)); ax = fig.add_subplot(111)
for name,g in cfo.groupby('method',sort=False):
    ax.errorbar(g.x,g.PER_mean,yerr=g.PER_95CI_halfwidth,marker='o',capsize=3,label=name)
ax.set_yscale('log'); ax.set_xlabel('CFO range (± degrees/sample)'); ax.set_ylabel('PER at 10 dB')
ax.set_title('Operating regime vs CFO'); ax.grid(True,which='both',alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(F/'fig_cfo_sweep.png', dpi=180); plt.close(fig)

# 8. Adapted prior-art head-to-head
pa = pd.read_csv(D/'head_to_head.csv')
fig = plt.figure(figsize=(7.5,5.0)); ax = fig.add_subplot(111)
for name,g in pa.groupby('method',sort=False):
    ax.errorbar(g.snr_db,np.maximum(g.PER_mean,1e-5),yerr=g.PER_95CI_halfwidth,marker='o',capsize=3,label=name)
ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Packet error rate')
ax.set_title('CFO ±10°/symbol, static I/Q mixing + DC'); ax.grid(True,which='both',alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(F/'fig_priorart_adapted.png', dpi=180); plt.close(fig)

print('Regenerated paper figures in', F)

# 9. Direct BMOCZ baseline, CFO +/-8 deg/sample
bm = pd.read_csv(D/'dmm_vs_bmocz.csv')
bm = bm[bm.cfo_range_deg_per_sample == 8]
fig = plt.figure(figsize=(7.3,4.8)); ax = fig.add_subplot(111)
for name,g in bm.groupby('method',sort=False):
    ax.errorbar(g.snr_db,g.PER_mean,yerr=g.PER_95CI_halfwidth,marker='o',capsize=3,label=name)
ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Packet error rate')
ax.set_title('8-bit packet, unknown 2-tap channel, CFO +/-8 deg/sample')
ax.grid(True,which='both',alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(F/'fig_mocz_cfo8.png',dpi=180); plt.close(fig)

# 10. Grass-Lattice native flat-fading baseline
gl = pd.read_csv(D/'dmm_vs_grasslattice_flat.csv')
fig = plt.figure(figsize=(7.3,4.8)); ax = fig.add_subplot(111)
for name,g in gl.groupby('method',sort=False):
    ax.errorbar(g.snr_db,g.PER_mean,yerr=g.PER_95CI_halfwidth,marker='o',capsize=3,label=name)
ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Packet error rate')
ax.set_title('8-bit packet, unknown flat Rayleigh gain, equal 10-sample duration/energy')
ax.grid(True,which='both',alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(F/'fig_grasslattice_flat.png',dpi=180); plt.close(fig)
