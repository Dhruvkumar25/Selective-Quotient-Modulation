"""Delay-only regime-transition experiment for online SQM rank control."""
from pathlib import Path
import math, json, sys
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_online_adaptive import make_trace, all_rank_stats, rank_maps, thresholds_from_validation, simulate_controller, metrics, RANKS
from run_final_headtohead import precompute_dmm_bases
ROOT=Path(__file__).resolve().parents[1]; SEEDS=[13,59,151]
SNRS=[10.,10.,10.,10.]; DELAYS=[30.,60.,100.,30.]
def pool(bases,base,blocks):
 ts=[make_trace(blocks,base+s,SNRS,DELAYS) for s in SEEDS]; st=[all_rank_stats(t[0],bases) for t in ts]
 stats={r:tuple(np.concatenate([x[r][k] for x in st]) for k in range(4)) for r in RANKS}
 return stats,np.concatenate([t[1] for t in ts]),np.concatenate([t[2] for t in ts]),np.concatenate([t[3] for t in ts]),np.concatenate([t[4] for t in ts])
def main():
 t=make_trace(10,777,SNRS,DELAYS); Xd,powr,D=t[5],t[6],t[7]; bases={r:precompute_dmm_bases(Xd,powr,D,r,4.)[1] for r in RANKS}
 vs,vi,vsn,vd,vseg=pool(bases,91000,250); best,sm,reg,rb,segdf=rank_maps(vs,vi,vseg,vsn,vd); th=thresholds_from_validation(vs,vi,vseg,W=8)
 rows=[]; by=[]
 for seed in SEEDS:
  tr=make_trace(300,120000+seed,SNRS,DELAYS); st=all_rank_stats(tr[0],bases); oracle={int(k):int(v) for k,v in reg.items()}
  methods={
   'S0_fixed_r4':('fixed',{}),
   'S2_delay_class_lookup':('S8',{'regime_to_rank':oracle}),
   'S4_window_confidence_residual':('S4',{'W':8,'tau_g':th['tau_g'],'tau_e_hi':th['tau_e_hi'],'tau_e_lo':th['tau_e_lo'],'min_dwell':2}),
   'S5_EWMA_confidence_residual':('S5',{'alpha':0.1,'tau_g':th['tau_g'],'tau_e_hi':th['tau_e_hi'],'tau_e_lo':th['tau_e_lo'],'min_dwell':2}),
   'S8_oracle_rank':('S8',{'regime_to_rank':oracle})}
  for name,(sch,kw) in methods.items():
   if sch=='fixed':
    r=4; res={'errors':(st[r][0]!=tr[1]).astype(float),'selected':np.full(len(tr[1]),r),'evals':np.full(len(tr[1]),2816.),'switches':0}
   else: res=simulate_controller(st,tr[1],tr[4],sch,**kw)
   mm=metrics(res,tr[4],oracle); mm.update({'seed':seed,'method':name,'blocks':len(tr[1])}); rows.append(mm)
   for seg in range(4):
    m=tr[4]==seg; by.append({'seed':seed,'method':name,'segment':seg,'delay_ns':DELAYS[seg],'PER':float(res['errors'][m].mean()),'mean_rank':float(res['selected'][m].mean())})
 df=pd.DataFrame(rows); summ=[]
 for method,g in df.groupby('method',sort=False):
  row={'method':method,'PER':g.PER.mean(),'PER_ci95':1.96*g.PER.std(ddof=1)/math.sqrt(len(g)),'mean_rank':g.mean_selected_rank.mean(),'switch_rate':g.switch_rate.mean(),'adaptation_delay':g.mean_adaptation_delay_packets.mean(),'mean_evals':g.mean_score_evals.mean()}; summ.append(row)
 out=ROOT/'Data'; pd.DataFrame(summ).to_csv(out/'online_delay_only_summary.csv',index=False); pd.DataFrame(rows).to_csv(out/'online_delay_only_seed_metrics.csv',index=False); pd.DataFrame(by).to_csv(out/'online_delay_only_by_segment.csv',index=False); (out/'online_delay_only_config.json').write_text(json.dumps({'snr_trace':SNRS,'delay_trace_ns':DELAYS,'regime_rank':reg,'thresholds':th},indent=2))
 print('regime ranks',reg,'thresholds',th); print(pd.DataFrame(summ).to_string(index=False))
if __name__=='__main__': main()
