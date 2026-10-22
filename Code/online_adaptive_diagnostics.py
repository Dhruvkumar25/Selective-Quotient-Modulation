"""Additional diagnostics for the online adaptive SQM extension.

Adds results beyond PER curves: proxy calibration, rank-regret/oracle agreement,
window/forgetting-factor sensitivity, and a validation-trained proxy recommender.
All policy parameters are learned from validation traces only.
"""
from pathlib import Path
import math, json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_online_adaptive import (make_trace, all_rank_stats, rank_maps,
    thresholds_from_validation, simulate_controller, metrics, RANKS, M)
from run_final_headtohead import precompute_dmm_bases

ROOT=Path(__file__).resolve().parents[1]
SNRS=[5.0,10.0,15.0,7.0]; DELAYS=[30.0,60.0,100.0,30.0]
SEEDS=[13,59,151]

def pooled_validation(bases, seeds=SEEDS, blocks=250):
    traces=[make_trace(blocks,91000+s,SNRS,DELAYS) for s in seeds]
    sts=[all_rank_stats(t[0],bases) for t in traces]
    stats={r:tuple(np.concatenate([x[r][k] for x in sts]) for k in range(4)) for r in RANKS}
    idx=np.concatenate([t[1] for t in traces]); snr=np.concatenate([t[2] for t in traces])
    delay=np.concatenate([t[3] for t in traces]); seg=np.concatenate([t[4] for t in traces])
    return stats,idx,snr,delay,seg

def auc(values, labels):
    values=np.asarray(values,float); labels=np.asarray(labels,bool)
    n1=labels.sum(); n0=(~labels).sum()
    if n1==0 or n0==0: return np.nan
    order=np.argsort(values,kind='mergesort')
    ranks=np.empty(len(values)); ranks[order]=np.arange(1,len(values)+1)
    return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))

def rolling(x,W):
    out=np.empty(len(x)); c=np.cumsum(np.r_[0.,x])
    for t in range(len(x)):
        lo=max(0,t-W+1); out[t]=(c[t+1]-c[lo])/(t-lo+1)
    return out

def proxy_recommender(val_stats, val_seg, regime_map, test_stats):
    # Validation-trained centroid classifier using only rank-4 recent decoder state.
    vg=np.concatenate([val_stats[4][1][val_seg==seg] for seg in sorted(set(val_seg))])
    ve=np.concatenate([val_stats[4][2][val_seg==seg] for seg in sorted(set(val_seg))])
    # Preserve time order by constructing directly from the validation arrays.
    vg=val_stats[4][1]; ve=val_stats[4][2]
    vG=rolling(vg,8); vE=rolling(ve,8)
    mu=np.array([vG.mean(),vE.mean()]); sd=np.array([vG.std()+1e-9,vE.std()+1e-9])
    cent={}
    for seg in sorted(set(val_seg)):
        r=int(regime_map[int(seg)]); f=np.column_stack([(vG[val_seg==seg]-mu[0])/sd[0],(vE[val_seg==seg]-mu[1])/sd[1]])
        cent.setdefault(r,[]).append(f.mean(axis=0))
    cent={r:np.mean(v,axis=0) for r,v in cent.items()}
    tg=rolling(test_stats[4][1],8); te=rolling(test_stats[4][2],8)
    f=np.column_stack([(tg-mu[0])/sd[0],(te-mu[1])/sd[1]])
    chosen=np.array([min(cent,key=lambda r: np.sum((f[t]-cent[r])**2)) for t in range(len(f))],int)
    return chosen, {'centroids':{str(r):cent[r].tolist() for r in cent},'feature_mean':mu.tolist(),'feature_std':sd.tolist()}

def main():
    # Load artifact and nominal receiver bases.
    t0=make_trace(10,777,SNRS,DELAYS); Xd,powers,D=t0[5],t0[6],t0[7]
    bases={r:precompute_dmm_bases(Xd,powers,D,r,4.0)[1] for r in RANKS}
    vstats,vidx,vsnr,vdelay,vseg=pooled_validation(bases)
    best_fixed,snr_map,regime_map,rank_bias,segdf=rank_maps(vstats,vidx,vseg,vsnr,vdelay)
    th=thresholds_from_validation(vstats,vidx,vseg,W=8)
    # Proxy-calibrated rank recommendation, trained only on validation.
    rec_rows=[]; diag_rows=[]; sens_rows=[]
    # Calibration diagnostics on pooled validation and final traces.
    for split,seeds,blocks,base in [('validation',SEEDS,250,91000),('final',SEEDS,300,120000)]:
        all_g=[]; all_e=[]; all_err=[]; all_need=[]; all_rank=[]
        for seed in seeds:
            tr=make_trace(blocks,base+seed,SNRS,DELAYS); stats=all_rank_stats(tr[0],bases)
            chosen,model=proxy_recommender(vstats,vseg,regime_map,stats)
            oracle=np.array([regime_map[int(x)] for x in tr[4]])
            p4,g4,e4,_=stats[4]
            # Recommender uses the selected rank's actual decoder output.
            pred=np.empty(len(tr[1]),int)
            for r in RANKS: pred[chosen==r]=stats[r][0][chosen==r]
            rec_rows.append({'split':split,'seed':seed,'PER':float(np.mean(pred!=tr[1])),
                             'oracle_rank_agreement':float(np.mean(chosen==oracle)),
                             'mean_abs_rank_regret':float(np.mean(np.abs(chosen-oracle))),
                             'mean_selected_rank':float(chosen.mean()),'switch_rate':float(np.mean(np.diff(chosen)!=0)),
                             'mean_score_evals':2816.0,'p95_score_evals':2816.0})
            all_g.append(g4); all_e.append(e4); all_err.append((p4!=tr[1]).astype(int)); all_need.append((oracle!=4).astype(int)); all_rank.append(oracle)
            if split=='final':
                for name,x in [('gap',g4),('residual',e4)]:
                    q=np.quantile(x,[0,.25,.5,.75,1.0])
                    for j in range(4):
                        m=(x>=q[j]) & ((x<=q[j+1]) if j==3 else (x<q[j+1]))
                        diag_rows.append({'seed':seed,'feature':name,'quartile':j+1,'range_low':float(q[j]),'range_high':float(q[j+1]),
                                          'packets':int(m.sum()),'PER_at_rank4':float(np.mean((p4!=tr[1])[m])) if m.any() else np.nan,
                                          'need_rank_change_fraction':float(np.mean((oracle!=4)[m])) if m.any() else np.nan})
        if split=='final':
            G=np.concatenate(all_g); E=np.concatenate(all_e); ERR=np.concatenate(all_err); NEED=np.concatenate(all_need)
            diag_rows.append({'seed':'pooled','feature':'AUC','quartile':'gap_high_correct','range_low':np.nan,'range_high':np.nan,'packets':len(G),
                              'PER_at_rank4':auc(G,ERR==0),'need_rank_change_fraction':auc(-G,NEED==1)})
            diag_rows.append({'seed':'pooled','feature':'AUC','quartile':'residual_high_error','range_low':np.nan,'range_high':np.nan,'packets':len(E),
                              'PER_at_rank4':auc(E,ERR==1),'need_rank_change_fraction':auc(E,NEED==1)})
    # Sensitivity: freeze validation thresholds and vary W/alpha on final traces.
    for W in [2,4,8,16,32]:
        vals=[]
        for seed in SEEDS:
            tr=make_trace(300,120000+seed,SNRS,DELAYS); st=all_rank_stats(tr[0],bases)
            res=simulate_controller(st,tr[1],tr[4],'S4',W=W,tau_g=th['tau_g'],tau_e_hi=th['tau_e_hi'],tau_e_lo=th['tau_e_lo'],min_dwell=2)
            mm=metrics(res,tr[4],regime_map); vals.append(mm)
        sens_rows.append({'policy':'S4_window','setting':f'W={W}','W':W,'alpha':np.nan,
                          'PER_mean':np.mean([x['PER'] for x in vals]),'PER_ci95':1.96*np.std([x['PER'] for x in vals],ddof=1)/np.sqrt(len(vals)),
                          'switch_rate_mean':np.mean([x['switch_rate'] for x in vals]),'adaptation_delay_mean':np.mean([x['mean_adaptation_delay_packets'] for x in vals])})
    for alpha in [0.05,0.10,0.20,0.35,0.50]:
        vals=[]
        for seed in SEEDS:
            tr=make_trace(300,120000+seed,SNRS,DELAYS); st=all_rank_stats(tr[0],bases)
            res=simulate_controller(st,tr[1],tr[4],'S5',alpha=alpha,tau_g=th['tau_g'],tau_e_hi=th['tau_e_hi'],tau_e_lo=th['tau_e_lo'],min_dwell=2)
            mm=metrics(res,tr[4],regime_map); vals.append(mm)
        sens_rows.append({'policy':'S5_EWMA','setting':f'alpha={alpha}','W':np.nan,'alpha':alpha,
                          'PER_mean':np.mean([x['PER'] for x in vals]),'PER_ci95':1.96*np.std([x['PER'] for x in vals],ddof=1)/np.sqrt(len(vals)),
                          'switch_rate_mean':np.mean([x['switch_rate'] for x in vals]),'adaptation_delay_mean':np.mean([x['mean_adaptation_delay_packets'] for x in vals])})
    out=ROOT/'Data'; out.mkdir(exist_ok=True)
    pd.DataFrame(rec_rows).to_csv(out/'online_proxy_recommender.csv',index=False)
    pd.DataFrame(diag_rows).to_csv(out/'online_proxy_diagnostics.csv',index=False)
    pd.DataFrame(sens_rows).to_csv(out/'online_controller_sensitivity.csv',index=False)
    cfg={'validation_seeds':SEEDS,'final_seeds':SEEDS,'regime_oracle_rank':regime_map,'best_fixed_rank':best_fixed,
         'thresholds':th,'proxy_recommender':'8-packet rolling rank-4 gap/residual nearest-centroid classifier',
         'proxy_model':model}
    (out/'online_diagnostics_config.json').write_text(json.dumps(cfg,indent=2))
    print('proxy recommender summary')
    print(pd.DataFrame(rec_rows).groupby('split').mean(numeric_only=True).to_string())
    print('\nproxy diagnostics')
    print(pd.DataFrame(diag_rows).tail(8).to_string(index=False))
    print('\nsensitivity')
    print(pd.DataFrame(sens_rows).to_string(index=False))
if __name__=='__main__': main()
