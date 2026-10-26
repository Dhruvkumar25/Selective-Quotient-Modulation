#!/usr/bin/env python3
"""Compare a matrix run with the supplied PDF's rounded Table VI; no data edits."""
import argparse,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path,help='complete_1plus4_summary.csv from a manuscript-mode run')
    p.add_argument('--output',type=Path,required=True,help='New comparison CSV (existing files rejected)')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; refusing to overwrite')
    ref=json.loads((ROOT/'Validation/v8_port_check/manuscript_table_vi_reference.json').read_text())
    df=pd.read_csv(a.summary);df=df[df.snr_db.eq(ref['snr_db'])]
    rows=[]
    for panel,methods in ref['values'].items():
        for method,value in methods.items():
            q=df[df.panel.eq(panel)&df.method.eq(method)]
            if len(q)!=1:p.error(f'Expected exactly one row for {panel}/{method}')
            if int(q.iloc[0].seeds)!=5 or int(q.iloc[0].blocks)!=3000:p.error('Table VI requires 5 seeds and 3000 total blocks per point')
            got=float(q.iloc[0].PER_mean)
            rows.append({'panel':panel,'method':method,'manuscript_PER_rounded':value,'fresh_PER':got,'fresh_minus_manuscript':got-value,'matches_4dp_rounding':abs(got-value)<=ref['rounding_tolerance']})
    out=pd.DataFrame(rows);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
    print(out.to_string(index=False))
    print('Matched:',int(out.matches_4dp_rounding.sum()),'of',len(out),'rounded manuscript means. This is not a significance test.')
if __name__=='__main__':main()
