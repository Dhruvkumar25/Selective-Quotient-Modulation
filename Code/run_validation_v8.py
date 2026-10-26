#!/usr/bin/env python3
"""Run selected, source-mapped stages of the supervisor's V8 notebook.

No historical data or codebook is replaced. See docs/V8_REPRODUCIBILITY.md.
"""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / 'Code/validation_v8/stages'
BASE = ['correction_setup', 'independent_setup', 'core_imports']
COMP = ['comparison_setup', 'comparison_helpers']
HWI = COMP + ['cube_setup', 'hwi_setup']
MATRIX = HWI + ['mocz_helpers', 'mocz_radius', 'normalization', 'line_helpers', 'two_tap_helpers']
TASKS = {
    'math': BASE + ['mathematics'],
    'intersection': BASE + ['intersection'],
    'transient': BASE + ['transient_statistics'],
    'adaptive': BASE + ['boundary_delays', 'transient_statistics', 's6_corrected', 'censoring'],
    'sensitivity': BASE + ['boundary_delays', 'sensitivity'],
    'margins': BASE + ['margin_all_messages'],
    'rank': BASE + ['independent_helpers', 'independent_bank', 'rank_sweep'],
    'cfo': BASE + ['independent_helpers', 'cfo_reconstruction'],
    'coarse-fine': BASE + ['independent_helpers', 'independent_bank', 'coarse_fine'],
    'ablation': COMP + ['ablation'],
    'cube-flat': COMP + ['cube_setup', 'cube_flat'],
    'hwi-flat': HWI + ['hwi_flat'],
    'matrix': MATRIX + ['matrix_setup', 'matrix_run', 'matrix_statistics'],
    'closest-hwi': HWI + ['ablation', 'closest_hwi'],
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def frozen_inputs():
    return {str(p.relative_to(ROOT)): sha256(p) for d in ('Data','Artifact') for p in sorted((ROOT/d).rglob('*')) if p.is_file()}

def display(value):
    print(value.to_string(index=False, max_rows=24) if hasattr(value, 'to_string') else value)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=TASKS)
    ap.add_argument('--mode', choices=['smoke','manuscript'], default='smoke')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--check', action='store_true', help='Compile stages and check dependency availability; no simulations')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--hwi-codebook', help='Optional supervisor-supplied cached NPZ (key codebook, shape 256x16). Recorded by hash; never replaces Artifact/.')
    ap.add_argument('--output', help='New directory inside Runs/. Existing directories are rejected.')
    a = ap.parse_args()
    if a.list:
        for task, stages in TASKS.items(): print(f'{task:14} ' + ' -> '.join(stages))
        return 0
    if a.check:
        for path in sorted(STAGES.glob('*.py')): compile(path.read_text(), str(path), 'exec')
        for package in ['numpy','pandas','scipy','matplotlib','tqdm']: print(package, importlib.metadata.version(package))
        assert (ROOT/'Code/config.json').is_file()
        assert len(frozen_inputs()) > 0
        print('Stage compilation and dependency checks passed. No simulation was run.')
        return 0
    if not a.task: ap.error('--task, --check or --list is required')
    if a.task in ('adaptive','sensitivity') and a.mode != 'manuscript':
        ap.error('adaptive/sensitivity preserve all original validation and final seeds. Use --mode manuscript; these are not smoke runs.')
    run_dir = Path(a.output) if a.output else Path('Runs')/f'v8_{a.mode}_{a.task}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if not run_dir.is_absolute(): run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((ROOT/'Runs').resolve()) or run_dir == ROOT/'Runs':
        ap.error('--output must be a subdirectory inside this repository’s Runs/')
    if run_dir.exists(): ap.error('Output directory already exists; choose a new directory. Outputs are never silently reused.')
    print('Task:',a.task,'Mode:',a.mode,'Output:',run_dir)
    print('Stages:',', '.join(TASKS[a.task]))
    if a.dry_run: return 0
    # Set before importing NumPy. These controls avoid accidental machine-dependent oversubscription.
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key] = '1'
    os.environ['MPLBACKEND']='Agg'
    os.environ['SQM_COMPARISON_MODE']=a.mode
    os.environ['SQM_V8_SCALE']=a.mode
    for key, full, smoke in [('SQM_IND_RANK_BLOCKS','1200','8'),('SQM_IND_CFO_BLOCKS','2500','8'),('SQM_IND_CTF_BLOCKS','900','8'),('SQM_MARGIN_CASES','500','2')]:
        os.environ[key]= full if a.mode=='manuscript' else smoke
    run_dir.mkdir(parents=True)
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT/'Code'))
    before=frozen_inputs()
    source_map=ROOT/'Code/validation_v8/source_map.json'
    manifest={'task':a.task,'mode':a.mode,'status':'running','python':sys.version,'platform':platform.platform(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_map_sha256':sha256(source_map),
        'stage_sha256':{s:sha256(STAGES/f'{s}.py') for s in TASKS[a.task]},'input_sha256':before,
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'Code').rglob('*')) if p.suffix in ('.py','.json')},
        'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','tqdm']},
        'environment':{k:v for k,v in os.environ.items() if k.startswith('SQM_') or k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'warning':'Smoke runs test execution only. Single-seed zero CI half-width is NOT evidence of certainty. Independent runs do not guarantee manuscript values.'}
    def save(): (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.hwi_codebook:
        if a.task not in ('matrix','closest-hwi','hwi-flat'): ap.error('--hwi-codebook is only used by matrix, closest-hwi or hwi-flat')
        import numpy as np
        import shutil
        supplied=Path(a.hwi_codebook).expanduser().resolve()
        with np.load(supplied, allow_pickle=False) as data:
            bank=data['codebook']
            if bank.shape!=(256,16) or not np.isfinite(bank).all() or not np.allclose(np.sum(abs(bank)**2,axis=1),16,atol=1e-6):
                ap.error('HWI codebook must be finite 256x16 with energy 16 per row; no automatic normalization is performed')
        dest=run_dir/'comparative_baselines_ablation'/'hwi_aware_grassmann_single_stream_codebook.npz'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(supplied,dest)
        manifest['supplied_hwi_codebook_sha256']=sha256(supplied)
    save();(run_dir/'source_map_snapshot.json').write_bytes(source_map.read_bytes());start=time.time()
    namespace={'__name__':'__v8_stages__','ROOT':ROOT,'RUN_DIR':run_dir,'RESULTS':run_dir,'REPRO':run_dir,'display':display}
    code=0
    try:
        for stage in TASKS[a.task]:
            manifest['current_stage']=stage;save()
            print('\n===',stage,'===',flush=True)
            p=STAGES/f'{stage}.py';namespace['__file__']=str(p)
            exec(compile(p.read_text(),str(p),'exec'),namespace)
        manifest['status']='completed'
    except Exception as e:
        manifest['status']='failed';manifest['error']=repr(e);traceback.print_exc();code=1
    finally:
        after=frozen_inputs()
        manifest['historical_inputs_unchanged']=before==after
        if before!=after:
            manifest['status']='failed';manifest['integrity_error']='Historical Data/ or Artifact/ changed';code=1
        manifest['elapsed_seconds']=time.time()-start
        manifest['outputs_sha256']={str(p.relative_to(run_dir)):sha256(p) for p in run_dir.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
        save()
    print('Status:',manifest['status'],'Historical inputs unchanged:',manifest['historical_inputs_unchanged'])
    return code
if __name__=='__main__': raise SystemExit(main())
