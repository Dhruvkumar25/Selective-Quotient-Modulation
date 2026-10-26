"""Small deterministic checks. These do NOT replace Monte Carlo validation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
STAGES=ROOT/'Code/validation_v8/stages'
sys.path.insert(0,str(ROOT/'Code'))
from fractional_tdla_core import centering_matrix, tdla_operators, weighted_nuisance_dictionary

def function(name, stage, namespace):
    tree=ast.parse((STAGES/f'{stage}.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),stage,'exec'),namespace)
    return namespace[name]

class ValidationV8(unittest.TestCase):
    def test_source_hashes(self):
        mapping=json.loads((STAGES.parent/'source_map.json').read_text())
        nb=ROOT/'Notebooks/SQM_Complete_Validation_Comparisons_Ablation_V8.ipynb'
        self.assertEqual(hashlib.sha256(nb.read_bytes()).hexdigest(),mapping['notebook_sha256'])
        notebook=json.loads(nb.read_text())
        for name,meta in mapping['stages'].items():
            self.assertEqual(hashlib.sha256((STAGES/f'{name}.py').read_bytes()).hexdigest(),meta['ported_sha256'])
            original=''.join(notebook['cells'][meta['cell_index_zero_based']]['source'])
            self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),meta['original_cell_sha256'])

    def test_centering_and_affine_quotient(self):
        rng=np.random.default_rng(21);H=centering_matrix(16)
        np.testing.assert_allclose(H@H,H,atol=1e-14)
        X=rng.normal(size=(2,16))@H;A=np.array([[2.,.4],[.1,.7]])
        Y=(A@X+np.array([[.2],[-.6]]))@H
        project=lambda Z:np.linalg.pinv(Z)@Z
        np.testing.assert_allclose(project(X),project(Y),atol=1e-12)

    def test_tdla_rank_saturation(self):
        a=np.load(ROOT/'Artifact/fractional_incidence_codebook.npz')['X_opt']
        _,powers,D=tdla_operators(16)
        Phi=weighted_nuisance_dictionary(a[0,0]+1j*a[0,1],0.,powers,D)
        self.assertEqual(Phi.shape,(46,16));self.assertEqual(np.linalg.matrix_rank(Phi),15)
        _,_,Vh=np.linalg.svd(Phi,full_matrices=False);Q=Vh[:15].T
        np.testing.assert_allclose(Q@Q.T,centering_matrix(16),atol=1e-10)

    def test_s6_current_packet_and_all_candidate_cost(self):
        ns={'np':np,'RANKS':[3,4,5,6],'FULL_EVALS':2816,'L_BLOCK':16}
        simulate=function('simulate_s6_same_packet','s6_corrected',ns)
        # r=4 triggers; r=5 wins immediately. On the next packet it remains active.
        stats={r:(np.array([r,r]),np.array([.01,.2]) if r!=5 else np.array([.3,.3]),np.array([.1,.1])) for r in [3,4,5,6]}
        result=simulate(stats,np.array([5,5]),tau_g=.1,rank_bias={})
        np.testing.assert_array_equal(result['pred'],[5,5])
        np.testing.assert_array_equal(result['selected'],[5,5])
        self.assertEqual(result['approx_macs'][0],2816*32*(3+4+5))
        self.assertEqual(result['approx_macs'][1],2816*32*5)
        self.assertEqual(result['evals'][0],2816*3)

    def test_censoring_and_only_true_boundaries(self):
        oracle=pd.Series([4,5,6],index=pd.MultiIndex.from_tuples([(13,0),(13,1),(13,2)]))
        ns={'np':np,'pd':pd,'oracle':oracle}
        events=function('ind_adaptation_events','censoring',ns)
        ranks=[4]*12+[4]*2+[5]*10+[5]*12
        df=pd.DataFrame({'seed':13,'method':'test','packet':range(36),'segment':np.repeat([0,1,2],12),'selected_rank':ranks})
        e=events(df,window=10)
        self.assertEqual(len(e),2);self.assertEqual(e.segment.tolist(),[1,2])
        self.assertTrue(e.iloc[0].success);self.assertEqual(e.iloc[0].delay_to_run_start,2)
        self.assertEqual(e.iloc[0].delay_to_run_confirmation,11)
        self.assertTrue(e.iloc[1].censored);self.assertTrue(pd.isna(e.iloc[1].delay_to_run_start))
        self.assertEqual(e.iloc[1].restricted_delay,12)

    def test_coarse_fine_counts(self):
        self.assertEqual(256*5+8*11,1368)
        self.assertEqual(256*5+8*(11-5),1328)
        ns={'np':np,'coarse_idx':np.array([1,3,5,7,9])}
        ctf=function('ind_ctf_predictions','coarse_fine',ns)
        scores=np.zeros((2,256,11));scores[:,7,:]=1
        np.testing.assert_array_equal(ctf(scores,8),[7,7])

    def test_energy_normalization(self):
        ns={'np':np};normal=function('normalize_rows_energy','normalization',ns)
        X=np.random.default_rng(4).normal(size=(256,16))+1j
        np.testing.assert_allclose(np.sum(abs(normal(X,8))**2,axis=1),8,atol=1e-12)

    def test_runner_rejects_unsafe_output_and_short_adaptive(self):
        import subprocess
        commands=[['--task','math','--output','Data/forbidden','--dry-run'],
                  ['--task','math','--output','Runs','--dry-run'],
                  ['--task','adaptive','--mode','smoke','--dry-run']]
        for args in commands:
            p=subprocess.run([sys.executable,str(ROOT/'Code/run_validation_v8.py'),*args],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
        self.assertFalse((ROOT/'Data/forbidden').exists())

    def test_config_matches_notebook_restoration(self):
        config=json.loads((ROOT/'Code/config.json').read_text())
        self.assertEqual(config['messages'],256);self.assertEqual(config['block_length'],16)
        self.assertEqual(config['cfo_search_grid_deg_per_sample'],[-5,5,1])

if __name__=='__main__': unittest.main()
