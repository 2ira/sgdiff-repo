"""Recover lost HL-GNN artifacts with frozen historical configs and a seed-0 diagnostic gate."""
import argparse,hashlib,importlib.metadata,json,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
 old=json.loads(a.historical.read_text())['formal']
 env={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['torch','torch-geometric','numpy','scipy','scikit-learn','networkx']},'configuration_search':False,'new_seeds':False,'test_used_for_selection':False,'pilot_validation_diagnostic_tolerance':a.gate_tolerance}
 (a.output/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 def train(d,v,s):
  label=f'{d}_readme_exact_{v}_s{s}'
  cmd=[sys.executable,'-u',str(ROOT/'scripts/train_hlgnn_full.py'),'--peer',str(a.peer),'--data-root',str(a.data_root),'--output',str(a.output),'--dataset',d,'--profile','readme_exact','--variant',v,'--seed',str(s),'--epochs','150','--eval-every','5','--patience','15','--device','cuda','--threads','4']
  (a.output/'status.json').write_text(json.dumps(dict(stage='training',current=label,updated=time.time()),indent=2))
  with (a.output/'logs'/f'{label}.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  r=json.loads((a.output/'models'/label/'complete.json').read_text())
  print(json.dumps(dict(run=label,best_epoch=r['best_epoch'],validation_auc=r['best_validation']['auc'])),flush=True)
  return r
 if a.prefetch_seed is not None:
  gates=json.loads((a.output/'pilot_gates.json').read_text())
  assert len(gates)==3 and all(g['passed'] for g in gates), 'All seed-0 gates must pass first'
  for variant in ['full','fixed_weights','shallow_k2']:train('pubmed',variant,a.prefetch_seed)
  return
 gates=[]
 for d in ['cora','citeseer','pubmed']:
  r=train(d,'full',0);previous=old[r['run']]['complete'];diff=r['best_validation']['auc']-previous['best_validation']['auc']
  gate=dict(dataset=d,seed=0,new_val_auc=r['best_validation']['auc'],old_val_auc=previous['best_validation']['auc'],delta=diff,passed=abs(diff)<=a.gate_tolerance)
  gates.append(gate);(a.output/'pilot_gates.json').write_text(json.dumps(gates,indent=2)+'\n')
  if not gate['passed']:raise RuntimeError(f'Investigate seed 0 validation discrepancy before remaining fits: {gate}')
 for d in ['cora','citeseer','pubmed']:
  for seed in [0,1,2]:
   for variant in ['full','fixed_weights','shallow_k2']:
    if seed==0 and variant=='full':continue
    train(d,variant,seed)
 assert len(list((a.output/'models').glob('*/complete.json')))==27
 (a.output/'TRAINING_COMPLETE').write_text('27 fits complete; full, fixed_weights, shallow_k2; seeds 0/1/2.\n')
 (a.output/'status.json').write_text(json.dumps(dict(stage='evaluation',updated=time.time()),indent=2))
 cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_hlgnn_full.py'),'--peer',str(a.peer),'--data-root',str(a.data_root),'--training',str(a.output),'--output',str(a.evaluation)]
 subprocess.run(cmd,check=True)
 (a.output/'status.json').write_text(json.dumps(dict(stage='complete',updated=time.time()),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['peer','data-root','historical','output','evaluation']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--gate-tolerance',type=float,default=.005)
 p.add_argument('--prefetch-seed',type=int,choices=[1,2],help='Optional bounded worker; per-fit locks prevent duplicate training')
 main(p.parse_args())
