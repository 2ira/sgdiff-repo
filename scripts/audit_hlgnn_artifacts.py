"""Validate complete saved models, training coverage, ablation invariants and evaluation provenance."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hlgnn.model import configuration,compute_init_temp

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main(a):
 formal=json.loads((a.historical/'training_metadata.json').read_text())['formal']
 provenance=json.loads((a.evaluation/'provenance.json').read_text())
 assert len(provenance)==27
 rows=[]
 for d in ['cora','citeseer','pubmed']:
  for seed in range(3):
   with np.load(a.training/'splits'/f'{d}_seed{seed}.npz') as f:num_pos=len(f['train_pos'])
   for variant in ['full','fixed_weights','shallow_k2']:
    label=f'{d}_readme_exact_{variant}_s{seed}';folder=a.training/'models'/label
    complete=json.loads((folder/'complete.json').read_text());receipt=json.loads((folder/'receipt.json').read_text());history=json.loads((folder/'history.json').read_text())
    assert receipt==formal[label]['receipt'],(label,'training contract changed')
    assert len(history)==complete['epochs_run']<=150
    assert complete['updates']==len(history)*math.ceil(num_pos/512)
    values=[r for r in history if 'val_auc' in r]
    best=max(values,key=lambda r:(r['val_auc'],-r['epoch']))
    assert best['epoch']==complete['best_epoch'] and best['val_auc']==complete['best_validation']['auc']
    path=folder/'best.pt';actual=digest(path);assert actual==complete['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['epoch']==complete['best_epoch'] and saved['receipt']==receipt
    assert saved['config']==configuration(d,'readme_exact')
    assert all(torch.isfinite(v).all().item() for v in saved['model'].values())
    k=2 if variant=='shallow_k2' else 20;weights=saved['model']['encoder.temp']
    assert weights.shape==(k+1,)
    if variant=='fixed_weights':assert torch.equal(weights,compute_init_temp(k,saved['config']['init'],saved['config']['alpha']))
    record=next(r for r in provenance if (r['dataset'],r['seed'],r['variant'])==(d,seed,variant))
    assert record['checkpoint_sha256']==actual
    assert record['split']==receipt['split'] and record['features']==receipt['features']
    rows.append(dict(run=label,bytes=path.stat().st_size,checkpoint_sha256=actual,epochs=len(history),updates=complete['updates'],best_epoch=best['epoch'],best_validation_auc=best['val_auc'],hops=k,fixed_weights_exactly_initial=variant=='fixed_weights',training_receipt_matches_historical=True))
    print('VERIFIED',label,flush=True)
 eval_audit=json.loads((a.evaluation/'audit.json').read_text());assert eval_audit['native_evaluator_cells']==378 and eval_audit['metric_recomputation_max_abs_error']<=1e-12
 output=dict(models_verified=len(rows),total_checkpoint_bytes=sum(r['bytes'] for r in rows),native_evaluation_cells=378,training_coverage='all fixed training pairs, every epoch',checkpoint_selection='earliest maximum clean validation AUROC',models=rows)
 a.output.write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({k:v for k,v in output.items() if k!='models'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['training','evaluation','historical','output']:p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
