"""Adapt real author-model predictions to node-rank inputs with explicit provenance."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from experiments.new_models_rank.peer import bind_peer,REVISION
from experiments.new_models_rank.train_pubmed import VARIANTS

def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(peer,evaluation,previous,manifests,datasets,allow_partial=False):
 protocol,native=bind_peer(Path(peer));evaluation=Path(evaluation).resolve();previous=Path(previous).resolve();manifests=Path(manifests).resolve();manifests.mkdir(parents=True,exist_ok=True)
 checks=[];global_rows=[];paths=[]
 for d in datasets:
  for seed in range(3):
   with np.load(evaluation/'splits'/f'{d}_seed{seed}.npz') as z:sp={k:z[k].copy() for k in z.files}
   records=[]
   for model,components in VARIANTS.items():
    for v in components:
     name=f'{d}_{model}_{v}_s{seed}';run=(evaluation if d=='pubmed' else previous)/'runs'/name
     if not (run/'complete.json').exists():
      if allow_partial:continue
      raise FileNotFoundError(run/'complete.json')
     complete=json.loads((run/'complete.json').read_text());new=d=='pubmed'
     config=complete['receipt'] if new else complete['config']
     # Native source and the full ordered split are immutable for reused weights.
     fp=hashlib.sha256(json.dumps({k:protocol.digest(sp[k]) for k in sp},sort_keys=True).encode()).hexdigest()
     assert fp==config['split_sha256']
     if not new:
      for filename,expected in config['code_sha256'].items():assert sha(Path(native.__file__).with_name(filename))==expected,(filename,name)
      with np.load(previous/'splits'/f'{d}_seed{seed}.npz') as z:assert all(np.array_equal(z[k],sp[k]) for k in sp)
     else:
      with np.load(run/'queries.npz') as z:assert np.array_equal(z['test_pos'],sp['test_pos']) and np.array_equal(z['test_neg'],sp['test_neg'])
      assert config['evaluator_sha256']==sha(native.__file__)
     checkpoint=sha(run/'model.pt')
     checks.append(dict(dataset=d,seed=seed,model=model,variant=v,origin='fresh PubMed fit' if new else 'reused original protected-target fit',checkpoint_sha256=checkpoint,split_sha256=fp))
     for r in complete['evaluations']:
      tag=r['attack'] if not r['ratio'] else f"{r['attack']}_{r['ratio']:g}"
      path=run/f'predictions_{tag}.npz'
      with np.load(path) as z:pos,neg=z['positive'],z['negative'];actual=protocol.metrics(pos,neg)
      for k in ['auc','ap','hits100','precision100']:assert abs(actual[k]-r['metrics'][k])<1e-12
      source_tag='clean' if tag=='feature_shuffle' else tag
      with np.load(evaluation/'attacks'/f'{d}_s{seed}_{source_tag}.npz') as z:
       graph=protocol.digest(z['edges']);assert graph==r['statistics']['graph_sha256']
      records.append(dict(model=model,variant=v,condition=tag,scores_path=os.path.relpath(path,manifests),positive_sha256=protocol.digest(sp['test_pos']),negative_sha256=protocol.digest(sp['test_neg']),query_order_evidence='Pinned native run.evaluate -> native.predict over vstack(test_pos,test_neg); complete/config split hash and score metrics checked by prepare_scores.py; new PubMed queries.npz checked explicitly',expected_auc=actual['auc'],graph_sha256=graph,checkpoint_sha256=checkpoint))
      global_rows.append(dict(dataset=d,seed=seed,model=model,variant=v,condition=tag,origin='fresh PubMed fit' if new else 'reused original protected-target fit',graph_sha256=graph,**{k:actual[k] for k in ['auc','ap','hits100','precision100']}))
   if not records:continue
   assert len({(r['model'],r['variant'],r['condition']) for r in records})==len(records)
   target=manifests/f'{d}_seed{seed}.json'
   dump(target,dict(dataset=d,seed=seed,num_nodes={'cora':2708,'citeseer':3327,'pubmed':19717}[d],split_path=os.path.relpath(evaluation/'splits'/f'{d}_seed{seed}.npz',manifests),protocol_commit=REVISION,records=records))
   paths.append(target)
 dump(manifests/'preparation_audit.json',dict(passed=True,weights=checks,weights_count=len(checks),evaluations=len(global_rows),new_training_count=sum(x['dataset']=='pubmed' for x in checks),reused_training_count=sum(x['dataset']!='pubmed' for x in checks)))
 with (manifests/'per_seed.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(global_rows[0]));w.writeheader();w.writerows(global_rows)
 return paths
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','evaluation','previous','manifests']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--datasets',nargs='+',default=['cora','citeseer','pubmed']);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();print(prepare(a.peer,a.evaluation,a.previous,a.manifests,a.datasets,a.allow_partial))
