"""Fresh PubMed author fits, on the existing shared graphs; never regenerate an attack."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.new_models_rank.peer import bind_peer,REVISION
from sgdiff.data.datasets import load_graph

VARIANTS={'ncnc':['full','no_completion','no_cn'],'lpformer':['full','uniform_attention','no_ppr_encoding'],'refined_gae':['full','no_norm','dot_decoder']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def fit_and_evaluate(a):
 protocol,native=bind_peer(a.peer)
 from experiments.new_models_rank.pubmed_models import PUBMED_ADAPTERS,pubmed_settings
 from experiments.new_link_models.sources import verify
 assert a.variant in VARIANTS[a.model]
 torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 native.seed_all(a.seed)
 graph=load_graph('pubmed',root=a.data_root,download=False)
 split,fp=protocol.split_data(graph,a.seed,a.output)
 base=native.graph_with_edges(graph,np.vstack([split.train_pos,split.val_pos]))
 graphs={}
 for attack,ratio in native.variants():
  tag=attack if not ratio else f'{attack}_{ratio:g}'
  if attack=='feature_shuffle':continue
  cache=a.output/'attacks'/f'pubmed_s{a.seed}_{tag}.npz'
  if not cache.exists():raise FileNotFoundError(f'Archived attack required: {cache}')
  with np.load(cache,allow_pickle=False) as z:
   assert str(z['base_sha256'])==protocol.digest(base.edges)
   stats=json.loads(str(z['statistics']));assert stats['graph_sha256']==protocol.digest(z['edges'])
   assert not set(map(tuple,z['edges'])).intersection(map(tuple,np.vstack([split.test_pos,split.test_neg])))
   graphs[tag]=stats['graph_sha256']
 cfg=pubmed_settings(a.model)
 args=SimpleNamespace(dataset='pubmed',seed=a.seed,model=a.model,variant=a.variant,output=a.output,epochs=100,eval_every=5,eval_batch_size=512,patience=30)
 run=a.output/'runs'/f'pubmed_{a.model}_{a.variant}_s{a.seed}';run.mkdir(parents=True,exist_ok=True)
 receipt=dict(dataset='pubmed',model=a.model,variant=a.variant,seed=a.seed,peer_commit=REVISION,
  evaluator_sha256=sha(native.__file__),training_function_sha256=sha(native.__file__),pubmed_adapter_sha256=sha(Path(__file__).with_name('pubmed_models.py')),
  settings=cfg,max_epochs=100,eval_every=5,patience_epochs=30,split_sha256=fp,features_sha256=protocol.digest(graph.x),graph_sha256=protocol.digest(graph.edges),
  query_sha256=protocol.digest(np.vstack([split.test_pos,split.test_neg])),graph_hashes=graphs,
  features='raw Planetoid attributes; no column standardization',validation='train graph only',
  origin='new fit for shared protected-target protocol; main-supplement weights are not reused',
  author_sources=json.loads(Path(native.__file__).with_name('sources.lock.json').read_text()))
 if (run/'receipt.json').exists():assert json.loads((run/'receipt.json').read_text())==receipt,'Input/implementation changed'
 else:dump(run/'receipt.json',receipt)
 if (run/'complete.json').exists():print('ALREADY_COMPLETE',run,flush=True);return
 np.savez_compressed(run/'queries.npz',test_pos=split.test_pos,test_neg=split.test_neg,pairs=np.vstack([split.test_pos,split.test_neg]))
 model=PUBMED_ADAPTERS[a.model](a.source_root,'pubmed',graph.num_features,a.variant,a.device,a.output/'ppr_cache')
 if (run/'training.json').exists():
  model.load_state_dict(torch.load(run/'model.pt',map_location=a.device,weights_only=True));training=json.loads((run/'training.json').read_text())
 else:training=native.fit(model,graph,split,cfg,args,run)
 # Even an inadvertently missing graph is fatal, rather than a new attack.
 def no_new_attack(*x,**kw):raise RuntimeError('Attack generation forbidden in shared-cache experiment')
 native.perturb=no_new_attack
 records=native.evaluate(model,graph,split,args,run)
 assert len(records)==14
 for r in records:
  tag=r['attack'] if not r['ratio'] else f"{r['attack']}_{r['ratio']:g}"
  assert r['statistics']['graph_sha256']==graphs['clean' if r['attack']=='feature_shuffle' else tag]
  with np.load(run/f'predictions_{tag}.npz') as z:actual=protocol.metrics(z['positive'],z['negative'])
  assert actual==r['metrics']
 dump(run/'complete.json',dict(receipt=receipt,training=training,evaluations=records,checkpoint_sha256=sha(run/'model.pt'),peak_cuda_bytes=torch.cuda.max_memory_allocated(),finished=True))
 print('COMPLETE',run.name,flush=True)
def main(a):
 protocol,_=bind_peer(a.peer)
 worker=a.output/'workers'/f'pubmed_{a.model}_{a.variant}_s{a.seed}'
 with protocol.cache_lock(worker):
  fit_and_evaluate(a)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','data-root','source-root','output']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--model',choices=VARIANTS,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=3);main(p.parse_args())
