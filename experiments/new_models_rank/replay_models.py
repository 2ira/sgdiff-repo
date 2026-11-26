"""Independent checkpoint reload on clean, EA50 and feature-shuffle; scores and AUROC/AP."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.new_models_rank.peer import bind_peer
from sgdiff.data.datasets import load_graph
from sgdiff.data.graph import GraphData

def main(a):
 protocol,native=bind_peer(a.peer)
 from experiments.new_models_rank.pubmed_models import PUBMED_ADAPTERS
 a.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 graph=load_graph('pubmed',root=a.data_root,download=False)
 while True:
  for file in sorted((a.evaluation/'runs').glob('pubmed_*/complete.json')):
   run=file.parent;target=a.output/(run.name+'.json')
   if target.exists():continue
   c=json.loads(file.read_text());cfg=c['receipt'];m,v,s=cfg['model'],cfg['variant'],cfg['seed']
   h=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest();assert h==c['checkpoint_sha256']
   split,fp=protocol.split_data(graph,s,a.evaluation);assert fp==cfg['split_sha256']
   pairs=np.vstack([split.test_pos,split.test_neg]);assert protocol.digest(pairs)==cfg['query_sha256']
   model=PUBMED_ADAPTERS[m](a.source_root,'pubmed',graph.num_features,v,a.device,a.evaluation/'ppr_cache')
   model.load_state_dict(torch.load(run/'model.pt',map_location=a.device,weights_only=True));rows=[]
   for tag in ['clean','ea_0.5','feature_shuffle']:
    with np.load(a.evaluation/'attacks'/f'pubmed_s{s}_{"clean" if tag=="feature_shuffle" else tag}.npz') as z:edges=z['edges']
    x=graph.x[np.random.default_rng(s+7183).permutation(graph.num_nodes)] if tag=='feature_shuffle' else graph.x
    visible=GraphData(graph.name,graph.num_nodes,edges,x=x,y=graph.y)
    native.seed_all(s+20000);scores,_=native.predict(model,visible,pairs,512)
    with np.load(run/f'predictions_{tag}.npz') as z:expected=np.r_[z['positive'],z['negative']]
    np.testing.assert_allclose(scores,expected,rtol=2e-5,atol=2e-5)
    vals=protocol.metrics(scores[:len(split.test_pos)],scores[len(split.test_pos):]);prior=protocol.metrics(expected[:len(split.test_pos)],expected[len(split.test_pos):])
    err=max(abs(vals[k]-prior[k]) for k in ['auc','ap']);assert err<=1e-5
    rows.append(dict(condition=tag,max_score_error=float(np.max(abs(scores-expected))),auc_ap_error=err))
   target.write_text(json.dumps(dict(passed=True,run=run.name,checkpoint_sha256=h,score_rtol=2e-5,score_atol=2e-5,conditions=rows),indent=2)+'\n')
   model.to('cpu');del model;gc.collect();torch.cuda.empty_cache();print('REPLAYED',run.name,flush=True)
  files=list(a.output.glob('pubmed_*.json'))
  if len(files)==27:break
  if not a.watch:break
  time.sleep(10)
 records=[json.loads(p.read_text()) for p in a.output.glob('pubmed_*.json')]
 audit=dict(passed=len(records)==27,checkpoints=len(records),evaluations=sum(len(r['conditions']) for r in records),max_auc_ap_error=max((c['auc_ap_error'] for r in records for c in r['conditions']),default=0),max_absolute_score_error=max((c['max_score_error'] for r in records for c in r['conditions']),default=0),score_rtol=2e-5,score_atol=2e-5)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','source-root','data-root','evaluation','output']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--watch',action='store_true');p.add_argument('--device',default='cuda');main(p.parse_args())
