"""Prepare exact-graph author PPR caches; no checkpoint or attack is generated."""
import argparse,sys,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiments.new_models_rank.peer import bind_peer

def one(job):
 peer,source,data,output,edges,label=job
 protocol,native=bind_peer(Path(peer))
 from experiments.new_link_models.adapters import ppr_matrix
 from sgdiff.data.datasets import load_graph
 graph=load_graph('pubmed',root=data,download=False);visible=native.graph_with_edges(graph,edges)
 ppr_matrix(visible,Path(source)/'lpformer',Path(output)/'ppr_cache',eps=1e-5)
 print('PPR_READY',label,protocol.digest(edges),flush=True)

def main(a):
 protocol,native=bind_peer(a.peer)
 from sgdiff.data.datasets import load_graph
 graph=load_graph('pubmed',root=a.data_root,download=False)
 dest=a.output/'ppr_cache';dest.mkdir(exist_ok=True)
 jobs=[];reused=[];seen=set()
 for seed in range(3):
  split,_=protocol.split_data(graph,seed,a.output)
  items=[(split.train_pos,f'train_s{seed}')]
  for cache in sorted((a.output/'attacks').glob(f'pubmed_s{seed}_*.npz')):
   with np.load(cache) as z:items.append((z['edges'].copy(),cache.stem))
  for edges,label in items:
   sha=protocol.digest(edges)
   if sha in seen:continue
   seen.add(sha);name=f'ppr_{sha}_a0.15_e1e-05.npz';old=a.reuse/name
   if old.exists():
    with np.load(old) as z:assert str(z['graph_sha256'])==sha
    target=dest/name
    if not target.exists():shutil.copy2(old,target)
    reused.append(dict(label=label,graph_sha256=sha,source=str(old)))
   else:jobs.append(tuple(map(str,[a.peer,a.source_root,a.data_root,a.output]))+(edges,label))
 (a.output/'ppr_preparation.json').write_text(json.dumps(dict(unique_graphs=len(seen),reused=reused,new=len(jobs),weights_reused=False),indent=2)+'\n')
 print('PPR_PLAN',len(seen),'graphs;',len(reused),'reused;',len(jobs),'new',flush=True)
 with ProcessPoolExecutor(max_workers=a.jobs) as p:list(p.map(one,jobs))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','source-root','data-root','output','reuse']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--jobs',type=int,default=2);main(p.parse_args())
