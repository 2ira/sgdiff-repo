"""Import published attack graphs without re-generating attacks; verify historical hashes."""
import argparse,csv,io,json,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_peer_protocol import bind_peer
from sgdiff.data.datasets import load_graph

def main(a):
 p,n=bind_peer(a.peer)
 historical=list(csv.DictReader(a.historical.open()))
 out=a.output/'attacks';out.mkdir(parents=True,exist_ok=True)
 checks=[]
 for d in ('cora','citeseer','pubmed'):
  g=load_graph(d,root=a.data_root,download=False)
  for seed in range(3):
   split,fp=p.split_data(g,seed,a.output)
   base=n.graph_with_edges(g,np.vstack([split.train_pos,split.val_pos]))
   original=set(map(tuple,base.edges));protected=set(map(tuple,np.vstack([split.test_pos,split.test_neg])))
   archive=a.archives/(f'pubmed_seed{seed}.zip' if d=='pubmed' else 'predictions-and-graphs.zip')
   with ZipFile(archive) as z:
    for attack,ratio in n.variants():
     tag=attack if not ratio else f'{attack}_{ratio:g}'
     if attack=='feature_shuffle':
      if d=='pubmed':
       with np.load(io.BytesIO(z.read(f'pubmed_s{seed}/graph_feature_shuffle.npz'))) as f:
        np.testing.assert_array_equal(f['permutation'],np.random.default_rng(seed+7183).permutation(g.num_nodes))
      continue
     name=f'{d}_s{seed}_{tag}.npz'
     if d!='pubmed':
      payload=z.read('attacks/'+name)
      with np.load(io.BytesIO(payload)) as f:
       edges=f['edges'];stats=json.loads(str(f['statistics']));assert str(f['base_sha256'])==p.digest(base.edges)
     else:
      with np.load(io.BytesIO(z.read(f'pubmed_s{seed}/graph_{tag}.npz'))) as f:edges=f['edges']
      edited=set(map(tuple,edges));deleted=len(original-edited);inserted=len(edited-original)
      stats=dict(attack=attack,ratio=ratio,requested_budget=round(ratio*len(original)),deleted=deleted,inserted=inserted,effective_edits=deleted+inserted,base_edges=len(original),graph_sha256=p.digest(edges),implementation='imported published graph, no regeneration',archive=str(archive.name))
      b=io.BytesIO();np.savez_compressed(b,edges=edges,base_sha256=p.digest(base.edges),statistics=json.dumps(stats));payload=b.getvalue()
     expected={r['graph_sha256'] for r in historical if r['dataset']==d and int(r['seed'])==seed and r['condition']==tag}
     assert expected=={p.digest(edges)},(d,seed,tag,expected)
     assert stats['graph_sha256']==p.digest(edges)
     assert not protected.intersection(map(tuple,edges)),(d,seed,tag,'test leakage')
     assert stats['effective_edits']==round(ratio*len(original))
     target=out/name
     if target.exists(): assert target.read_bytes()==payload, target
     else: target.write_bytes(payload)
     checks.append(dict(dataset=d,seed=seed,condition=tag,graph_sha256=p.digest(edges)))
 assert len(checks)==117
 (a.output/'graph_import_audit.json').write_text(json.dumps(dict(graph_count=len(checks),graphs=checks,attack_generation=False),indent=2)+'\n')
 print('117 archived graphs verified against historical HL-GNN; no attacks generated.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['peer','data-root','archives','historical','output']:p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
