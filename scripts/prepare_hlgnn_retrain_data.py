"""Restore pinned data and historical inputs; never generate a new attack."""
import argparse, concurrent.futures, hashlib, json, shutil, sys, urllib.request
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--peer',type=Path,required=True)
parser.add_argument('--historical',type=Path,required=True)
a=parser.parse_args()
B=a.workspace.resolve()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
(B/'training').mkdir(parents=True,exist_ok=True)
from evaluate_peer_protocol import bind_peer
from sgdiff.data.datasets import load_graph
p,native=bind_peer(a.peer)
REV='af6abf468a772cc055bec88c52e4d6f51b7c37e3'
def download(item):
 d,s=item; target=B/'data/raw'/d/f'ind.{d}.{s}';target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  url=f'https://raw.githubusercontent.com/kimiyoung/planetoid/{REV}/data/ind.{d}.{s}'
  payload=urllib.request.urlopen(url,timeout=120).read();target.write_bytes(payload)
 return {'path':str(target.relative_to(B)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 files=list(pool.map(download,[(d,s) for d in ['cora','citeseer','pubmed'] for s in ['x','y','tx','ty','allx','ally','graph','test.index']]))
(B/'data_receipt.json').write_text(json.dumps(files,indent=2))
old=json.loads((a.historical/'provenance.json').read_text())
for d in ['cora','citeseer','pubmed']:
 g=load_graph(d,root=B/'data',download=False)
 for seed in range(3):
  sp,fp=p.split_data(g,seed,B/'training')
  exp=next(x for x in old if x['dataset']==d and x['seed']==seed)
  assert fp==exp['split'],(d,seed,'split')
  assert p.digest(g.x)==exp['features'],(d,'features')
  assert p.digest(np.vstack([sp.test_pos,sp.test_neg]))==exp['query_sha256']
 print('VERIFIED',d,g.num_nodes,g.num_features,flush=True)
shutil.copyfile(a.historical/'selection.json',B/'training/selection.json')
(B/'DATA_VERIFIED').write_text('All 9 splits and feature/query hashes match historical HL-GNN.\n')
print('DATA_READY',flush=True)
