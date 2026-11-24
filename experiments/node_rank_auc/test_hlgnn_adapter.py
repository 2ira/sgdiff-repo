import csv,json
import numpy as np
import pytest
from evaluate import digest
from prepare_hlgnn import prepare

def test_query_reordering_is_rejected(tmp_path):
 e=tmp_path/'evaluation';(e/'splits').mkdir(parents=True)
 run=e/'runs/cora_full_s0';run.mkdir(parents=True)
 tp=np.array([[0,1],[1,2]],dtype=np.int64);tn=np.array([[0,2],[0,3]],dtype=np.int64)
 np.savez(e/'splits/cora_seed0.npz',test_pos=tp,test_neg=tn)
 pairs=np.vstack([tp,tn]);np.savez(run/'queries.npz',pairs=pairs[::-1],positive_count=2)
 (run/'receipt.json').write_text(json.dumps({'query_sha256':digest(pairs)}))
 fields=['dataset','seed','variant','condition','graph_sha256','auc']
 with (e/'per_seed.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(dataset='cora',seed=0,variant='full',condition='clean',graph_sha256='graph',auc=.5))
 with pytest.raises(AssertionError):prepare(e,tmp_path/'manifests',e/'per_seed.csv')
