"""Regression checks for exact shared rankings, including rejection of a changed train graph."""
import io,json,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1]/'node_rank_auc'
sys.path.insert(0,str(HERE))
from evaluate import run as original,digest
from experiments.new_models_rank.frozen_rank import run

def setup(tmp_path):
 train=np.array([[0,1],[0,2],[0,3],[3,4]],dtype=np.int64)
 pos=np.array([[1,2],[1,3]],dtype=np.int64);neg=np.array([[2,4],[1,4]],dtype=np.int64)
 np.savez(tmp_path/'split.npz',train_pos=train,test_pos=pos,test_neg=neg)
 np.savez(tmp_path/'scores.npz',positive=[.8,.6],negative=[.7,.1],test_pos=pos,test_neg=neg)
 manifest=dict(dataset='tiny',seed=0,num_nodes=5,split_path='split.npz',records=[dict(model='toy',variant='full',condition='clean',scores_path='scores.npz',expected_auc=.75)])
 p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest));old=tmp_path/'original';original(p,old)
 arc=tmp_path/'frozen.zip'
 with ZipFile(arc,'w',ZIP_DEFLATED) as z:
  for n in ['audit.json','node_ranks.csv']:z.write(old/n,'curves/hlgnn_tiny_seed0/'+n)
 return p,arc,old

def test_shared_orders_and_auc_are_identical(tmp_path):
 p,arc,old=setup(tmp_path);out=tmp_path/'new';run(p,out,arc)
 assert (out/'node_ranks.csv').read_bytes()==(old/'node_ranks.csv').read_bytes()
 assert (out/'per_seed_curves.csv').read_bytes()==(old/'per_seed_curves.csv').read_bytes()
 a=json.loads((out/'audit.json').read_text());assert a['frozen_ranking']['member'].endswith('node_ranks.csv')

def test_changed_training_graph_rejects_frozen_rank(tmp_path):
 p,arc,old=setup(tmp_path)
 with np.load(tmp_path/'split.npz') as z:data={k:z[k] for k in z.files}
 data['train_pos']=data['train_pos'][:-1]
 np.savez(tmp_path/'split.npz',**data)
 with pytest.raises(AssertionError):run(p,tmp_path/'changed',arc)
