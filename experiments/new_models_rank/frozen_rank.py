"""Use published node orders verbatim, retaining the existing evaluator's scoring logic."""
import csv,io,json,hashlib,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
HERE=Path(__file__).resolve().parents[1]/'node_rank_auc'
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import evaluate as scoring


def run(manifest,output,archive):
 m=json.loads(Path(manifest).read_text());d,s=m['dataset'],int(m['seed'])
 prefix=f'curves/hlgnn_{d}_seed{s}/'
 with ZipFile(archive) as z:
  payload=z.read(prefix+'node_ranks.csv');prior=json.loads(z.read(prefix+'audit.json'))
 rows=list(csv.DictReader(io.StringIO(payload.decode())))
 original=scoring.ranks;checks={}
 def frozen(n,train,dataset,seed):
  assert (dataset,seed)==(d,s)
  assert prior['array_hashes']['train_pos']==scoring.digest(train)
  computed,tie=original(n,train,dataset,seed);result={}
  for method in ['degree','pagerank']:
   rr=sorted([r for r in rows if r['rank']==method],key=lambda r:int(r['node_id']))
   assert len(rr)==n and all(int(r['node_id'])==i for i,r in enumerate(rr))
   assert [r['tie_key'] for r in rr]==tie
   values=np.array([float(r['value']) for r in rr]);position=np.array([int(r['position']) for r in rr],dtype=np.int64)
   np.testing.assert_array_equal(np.sort(position),np.arange(1,n+1))
   np.testing.assert_allclose(values,computed[method][0],rtol=1e-11,atol=1e-14)
   if method=='degree':np.testing.assert_array_equal(position,computed[method][2])
   # Numerical PageRank ties can exchange order across dependency versions.
   order=np.argsort(position);assert np.all(np.diff(values[order])<=1e-15)
   result[method]=(values,order,position)
   checks[method]=dict(recomputed_position_differences=int(np.count_nonzero(position!=computed[method][2])),max_value_error=float(np.max(abs(values-computed[method][0]))))
  return result,tie
 scoring.ranks=frozen
 try:curves=scoring.run(manifest,output)
 finally:scoring.ranks=original
 assert (Path(output)/'node_ranks.csv').read_bytes()==payload,'Frozen rank CSV must match published bytes'
 audit=Path(output)/'audit.json';a=json.loads(audit.read_text())
 a['frozen_ranking']=dict(source='published HL-GNN node_ranks.csv; same as Katz',archive_sha256=scoring.file_hash(archive),member=prefix+'node_ranks.csv',member_sha256=hashlib.sha256(payload).hexdigest(),adapter_sha256=scoring.file_hash(__file__),recomputation_checks=checks)
 audit.write_text(json.dumps(a,indent=2)+'\n')
 return curves
