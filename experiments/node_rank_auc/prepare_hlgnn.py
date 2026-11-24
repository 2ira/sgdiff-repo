"""Adapt audited HL-GNN predictions to the shared Top-k% metric, without training."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from evaluate import digest

def prepare(evaluation, output, historical):
 evaluation=Path(evaluation).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((evaluation/'per_seed.csv').open()))
 old=list(csv.DictReader(Path(historical).open()))
 graphs={(r['dataset'],int(r['seed']),r['condition']):r['graph_sha256'] for r in old}
 sizes={'cora':2708,'citeseer':3327,'pubmed':19717}
 manifests=[]
 for dataset,seed in sorted({(r['dataset'],int(r['seed'])) for r in rows}):
  path=evaluation/'splits'/f'{dataset}_seed{seed}.npz'
  with np.load(path) as f:tp,tn=f['test_pos'],f['test_neg']
  records=[]
  for row in rows:
   if row['dataset']!=dataset or int(row['seed'])!=seed:continue
   variant,condition=row['variant'],row['condition']
   run=evaluation/'runs'/f'{dataset}_{variant}_s{seed}'
   receipt=json.loads((run/'receipt.json').read_text())
   with np.load(run/'queries.npz') as f:
    np.testing.assert_array_equal(f['pairs'],np.vstack([tp,tn]));assert int(f['positive_count'])==len(tp)
   assert receipt['query_sha256']==digest(np.vstack([tp,tn]))
   assert row['graph_sha256']==graphs[(dataset,seed,condition)]
   records.append(dict(model='hlgnn',variant=variant,condition=condition,
    scores_path=os.path.relpath(run/f'predictions_{condition}.npz',output),positive_sha256=digest(tp),negative_sha256=digest(tn),
    query_order_evidence='Explicit queries.npz + receipt query hash verified by prepare_hlgnn.py',
    expected_auc=float(row['auc']),graph_sha256=row['graph_sha256'],checkpoint_sha256=receipt['checkpoint_sha256']))
  assert len({(r['variant'],r['condition']) for r in records})==len(records)
  m=dict(dataset=dataset,seed=seed,num_nodes=sizes[dataset],split_path=os.path.relpath(path,output),
   protocol_commit='95083da808e500c02da458e205f4b861ceb80871',records=records)
  target=output/f'{dataset}_seed{seed}.json';target.write_text(json.dumps(m,indent=2)+'\n');manifests.append(str(target))
 print(json.dumps({'manifests':manifests,'evaluations':len(rows)}))
 return manifests
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['evaluation','output','historical']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();prepare(a.evaluation,a.output,a.historical)
