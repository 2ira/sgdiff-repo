"""Audit model-independent rankings, matched attacks/queries and complete released cells."""
import argparse,hashlib,json
from pathlib import Path

def main(a):
 checks=[];cells=0;curve_rows=0;max_error=0
 expected={'katz':{'full_h5','h3','h2'},'hlgnn':{'full','fixed_weights','shallow_k2'}}
 for d in a.datasets:
  for seed in range(3):
   dirs={m:a.curves/f'{m}_{d}_seed{seed}' for m in expected}
   audits={m:json.loads((p/'audit.json').read_text()) for m,p in dirs.items()}
   rank_bytes=[(p/'node_ranks.csv').read_bytes() for p in dirs.values()]
   assert rank_bytes[0]==rank_bytes[1],(d,seed,'different rankings')
   assert audits['katz']['array_hashes']==audits['hlgnn']['array_hashes'],(d,seed,'query/train mismatch')
   graphs={};counts={}
   for m,au in audits.items():
    assert len(au['records'])==42
    assert {r['variant'] for r in au['records']}==expected[m]
    assert au['n_curve_rows']==42*202 and au['rank_source']=='train_pos only'
    assert au['protocol_commit']=='95083da808e500c02da458e205f4b861ceb80871'
    for r in au['records']:
     graphs.setdefault(r['condition'],set()).add(r['graph_sha256'])
     counts[(m,r['variant'],r['condition'])]=r
     assert r['graph_sha256'] and r['sklearn_error']<=1e-12 and r['published_auc_error']<=1e-10
     max_error=max(max_error,r['sklearn_error'],r['published_auc_error'])
    cells+=len(au['records']);curve_rows+=au['n_curve_rows']
   assert len(graphs)==14 and all(len(s)==1 for s in graphs.values()),(d,seed,'attacks mismatch')
   for variant in expected['katz']:
    clean=counts['katz',variant,'clean'];shuffled=counts['katz',variant,'feature_shuffle']
    for field in ['positive_scores_sha256','negative_scores_sha256']:assert clean[field]==shuffled[field]
   checks.append(dict(dataset=d,seed=seed,node_rank_csv_sha256=hashlib.sha256(rank_bytes[0]).hexdigest(),same_rankings=True,same_queries=True,same_14_graphs=True,katz_feature_shuffle_exactly_clean=True))
 result=dict(evaluations=cells,curve_rows=curve_rows,datasets=a.datasets,checks=checks,max_auc_recompute_error=max_error)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curves',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['cora','citeseer','pubmed']);main(p.parse_args())
