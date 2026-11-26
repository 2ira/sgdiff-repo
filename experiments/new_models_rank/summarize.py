"""Generic aggregation and audits for the three new families, using paired split seeds."""
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

DATASETS=['cora','citeseer','pubmed']
VARIANTS={'ncnc':['full','no_completion','no_cn'],'lpformer':['full','uniform_attention','no_ppr_encoding'],'refined_gae':['full','no_norm','dot_decoder']}
CONDITIONS=['clean','rf_0.1','rf_0.25','rf_0.5','dice_0.1','dice_0.25','dice_0.5','ea_0.1','ea_0.25','ea_0.5','delete_0.5','delete_hub_0.5','insert_0.5','feature_shuffle']
SELECTED=['clean','ea_0.5','delete_hub_0.5','feature_shuffle']
RANKS=['degree','pagerank']
KEYS=['dataset','model','variant','condition','rank','k']

def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def write(p,rows):
 rows=list(rows)
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def meanstd(v):
 return (statistics.mean(v),statistics.stdev(v) if len(v)>1 else None) if v else (None,None)
def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 raw=[];checks=[];audits=[]
 with ZipFile(a.rank_archive) as z:
  for path in sorted(a.curves.glob('*_seed*/per_seed_curves.csv')):
   rows=read(path);au=json.loads(path.with_name('audit.json').read_text());d,s=au['dataset'],au['seed']
   assert path.with_name('node_ranks.csv').read_bytes()==z.read(f'curves/hlgnn_{d}_seed{s}/node_ranks.csv')
   assert path.with_name('node_ranks.csv').read_bytes()==z.read(f'curves/katz_{d}_seed{s}/node_ranks.csv')
   peer=json.loads(z.read(f'curves/hlgnn_{d}_seed{s}/audit.json'))
   assert peer['array_hashes']==au['array_hashes']
   peer_graphs={r['condition']:r['graph_sha256'] for r in peer['records']}
   assert len(rows)==126*202 and len(au['records'])==126
   for r in au['records']:
    assert r['graph_sha256']==peer_graphs[r['condition']]
    assert r['published_auc_error']<=1e-10 and r['sklearn_error']<=1e-12
   raw+=rows;audits.append(au)
   checks.append(dict(dataset=d,seed=s,rankings_byte_identical_to_hlgnn_and_katz=True,queries_identical=True,graphs_identical=True))
 if not a.partial:assert len(raw)==229068 and len(checks)==9
 for r in raw:
  r['seed']=int(r['seed']);r['k']=int(r['k']);r['auc']=float(r['auc']) if r['auc'] else None
  for k in ['n_pos','n_neg','low_support']:r[k]=int(r[k])
  for k in ['pos_coverage','neg_coverage']:r[k]=float(r[k])
 lookup={tuple(r[k] for k in KEYS)+ (r['seed'],):r for r in raw};assert len(lookup)==len(raw)
 groups=defaultdict(list)
 for r in raw:groups[tuple(r[k] for k in KEYS)].append(r)
 summary=[];delta=[];conditional=[]
 for key,rr in sorted(groups.items()):
  assert len(rr)==3 and {r['seed'] for r in rr}=={0,1,2}
  vals=[r['auc'] for r in rr if r['auc'] is not None];m,sd=meanstd(vals)
  row=dict(zip(KEYS,key));row.update(n_seeds_present=len(rr),n_seeds_valid=len(vals),mean=m,sd=sd,min_n_pos=min(r['n_pos'] for r in rr),min_n_neg=min(r['n_neg'] for r in rr),n_low_support=sum(r['low_support'] for r in rr),mean_pos_coverage=statistics.mean(r['pos_coverage'] for r in rr),mean_neg_coverage=statistics.mean(r['neg_coverage'] for r in rr))
  summary.append(row)
  if key[3]!='clean':
   diffs=[]
   for r in rr:
    c=lookup[(r['dataset'],r['model'],r['variant'],'clean',r['rank'],r['k'],r['seed'])]
    assert (c['n_pos'],c['n_neg'])==(r['n_pos'],r['n_neg'])
    if r['auc'] is not None and c['auc'] is not None:diffs.append(r['auc']-c['auc'])
   m,sd=meanstd(diffs);delta.append({**row,'mean':m,'sd':sd})
  if key[2]!='full':
   by_seed=[]
   for r in sorted(rr,key=lambda x:x['seed']):
    f=lookup[(r['dataset'],r['model'],'full',r['condition'],r['rank'],r['k'],r['seed'])]
    assert (f['n_pos'],f['n_neg'])==(r['n_pos'],r['n_neg'])
    by_seed.append(100*(f['auc']-r['auc']) if r['auc'] is not None else None)
   m,sd=meanstd([v for v in by_seed if v is not None])
   conditional.append(dict(dataset=key[0],model=key[1],variant=key[2],condition=key[3],rank=key[4],k=key[5],mean_pp=m,sd_pp=sd,seed0_pp=by_seed[0],seed1_pp=by_seed[1],seed2_pp=by_seed[2],min_n_pos=row['min_n_pos'],min_n_neg=row['min_n_neg'],n_low_support=row['n_low_support']))
 write(a.output/'summary.csv',summary);write(a.output/'delta_summary.csv',delta);write(a.output/'key_points.csv',[r for r in summary if r['k'] in [1,5,10,20,50,100]])
 write(a.output/'conditional_component_deltas.csv',conditional)
 write(a.output/'conditional_component_key_points.csv',[r for r in conditional if r['k'] in [1,5,10,20,50,100]])
 native=read(a.native);nl={(r['dataset'],r['model'],r['variant'],r['condition'],int(r['seed'])):r for r in native}
 assert len(nl)==len(native)
 for r in raw:
  if r['k']==100:assert abs(r['auc']-float(nl[(r['dataset'],r['model'],r['variant'],r['condition'],r['seed'])]['auc']))<1e-12
  if r['k']==0:assert r['auc'] is None and r['n_pos']==r['n_neg']==0
 native_groups=defaultdict(list)
 for r in native:native_groups[(r['dataset'],r['model'],r['variant'],r['condition'])].append(r)
 robust=[];paired=[]
 for key,rr in sorted(native_groups.items()):
  assert len(rr)==3
  row=dict(zip(['dataset','model','variant','condition'],key));row['n_seeds']=3
  for k in ['auc','ap','hits100','precision100']:row[k+'_mean'],row[k+'_sd']=meanstd([float(r[k]) for r in rr])
  row['drop_auc_mean'],row['drop_auc_sd']=meanstd([float(nl[(r['dataset'],r['model'],r['variant'],'clean',int(r['seed']))]['auc'])-float(r['auc']) for r in rr]);robust.append(row)
  if key[2]!='full':
   vals=[100*(float(nl[(r['dataset'],r['model'],'full',r['condition'],int(r['seed']))]['auc'])-float(r['auc'])) for r in sorted(rr,key=lambda x:int(x['seed']))]
   m,sd=meanstd(vals);paired.append(dict(dataset=key[0],model=key[1],variant=key[2],condition=key[3],mean_pp=m,sd_pp=sd,seed0_pp=vals[0],seed1_pp=vals[1],seed2_pp=vals[2]))
 write(a.output/'per_seed.csv',native);write(a.output/'robustness_summary.csv',robust);write(a.output/'paired_component_deltas.csv',paired)
 peer=[r for r in read(a.peer_summary) if (r['model'],r['variant']) in [('katz','full_h5'),('hlgnn','full')]]
 write(a.output/'five_model_full_summary.csv',peer+[r for r in summary if r['variant']=='full'])
 expected=dict(original_predictions=1134,curve_rows=229068,summary_rows=76356,conditional_component_rows=50904,global_summary_rows=378,global_component_rows=252)
 if not a.partial:
  assert len(native)==expected['original_predictions'];assert len(summary)==expected['summary_rows'];assert len(conditional)==expected['conditional_component_rows'];assert len(robust)==378 and len(paired)==252
 dump(a.output/'delivery_audit.json',dict(passed=True,partial=a.partial,original_predictions=len(native),curve_rows=len(raw),summary_rows=len(summary),conditional_component_rows=len(conditional),global_summary_rows=len(robust),checks=checks,k0_undefined=True,k100_matches_all_original_auc=True,paired_differences_before_seed_aggregation=True,sample_standard_deviation_ddof=1,rank_adapter='frozen_rank.py; published rankings retained verbatim',expected_complete_grid=expected))
 print('SUMMARIZED',len(native),'predictions,',len(raw),'curves',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['curves','native','output','rank-archive','peer-summary']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--partial',action='store_true');main(p.parse_args())
