"""Bundle real scores, frozen splits, and training provenance for CPU-only replay."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from experiments.new_models_rank.summarize import read,dump,write,VARIANTS,CONDITIONS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 out=a.publish;out.mkdir(parents=True,exist_ok=True)
 prep=json.loads((a.manifests/'preparation_audit.json').read_text());assert prep['weights_count']==81 and prep['evaluations']==1134
 checkpoints=[];training=[];histories=[]
 with ZipFile(out/'training_records.zip','w',ZIP_DEFLATED,compresslevel=9) as z:
  for w in prep['weights']:
   d,m,v,s=w['dataset'],w['model'],w['variant'],w['seed'];name=f'{d}_{m}_{v}_s{s}';run=(a.evaluation if d=='pubmed' else a.previous)/'runs'/name
   assert sha(run/'model.pt')==w['checkpoint_sha256']
   checkpoints.append(dict(**w,persistent_path=str((run/'model.pt').resolve()),size_bytes=(run/'model.pt').stat().st_size))
   info=json.loads((run/'training.json').read_text())
   training.append(dict(dataset=d,model=m,variant=v,seed=s,origin=w['origin'],best_epoch=info['best_epoch'],stopped_epoch=info['stopped_epoch'],best_val_auc=info['best_val_auc'],seconds=info['seconds'],checkpoint_sha256=w['checkpoint_sha256']))
   for r in info['history']:histories.append(dict(dataset=d,model=m,variant=v,seed=s,**r))
   for n in ['config.json','receipt.json','training.json','complete.json']:
    p=run/n
    if p.exists():z.write(p,f'runs/{name}/{n}')
   p=a.evaluation/'logs'/(name+'.log')
   if p.exists():z.write(p,'logs/'+p.name)
 write(out/'training_summary.csv',training);dump(out/'checkpoints.json',checkpoints);dump(out/'preparation_audit.json',prep)
 # Portable manifests retain exact score NPZ bytes and per-query ordering evidence.
 with ZipFile(out/'predictions.zip','w',ZIP_DEFLATED,compresslevel=9) as z:
  included=set()
  for mp in sorted(a.manifests.glob('*_seed*.json')):
   m=json.loads(mp.read_text());d,s=m['dataset'],m['seed'];split=(mp.parent/m['split_path']).resolve();split_member=f'splits/{split.name}'
   z.write(split,split_member);m['split_path']='../'+split_member
   for record in m['records']:
    source=(mp.parent/record['scores_path']).resolve();name=f'{d}_{record["model"]}_{record["variant"]}_s{s}';dest=f'runs/{name}/{source.name}'
    assert dest not in included;included.add(dest);z.write(source,dest);record['scores_path']='../'+dest
    record['scores_file_sha256']=sha(source)
   z.writestr('manifests/'+mp.name,json.dumps(m,indent=2)+'\n')
  z.writestr('origin.json',json.dumps(dict(evaluations=len(included),peer_commit='95083da808e500c02da458e205f4b861ceb80871',rank_archive_sha256=sha(a.rank_archive),query_provenance='Native predict receives vstack(test_pos,test_neg); ordered split hashes audited; original NPZ bytes unchanged',weights=prep['weights']),indent=2)+'\n')
  assert len(included)==1134
 with ZipFile(out/'per_seed_curves.zip','w',ZIP_DEFLATED,compresslevel=9) as z:
  for folder in sorted(a.curves.glob('*_seed*')):
   for n in ['audit.json','node_ranks.csv','per_seed_curves.csv','paired_deltas.csv']:z.write(folder/n,f'curves/{folder.name}/{n}')
 for name in ['predictions.zip','training_records.zip','per_seed_curves.zip']:
  with ZipFile(out/name) as z:assert z.testzip() is None
 dump(out/'package_audit.json',dict(passed=True,evaluations=1134,models=81,reused_weights=54,new_weights=27,peer_rank_archive_sha256=sha(a.rank_archive),archive_sha256={n:sha(out/n) for n in ['predictions.zip','training_records.zip','per_seed_curves.zip']}))
 print('Packaged 81 fits and 1134 actual score arrays',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['evaluation','previous','manifests','publish','curves','rank-archive']:p.add_argument('--'+x,type=Path,required=True)
 main(p.parse_args())
