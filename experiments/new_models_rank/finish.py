"""Finish the local delivery after 27 fits; never commit or push a repository."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from experiments.new_models_rank.prepare_scores import prepare
from experiments.new_models_rank.frozen_rank import run
from experiments.new_models_rank.summarize import main as summarize
from experiments.new_models_rank.plotting import main as plot
from experiments.new_models_rank.package import main as package
from experiments.new_models_rank.report import main as report

def call(module,*args):subprocess.run([sys.executable,'-m','experiments.new_models_rank.'+module,*map(str,args)],check=True)
def main(a):
 while len(list((a.evaluation/'runs').glob('pubmed_*/complete.json')))<27:
  if not a.watch:raise RuntimeError('27 PubMed fits not complete')
  time.sleep(10)
 manifests=a.work/'manifests'
 paths=prepare(a.peer,a.evaluation,a.previous,manifests,['cora','citeseer','pubmed'])
 curves=a.work/'curves'
 for m in paths:
  out=curves/m.stem
  if out.exists():
   old=json.loads((out/'audit.json').read_text());assert old['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest()
  else:run(m,out,a.rank_archive)
 summarize(SimpleNamespace(curves=curves,native=manifests/'per_seed.csv',output=a.publish,rank_archive=a.rank_archive,peer_summary=a.repo/'results/node-rank-auc/summary.csv',partial=False))
 plot(SimpleNamespace(results=a.publish,output=a.publish))
 package(SimpleNamespace(evaluation=a.evaluation,previous=a.previous,manifests=manifests,publish=a.publish,curves=curves,rank_archive=a.rank_archive))
 for n in ['data_match_audit.json','graph_import_audit.json','ppr_preparation.json']:
  shutil.copy2(a.evaluation/n,a.publish/n)
 for n in ['metric_tests.log']:
  if (a.work/n).exists():shutil.copy2(a.work/n,a.publish/n)
 call('parameters','--results',a.publish)
 call('training_plots','--records',a.publish/'training_records.zip','--output',a.publish)
 report(SimpleNamespace(output=a.publish))
 (a.work/'RESULTS_READY').write_text('Local results complete. User approval required before push.\n')
 print('LOCAL_RESULTS_READY',a.publish,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','evaluation','previous','work','publish','rank-archive','repo']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--watch',action='store_true');main(p.parse_args())
