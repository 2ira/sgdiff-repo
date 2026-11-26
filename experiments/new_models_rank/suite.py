"""Schedule exactly 27 new PubMed fits without repeating completed configurations."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from experiments.new_models_rank.train_pubmed import VARIANTS

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
 # First run the three existing full/seed0 configurations as acceptance fits.
 jobs=[(m,'full',0) for m in VARIANTS]
 jobs += [(m,v,s) for m in VARIANTS for v in VARIANTS[m] for s in range(3) if (m,v,s) not in jobs]
 def run(job):
  m,v,s=job;log=a.output/'logs'/f'pubmed_{m}_{v}_s{s}.log'
  cmd=[sys.executable,'-u','-m','experiments.new_models_rank.train_pubmed','--peer',str(a.peer),'--data-root',str(a.data_root),'--source-root',str(a.source_root),'--output',str(a.output),'--model',m,'--variant',v,'--seed',str(s)]
  with log.open('a') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'Fit failed; inspect {log}')
  print('FINISHED',m,v,s,flush=True)
 for job in jobs[:3]:run(job)
 with ThreadPoolExecutor(max_workers=a.jobs) as p:list(p.map(run,jobs[3:]))
 complete=list((a.output/'runs').glob('pubmed_*/complete.json'));assert len(complete)==27
 (a.output/'PUBMED_COMPLETE.json').write_text(json.dumps(dict(training=27,evaluations=378,attack_generation=False),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['peer','data-root','source-root','output']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--jobs',type=int,default=2);main(p.parse_args())
