"""Run resumable paired network replicates, with code/config signatures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import multiprocessing
import time
from .experiment import experiment

ROOT=Path(__file__).resolve().parents[2]
CASES={
    'clean':dict(n=60,beta=[1.,-.5,2.],rho=.8,fp=0.,fn=0.,audit_fraction=.05,noise='iid'),
    'moderate':dict(n=60,beta=[1.,-.5,2.],rho=.8,fp=.05,fn=.2,audit_fraction=.05,noise='iid'),
    'strong':dict(n=60,beta=[1.,-.5,2.],rho=.8,fp=.1,fn=.3,audit_fraction=.05,noise='iid'),
}


def source_hash():
    paths=[Path(__file__).with_name(n) for n in ['model.py','experiment.py','run.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def job(args):
    case,config,seed,output,code=args
    out=Path(output)/case/f'seed{seed}.json'
    signature=hashlib.sha256(json.dumps(dict(config=config,seed=seed,code=code),sort_keys=True).encode()).hexdigest()
    if out.exists():
        old=json.loads(out.read_text())
        if old['signature']!=signature:raise ValueError(f'Changed code/config: {out}')
        return case,seed,True
    start=time.perf_counter()
    data=experiment(config,seed)
    data.update(signature=signature,source_hash=code,elapsed_seconds=time.perf_counter()-start,case=case)
    write_json(out,data)
    return case,seed,all(r['success'] for r in data['results'].values())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/noisy_probit/minimal')
    p.add_argument('--replicates',type=int,default=100)
    p.add_argument('--seed-start',type=int,default=202101)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--config',help='JSON mapping of case names to simulation arguments')
    a=p.parse_args()
    if min(a.replicates,a.workers)<1:p.error('Positive replicate/worker counts required')
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    code=source_hash()
    selected=json.loads(Path(a.config).read_text()) if a.config else {k:CASES[k] for k in a.cases}
    jobs=[(case,config,seed,a.output,code) for case,config in selected.items() for seed in range(a.seed_start,a.seed_start+a.replicates)]
    write_json(Path(a.output)/'manifest.json',dict(source_hash=code,cases=selected,
              replicates=a.replicates,seed_start=a.seed_start,total_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(job,jobs),1):
            if i%10==0 or i==len(jobs):print(i,'/',len(jobs),result,flush=True)


if __name__=='__main__':main()
