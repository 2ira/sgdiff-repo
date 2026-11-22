"""Freeze code/plan/configuration before resumable independent validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from .model import fit
from .simulation import simulate,VALIDATION,DEVELOPMENT
from ..run import write_json


def hashes():
    here=Path(__file__).parent
    files=[here/n for n in ['model.py','simulation.py','run.py','PROTOCOL.md']]+[here.parent/'model.py',here.parent/'experiment.py']
    return {str(p.relative_to(here.parent.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def job(task):
    root,case,config,seed,q,code,stage=task
    out=Path(root)/case/f'q{q:g}'/f'seed{seed}.json'
    sig=signature(dict(case=case,config=config,seed=seed,q=q,code=code))
    if hashes()!=code:raise RuntimeError('Code/plan changed during frozen run')
    if out.exists():
        old=json.loads(out.read_text());assert old['signature']==sig
        return case,q,seed,all(r['success'] for r in old['results'].values())
    data=simulate(config,seed,q)
    independent=fit(data,'independent')
    results=dict(independent=independent,copula=fit(data,'copula',independent),full=fit(data,'full',independent))
    if stage=='validation' and case=='positive' and q in [.025,.05]:
        results['copula_shrink']=fit(data,'copula_shrink',independent)
        results['full_shrink']=fit(data,'full_shrink',independent)
    data_hash=hashlib.sha256(data.x.tobytes()+data.observed.tobytes()+data.audit.tobytes()+data.truth.tobytes()).hexdigest()
    kappa=(0. if config['legacy']=='iid' else 1.) if 'legacy' in config else config['kappa']
    write_json(out,dict(case=case,config=config,seed=seed,audit_fraction=q,audited_dyads=int(data.audit.sum()),
                       dyads=len(data.x),true_density=float(data.truth.mean()),
                       truth=[*data.beta.tolist(),data.rho,data.fp,data.fn,kappa],
                       results=results,signature=sig,source_hashes=code,data_hash=data_hash))
    return case,q,seed,all(r['success'] for r in results.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','validation'],required=True)
    p.add_argument('--replicates',type=int);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--output');a=p.parse_args()
    default_reps=100 if a.stage=='development' else 200
    reps=a.replicates or default_reps
    root=Path(a.output or f'results/noisy_probit/lowdim/{a.stage}')
    cases=DEVELOPMENT if a.stage=='development' else VALIDATION
    budgets=[.05] if a.stage=='development' else [.01,.025,.05]
    code=hashes();content=dict(stage=a.stage,cases=cases,budgets=budgets,replicates=reps,source_hashes=code)
    lock=root/'LOCK.json'
    if lock.exists():
        old=json.loads(lock.read_text());assert old['design']==content,'Frozen design differs'
    else:
        write_json(lock,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),design=content,
                            lock_hash=signature(content),note='Frozen before generating/inspecting this validation run.'))
    tasks=[(str(root),case,config,seed,q,code,a.stage) for case,config in cases.items()
           for seed in range(config['seed_start'],config['seed_start']+reps) for q in budgets]
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(job,tasks),1):
            if i%20==0 or i==len(tasks):print(i,'/',len(tasks),r,flush=True)


if __name__=='__main__':main()
