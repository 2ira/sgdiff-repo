"""Prospectively locked validation; previous holdout used only as development."""
import argparse,datetime,hashlib,json,multiprocessing,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .model import fit
from ..lowdim.simulation import simulate
from ..lowdim.run import signature
from ..run import write_json

CASES={
 'state':dict(n=90,beta=[-.2,.6,1.1],rho=.5,fp=.06,fn=.22,kappa=[.75,-.25,-.25,.2],seed_start=910000),
 'common':dict(n=90,beta=[-.2,.6,1.1],rho=.5,fp=.06,fn=.22,kappa=[.5]*4,seed_start=920000),
 'asymmetric':dict(n=90,beta=[-.2,.6,1.1],rho=.5,fp=.06,fn=.22,kappa=[.75,-.65,.45,.2],seed_start=930000),
}


def hashes():
    here=Path(__file__).parent;root=here.parent.parent.parent
    files=[here/n for n in ['model.py','run.py','PROTOCOL.md']]+[
        here.parent/'model.py',here.parent/'experiment.py',here.parent/'lowdim/model.py',here.parent/'lowdim/simulation.py']
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def job(task):
    root,case,config,seed,q,code,development=task
    out=Path(root)/case/f'q{q:g}'/f'seed{seed}.json'
    sig=signature(dict(case=case,config=config,seed=seed,q=q,code=code))
    if hashes()!=code:raise RuntimeError('Frozen source changed')
    if out.exists():
        old=json.loads(out.read_text());assert old['signature']==sig
        return case,q,seed,all(r['success'] for r in old['results'].values())
    data=simulate(config,seed,q)
    if development:
        oldpath=Path('results/noisy_probit/lowdim/validation/state_dependent/q0.05')/f'seed{seed}.json'
        old=json.loads(oldpath.read_text());assert old['config']==config
        common=old['results']['copula'];full=old['results']['full']
    else:
        common=fit(data,'common');full=fit(data,'full',common)
    results=dict(common=common,state=fit(data,'state',common),full=full)
    if not development and case=='state' and q in [.025,.05]:
        results['state_shrink']=fit(data,'state_shrink',common)
        results['full_shrink']=fit(data,'full_shrink',common)
    dh=hashlib.sha256(data.x.tobytes()+data.observed.tobytes()+data.audit.tobytes()+data.truth.tobytes()).hexdigest()
    if development:assert dh==old['data_hash']
    write_json(out,dict(case=case,config=config,seed=seed,audit_fraction=q,audited_dyads=int(data.audit.sum()),
                        truth=[*data.beta.tolist(),data.rho,data.fp,data.fn,config['kappa']],results=results,
                        signature=sig,source_hashes=code,data_hash=dh,development_reuse=development))
    return case,q,seed,all(r['success'] for r in results.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','validation'],required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--replicates',type=int,default=200);p.add_argument('--output')
    a=p.parse_args();development=a.stage=='development'
    if development:
        from ..lowdim.simulation import VALIDATION
        cases={'old_failure':VALIDATION['state_dependent']};budgets=[.05]
    else:cases=CASES;budgets=[.01,.025,.05]
    root=Path(a.output or f'results/noisy_probit/state_channel/{a.stage}');code=hashes()
    design=dict(stage=a.stage,cases=cases,budgets=budgets,replicates=a.replicates,source_hashes=code)
    lock=root/'LOCK.json'
    if lock.exists():assert json.loads(lock.read_text())['design']==design
    else:write_json(lock,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),design=design,lock_hash=signature(design)))
    tasks=[(str(root),case,cfg,seed,q,code,development) for case,cfg in cases.items()
           for seed in range(cfg['seed_start'],cfg['seed_start']+a.replicates) for q in budgets]
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(job,tasks),1):
            if i%30==0 or i==len(tasks):print(i,'/',len(tasks),r,flush=True)


if __name__=='__main__':main()
