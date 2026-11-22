"""Prospectively locked, paired directional-channel validation."""
import argparse,datetime,hashlib,json,multiprocessing,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .model import fit
from ..lowdim.simulation import simulate
from ..lowdim.run import signature
from ..run import write_json

CASES={
 'asymmetric':dict(n=90,beta=[-.2,.6,1.1],rho=.5,fp=.06,fn=.22,kappa=[.75,-.65,.45,.2],seed_start=1010000),
 'symmetric':dict(n=90,beta=[-.2,.6,1.1],rho=.5,fp=.06,fn=.22,kappa=[.75,-.25,-.25,.2],seed_start=1020000),
}


def hashes():
    here=Path(__file__).parent;root=here.parent.parent.parent
    files=[here/n for n in ['model.py','run.py','density.py','PROTOCOL.md']]+[
        here.parent/n for n in ['model.py','experiment.py','lowdim/model.py','lowdim/simulation.py','state_channel/model.py']]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def job(task):
    root,case,cfg,seed,q,code=task;out=Path(root)/case/f'q{q:g}'/f'seed{seed}.json'
    sig=signature(dict(case=case,config=cfg,seed=seed,q=q,code=code))
    assert hashes()==code
    if out.exists():
        old=json.loads(out.read_text());assert old['signature']==sig
        return case,q,seed,all(r['success'] for r in old['results'].values())
    d=simulate(cfg,seed,q);common=fit(d,'common')
    results=dict(common=common,state=fit(d,'state',common),six=fit(d,'six',common),full=fit(d,'full',common))
    if case=='asymmetric':
        results['six_shrink']=fit(d,'six_shrink',common);results['full_shrink']=fit(d,'full_shrink',common)
    dh=hashlib.sha256(d.x.tobytes()+d.observed.tobytes()+d.audit.tobytes()+d.truth.tobytes()).hexdigest()
    write_json(out,dict(case=case,config=cfg,seed=seed,audit_fraction=q,audited_dyads=int(d.audit.sum()),
        truth=[*d.beta.tolist(),d.rho,d.fp,d.fn,cfg['kappa']],results=results,signature=sig,source_hashes=code,data_hash=dh))
    return case,q,seed,all(r['success'] for r in results.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);p.add_argument('--replicates',type=int,default=400)
    p.add_argument('--output',default='results/noisy_probit/directional_channel/validation');a=p.parse_args()
    root=Path(a.output);code=hashes();budgets=[.025,.05]
    design=dict(cases=CASES,budgets=budgets,replicates=a.replicates,source_hashes=code)
    lock=root/'LOCK.json'
    if lock.exists():assert json.loads(lock.read_text())['design']==design
    else:write_json(lock,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),design=design,lock_hash=signature(design)))
    tasks=[(str(root),case,cfg,seed,q,code) for case,cfg in CASES.items() for seed in range(cfg['seed_start'],cfg['seed_start']+a.replicates) for q in budgets]
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(job,tasks),1):
            if i%40==0 or i==len(tasks):print(i,'/',len(tasks),r,flush=True)


if __name__=='__main__':main()
