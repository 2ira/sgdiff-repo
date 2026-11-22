"""Fixed-data posterior diagnostics; does not replace validation estimators."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
from .model import make_posterior
from .density import negative_log_density
from ..lowdim.simulation import simulate
from .run import hashes
from ..mcmc import diagnostics
from ..run import write_json


def chain(task):
    record,kind,chain_id,warmup,draws,logit_bound=task
    data=simulate(record['config'],record['seed'],record['audit_fraction'])
    post=make_posterior(data,kind);fit=record['results'][kind]
    mode=np.array(fit['mode']);chol=np.linalg.cholesky(fit['covariance'])
    rng=np.random.default_rng(record['seed']+32147*(chain_id+1))
    def inside(z):
        if abs(z[3])>=4:return False
        if kind=='six':return max(abs(z[4:6]))<12 and max(abs(z[6:]))<6
        return max(abs(z[4:]))<logit_bound
    z=mode.copy()
    for _ in range(100):
        candidate=mode+1.5*chol@rng.normal(size=post.dim)
        if inside(candidate):z=candidate;break
    value=negative_log_density(post,z);scale=2.38/np.sqrt(post.dim)
    kept=[];accepted=0;window=0;started=time.perf_counter()
    for i in range(warmup+draws):
        candidate=z+scale*(chol@rng.normal(size=post.dim))
        new=negative_log_density(post,candidate) if inside(candidate) else np.inf
        accept=np.log(rng.uniform())<value-new
        if accept:z=candidate;value=new
        if i<warmup:
            window+=accept
            if (i+1)%100==0:
                scale*=np.exp((window/100-.25)/np.sqrt((i+1)/100));window=0
        else:kept.append(z.copy());accepted+=accept
    return np.array(kept),dict(acceptance=accepted/draws,scale=scale,seconds=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--kind',choices=['six','full'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=4000);p.add_argument('--draws',type=int,default=12000)
    p.add_argument('--logit-bound',type=float,default=12.)
    a=p.parse_args();record=json.loads(Path(a.input).read_text())
    assert record['source_hashes']==hashes();assert record['results'][a.kind]['success']
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4) as pool:
        runs=list(pool.map(chain,[(record,a.kind,j,a.warmup,a.draws,a.logit_bound) for j in range(4)]))
    z=np.stack([r[0] for r in runs]);diag=diagnostics(z)
    data=simulate(record['config'],record['seed'],record['audit_fraction']);post=make_posterior(data,a.kind)
    native=np.array([[post.native(row) for row in ch] for ch in z]);flat=native.reshape(-1,native.shape[-1])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'draws.npz',unconstrained=z,native=native)
    payload=dict(input=str(Path(a.input)),kind=a.kind,source_hashes=hashes(),input_signature=record['signature'],
                 warmup=a.warmup,draws=a.draws,chains=[r[1] for r in runs],diagnostics=diag,
                 converged=bool(max(diag['rank_split_rhat'])<1.01 and min(diag['bulk_ess'])>400),
                 mean=flat.mean(0).tolist(),lower=np.quantile(flat,.025,axis=0).tolist(),upper=np.quantile(flat,.975,axis=0).tolist(),
                 sd=flat.std(0,ddof=1).tolist(),laplace=record['results'][a.kind],
                 coordinate_abs_max=np.max(np.abs(z),axis=(0,1)).tolist(),
                 numerical_domains=dict(rho=4,kappa=6,rate_or_row_logits=a.logit_bound if a.kind=='full' else 12),
                 sampler_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'result.json',payload)
    print(json.dumps({k:payload[k] for k in ['kind','converged','diagnostics','mean']}),flush=True)


if __name__=='__main__':main()
