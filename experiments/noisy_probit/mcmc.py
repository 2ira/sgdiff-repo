"""Independent random-walk Metropolis validation of collapsed Laplace fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from .model import Posterior
from .experiment import simulate
from .run import write_json,source_hash


def chain(task):
    config,seed,mode,cov,chain_id,warmup,draws=task
    data=simulate(**config,seed=seed);post=Posterior(data)
    rng=np.random.default_rng(seed+7919*(chain_id+1))
    mode=np.asarray(mode);chol=np.linalg.cholesky(cov)
    z=mode+chol@rng.normal(size=len(mode))*1.5
    value=post.value_grad(z)[0];scale=2.38/np.sqrt(len(mode))
    kept=[];accepted=0;window=0;start=time.perf_counter()
    for i in range(warmup+draws):
        candidate=z+scale*(chol@rng.normal(size=len(mode)))
        # Restrict to the same negligible-tail numerical domain as optimization.
        inside=abs(candidate[3])<4 and max(abs(candidate[-2:]))<12
        new=post.value_grad(candidate)[0] if inside else np.inf
        accept=np.log(rng.uniform())<value-new
        if accept:z=candidate;value=new
        if i<warmup:
            window+=accept
            if (i+1)%100==0:
                scale*=np.exp((window/100-.25)/np.sqrt((i+1)/100))
                window=0
        else:
            kept.append(z.copy());accepted+=accept
    return np.asarray(kept),dict(acceptance=accepted/draws,scale=scale,seconds=time.perf_counter()-start)


def split_chains(x):
    n=x.shape[1]//2
    return np.concatenate([x[:,:n],x[:,-n:]],axis=0)


def rhat(x):
    x=split_chains(x);n=x.shape[1]
    within=x.var(axis=1,ddof=1).mean(axis=0)
    between=n*x.mean(axis=1).var(axis=0,ddof=1)
    return np.sqrt(((n-1)/n*within+between/n)/within)


def diagnostics(x):
    ranked=np.empty_like(x)
    for j in range(x.shape[2]):
        ranks=rankdata(x[:,:,j]).reshape(x.shape[:2])
        ranked[:,:,j]=ndtri((ranks-.375)/(ranks.size+.25))
    folded=np.abs(x-np.median(x,axis=(0,1)))
    folded_rank=np.empty_like(x)
    for j in range(x.shape[2]):
        ranks=rankdata(folded[:,:,j]).reshape(x.shape[:2])
        folded_rank[:,:,j]=ndtri((ranks-.375)/(ranks.size+.25))
    rh=np.maximum(rhat(ranked),rhat(folded_rank))
    # Split rank-normalized ESS with the initial positive paired sequence.
    a=split_chains(ranked);m,n,p=a.shape
    centered=a-a.mean(axis=1,keepdims=True)
    freq=np.fft.rfft(centered,n=2*n,axis=1)
    acov=np.fft.irfft(freq*np.conj(freq),n=2*n,axis=1)[:,:n]/n
    w=a.var(axis=1,ddof=1).mean(axis=0)
    varplus=(n-1)/n*w+a.mean(axis=1).var(axis=0,ddof=1)
    autocorr=1-(w-acov.mean(axis=0))/varplus
    autocorr[0]=1
    ess=[]
    for j in range(p):
        total=0.;previous=np.inf
        for t in range(0,n-1,2):
            pair=autocorr[t,j]+autocorr[t+1,j]
            if pair<0:break
            pair=min(pair,previous);previous=pair;total+=pair
        ess.append(min(m*n,m*n/max(-1+2*total,1.)))
    return dict(rank_split_rhat=rh.tolist(),bulk_ess=ess)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--draws',type=int,default=8000);a=p.parse_args()
    d=json.loads(Path(a.input).read_text());fit=d['results']['joint']
    if d['source_hash']!=source_hash():raise ValueError('Source changed since fit')
    if not fit['success']:raise ValueError('Need a successful joint fit')
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    tasks=[(d['config'],d['seed'],fit['mode'],fit['covariance'],i,a.warmup,a.draws) for i in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(chain,tasks))
    draws=np.stack([r[0] for r in results]);diagnostic=diagnostics(draws)
    native=draws.copy();native[:,:,3]=np.tanh(native[:,:,3]);native[:,:,4:]=.5/(1+np.exp(-native[:,:,4:]))
    flat=native.reshape(-1,6)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'draws.npz',unconstrained=draws,native=native)
    payload=dict(input=str(Path(a.input).resolve()),source_hash=source_hash(),warmup=a.warmup,draws=a.draws,
        diagnostics=diagnostic,chains=[r[1] for r in results],mean=flat.mean(axis=0).tolist(),
        sd=flat.std(axis=0,ddof=1).tolist(),lower=np.quantile(flat,.025,axis=0).tolist(),
        upper=np.quantile(flat,.975,axis=0).tolist(),laplace=fit,
        converged=bool(max(diagnostic['rank_split_rhat'])<1.01 and min(diagnostic['bulk_ess'])>400))
    write_json(out/'result.json',payload)
    print(json.dumps({k:payload[k] for k in ['converged','diagnostics','mean','sd']}),flush=True)


if __name__=='__main__':main()
