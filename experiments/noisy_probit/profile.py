"""One-dimensional rho integration with conditional Laplace over nuisances.

This is an application of conditional/nested Laplace ideas, not a new generic
approximation. It preserves skewness along reciprocity at extra computational
cost. The nuisance integral remains approximate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from .model import Posterior
from .experiment import simulate
from .run import source_hash,write_json


def integrate(data, joint, points=81, eta_lower=-2.):
    post=Posterior(data)
    nuisance=np.array([0,1,2,4,5])
    # Explicit tail truncation; reject rather than report if left tail is material.
    # Extreme negative rho causes cancellation in the Owen-T CDF implementation.
    grid=np.linspace(eta_lower,4,points)
    logmass=[];modes=[];issues=[]
    start=time.perf_counter();initial=np.asarray(joint['mode'])[nuisance]
    # Sweep from the center out, warm starting only on observed data.
    order=np.argsort(abs(grid-joint['mode'][3]))
    values=np.empty(points);solutions={}
    for index in order:
        eta=grid[index]
        if solutions:
            nearest=min(solutions,key=lambda k:abs(grid[k]-eta));initial=solutions[nearest]
        def expand(u):
            z=np.empty(6);z[3]=eta;z[nuisance]=u;return z
        def fun(u):
            value,g=post.value_grad(expand(u));return value,g[nuisance]
        opt=minimize(fun,initial,jac=True,method='L-BFGS-B',
                     bounds=[(None,None)]*3+[(-12,12)]*2,
                     options=dict(gtol=1e-6,ftol=1e-12,maxiter=400,maxls=30))
        z=expand(opt.x);h=post.hessian(z)[np.ix_(nuisance,nuisance)]
        sign,logdet=np.linalg.slogdet(h)
        if sign<=0 or np.linalg.eigvalsh(h).min()<=0 or np.max(abs(opt.jac))>.01:
            issues.append(dict(index=int(index),eta=float(eta),gradient=float(np.max(abs(opt.jac))),det_sign=float(sign)))
        values[index]=-opt.fun+2.5*np.log(2*np.pi)-.5*logdet if sign>0 else -np.inf
        solutions[index]=opt.x
    # Tail optimizations can be ill-conditioned where audited-state probabilities
    # reach machine precision. Keep a contiguous valid region around the peak;
    # accept truncation only if BOTH cut boundaries have negligible density.
    original_grid=grid.copy();trimmed=False
    if issues:
        peak=int(np.argmax(values));bad=[v['index'] for v in issues]
        if peak in bad:
            return dict(success=False,issues=issues,seconds=time.perf_counter()-start)
        lo=max([i+1 for i in bad if i<peak],default=0)
        hi=min([i for i in bad if i>peak],default=len(grid))
        mass=np.exp(values-values[peak])
        if (lo>0 and mass[lo]>1e-7) or (hi<len(grid) and mass[hi-1]>1e-7) or max(mass[bad])>1e-7:
            return dict(success=False,issues=issues,seconds=time.perf_counter()-start)
        grid=grid[lo:hi];values=values[lo:hi];trimmed=True
    density=np.exp(values-values.max())
    if density[0]>1e-6:
        return dict(success=False,issues=[dict(left_tail_density_ratio=float(density[0]))],seconds=time.perf_counter()-start)
    norm=np.trapezoid(density,grid);density/=norm
    cdf=np.r_[0,cumulative_trapezoid(density,grid)];cdf/=cdf[-1]
    rho=np.tanh(grid)
    lo,median,hi=np.tanh(np.interp([.025,.5,.975],cdf,grid))
    mean=np.trapezoid(rho*density,grid)
    sd=np.sqrt(np.trapezoid((rho-mean)**2*density,grid))
    return dict(success=True,mean=float(mean),median=float(median),lower=float(lo),upper=float(hi),
                sd=float(sd),seconds=time.perf_counter()-start,points=points,
                eta_grid=grid.tolist(),eta_density=density.tolist(),
                tail_trimmed=trimmed,tail_issues=issues,requested_eta_range=[float(original_grid[0]),float(original_grid[-1])],
                boundary_density_ratio=float(max(density[0],density[-1])/density.max()))


def job(args):
    path,output,points=args;path=Path(path);d=json.loads(path.read_text())
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(output)/d['case']/path.name
    if out.exists():
        previous=json.loads(out.read_text())
        if previous['profile_hash']!=code or previous['input_signature']!=d['signature'] or previous['points']!=points:
            raise ValueError('Changed integration config/code')
        return out.name,previous['success']
    if d['source_hash']!=source_hash():raise ValueError('Fit source changed')
    data=simulate(**d['config'],seed=d['seed'])
    result=integrate(data,d['results']['joint'],points)
    result.update(profile_hash=code,input_signature=d['signature'],seed=d['seed'],case=d['case'],
                  points=points,true_rho=d['truth'][3])
    write_json(out,result)
    return str(out),result['success']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/noisy_probit/minimal')
    p.add_argument('--output',default='results/noisy_probit/rho_integrated')
    p.add_argument('--cases',nargs='+',default=['moderate','strong'])
    p.add_argument('--replicates',type=int,default=100);p.add_argument('--points',type=int,default=81)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    files=[path for c in a.cases for path in sorted((Path(a.input)/c).glob('seed*.json'))[:a.replicates]]
    if len(files)!=len(a.cases)*a.replicates:raise ValueError('Missing source fits')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(job,[(str(p),a.output,a.points) for p in files]),1):
            if i%10==0 or i==len(files):print(i,'/',len(files),r,flush=True)
