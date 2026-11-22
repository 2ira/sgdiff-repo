"""Paired simulation, fitting and transparent failure records."""
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from .model import Data, Posterior


def simulate(n=60, beta=(1.,-.5,2.), rho=.8, fp=.05, fn=.2, seed=0,
             audit_fraction=.05, noise='iid'):
    rng = np.random.default_rng(seed)
    m = n*(n-1)//2
    x = rng.normal(size=(m,2,len(beta))); x[:,:,0] = 1
    eps = rng.normal(size=(m,2)); eps[:,1] = rho*eps[:,0]+np.sqrt(1-rho*rho)*eps[:,1]
    y = (x@np.asarray(beta)+eps > 0).astype(int)
    uniforms = rng.uniform(size=(m,2))
    if noise == 'correlated':
        # Shared uniform gives dependent directional measurement errors.
        uniforms[:,1] = uniforms[:,0]
    if noise == 'heterogeneous':
        # Correct marginal base rate is not asserted: explicit x-dependent channel.
        fpa = expit(logit(max(fp,1e-9)) + 1.2*x[:,:,1])
        fna = expit(logit(max(fn,1e-9)) + 1.2*x[:,:,1])
    elif noise in ('iid','correlated'):
        fpa,fna = fp,fn
    else:
        raise ValueError(noise)
    observed = np.where(y==0, uniforms<fpa, uniforms>=fna).astype(int)
    audit = np.zeros(m,bool)
    audit[rng.permutation(m)[:max(0,int(round(audit_fraction*m)))]] = True
    return Data(x,observed,y,audit,np.array(beta),rho,fp,fn,n)


def pack_result(posterior, mode, hessian, duration, optimizer, starts):
    eigen = np.linalg.eigvalsh(hessian)
    good = bool(np.isfinite(eigen).all() and eigen.min()>1e-7)
    covariance = np.linalg.inv(hessian) if good else None
    beta,rho,rates = posterior.decode(mode)
    native = np.r_[beta,rho,rates] if posterior.rates is None else np.r_[beta,rho]
    gradient_norm = float(np.max(np.abs(posterior.value_grad(mode)[1])))
    success = good and gradient_norm<.005 and np.isfinite(optimizer.fun)
    result = dict(success=bool(success), optimizer_success=bool(optimizer.success),
                  message=str(optimizer.message), value=float(optimizer.fun),
                  gradient_max=gradient_norm, hessian_eigenvalues=eigen.tolist(),
                  mode=mode.tolist(), estimate=native.tolist(), seconds=duration,
                  starts=starts, iterations=int(optimizer.nit),
                  covariance=None, lower=None, upper=None)
    if good:
        sd = np.sqrt(np.diag(covariance))
        lower,upper = mode-1.95996398454*sd,mode+1.95996398454*sd
        lo_b,lo_r,lo_e = posterior.decode(lower); hi_b,hi_r,hi_e = posterior.decode(upper)
        result.update(covariance=covariance.tolist(),
                      lower=np.r_[lo_b,lo_r,lo_e].tolist() if posterior.rates is None else np.r_[lo_b,lo_r].tolist(),
                      upper=np.r_[hi_b,hi_r,hi_e].tolist() if posterior.rates is None else np.r_[hi_b,hi_r].tolist())
    return result


def fit(posterior, initial=None):
    started = time.perf_counter()
    if initial is None:
        base = np.zeros(posterior.dim)
        # Starting values use observed data only, never generative parameters.
        from scipy.special import ndtri
        base[0] = ndtri(np.clip(posterior.data.observed.mean(),.02,.98))
        base[posterior.p] = np.arctanh(.3)
        if posterior.rates is None:
            base[-2:] = logit(np.array([.05,.15])/.5)
    else:
        base = np.asarray(initial).copy()
    points = [base]
    if posterior.rates is None:
        second = base.copy(); second[posterior.p] = np.arctanh(.8)
        second[-2:] = logit(np.array([.15,.3])/.5); points.append(second)
    solutions=[]
    bounds=[(None,None)]*posterior.p+[(-4.,4.)]
    if posterior.rates is None:bounds += [(-12.,12.),(-12.,12.)]
    for point in points:
        sol = minimize(posterior.value_grad, point, jac=True, method='L-BFGS-B',bounds=bounds,
                       options=dict(gtol=1e-6,ftol=1e-12,maxiter=500,maxls=30))
        solutions.append(sol)
    opt = min(solutions,key=lambda s:s.fun if np.isfinite(s.fun) else np.inf)
    hessian = posterior.hessian(opt.x)
    result = pack_result(posterior,opt.x,hessian,time.perf_counter()-started,opt,
                         [dict(value=float(s.fun), gradient_max=float(np.max(np.abs(s.jac))),
                               success=bool(s.success)) for s in solutions])
    result['at_numerical_bound']=bool(abs(opt.x[posterior.p])>3.999 or
                                    (posterior.rates is None and max(abs(opt.x[-2:]))>11.999))
    if result['at_numerical_bound']:result['success']=False
    return result


def conditional_covariance(joint, p=3):
    """Same joint mode, noise coordinates held fixed: isolate nuisance uncertainty."""
    import copy
    result = copy.deepcopy(joint)
    if joint['covariance'] is None:
        return result
    full = np.asarray(joint['covariance'])
    hessian = np.linalg.inv(full)
    cov = np.linalg.inv(hessian[:p+1,:p+1])
    z=np.asarray(joint['mode'])[:p+1]; sd=np.sqrt(np.diag(cov))
    lo,hi=z-1.95996398454*sd,z+1.95996398454*sd
    lo[-1],hi[-1]=np.tanh(lo[-1]),np.tanh(hi[-1])
    result.update(estimate=joint['estimate'][:p+1],mode=z.tolist(),covariance=cov.tolist(),
                  lower=lo.tolist(),upper=hi.tolist(),seconds=0.,derived_from='joint')
    return result


def experiment(config, seed):
    data=simulate(**config,seed=seed)
    no_audit=np.zeros(len(data.x),bool)
    clean=replace(data,observed=data.truth,audit=no_audit)
    results={}
    results['clean']=fit(Posterior(clean,rates=(0.,0.)))
    results['naive']=fit(Posterior(data,rates=(0.,0.),audit=no_audit))
    results['oracle']=fit(Posterior(data,rates=(data.fp,data.fn),audit=no_audit))
    audited_y, audited_o=data.truth[data.audit],data.observed[data.audit]
    negatives=audited_y==0; positives=audited_y==1
    fp_hat=((audited_o[negatives]==1).sum()+.5)/(negatives.sum()+1)
    fn_hat=((audited_o[positives]==0).sum()+.5)/(positives.sum()+1)
    rates=np.clip([fp_hat,fn_hat],1e-5,.499)
    results['audit_plugin']=fit(Posterior(data,rates=rates))
    results['joint']=fit(Posterior(data))
    results['joint_conditional']=conditional_covariance(results['joint'])
    return dict(config=config,seed=seed,truth=np.r_[data.beta,data.rho,data.fp,data.fn].tolist(),
                dyads=len(data.x),audited_dyads=int(data.audit.sum()),
                true_density=float(data.truth.mean()),observed_density=float(data.observed.mean()),
                audited_true_zeros=int(negatives.sum()),audited_true_ones=int(positives.sum()),
                plugin_rates=rates.tolist(),results=results)
