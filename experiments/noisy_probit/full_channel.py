"""Exploratory robustness component: unrestricted 4x4 dyad measurement channel.

Each true-state row has a Dirichlet(1/2,...,1/2) prior. The 12 free
logits, beta and atanh(rho) are fit jointly with all Jacobians. This allows
direction-dependent and dependent errors, but remains homogeneous in x.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, log_expit
from .model import Posterior, state_probabilities
from .experiment import simulate
from .run import source_hash, write_json


class FullChannel(Posterior):
    def __init__(self,data,alpha=.5):
        super().__init__(data)
        self.dim=16;self.alpha=alpha
        self.obs_index=2*data.observed[:,0]+data.observed[:,1]

    def unpack(self,z):
        logits=np.column_stack([z[4:].reshape(4,3),np.zeros(4)])
        logc=logits-logsumexp(logits,axis=1,keepdims=True)
        return z[:3],np.tanh(z[3]),np.exp(logc),logc

    def value_grad(self,z):
        beta,rho,c,logc=self.unpack(z)
        p,dp=state_probabilities(self.data.x,beta,rho)
        w=c[:,self.obs_index].T*self.allowed
        likelihood=np.maximum((p*w).sum(1),1e-14)
        responsibility=p*w/likelihood[:,None]
        gtheta=(dp*w[:,:,None]).sum(1)/likelihood[:,None]
        gtheta[:,3]*=1-rho*rho
        # Sum posterior state counts by observed category, then softmax score.
        counts=np.column_stack([responsibility[self.obs_index==k].sum(0) for k in range(4)])
        score=counts-counts.sum(1)[:,None]*c
        score+=self.alpha-4*self.alpha*c  # Dirichlet + softmax Jacobian.
        value=np.log(likelihood).sum()-.5*beta@self.precision@beta
        value+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3])+self.alpha*logc.sum()
        g=gtheta.sum(0);g[:3]-=self.precision@beta;g[3]-=2*rho
        return -float(value),-np.r_[g,score[:,:3].ravel()]


def fit_full(data,joint):
    p=FullChannel(data);start=time.perf_counter()
    audited=data.audit
    true_index=2*data.truth[:,0]+data.truth[:,1]
    counts=np.ones((4,4))*.5
    np.add.at(counts,(true_index[audited],p.obs_index[audited]),1)
    c=counts/counts.sum(1)[:,None]
    initial=np.r_[joint['mode'][:4],np.log(c[:,:3]/c[:,3,None]).ravel()]
    starts=[initial,initial.copy()];starts[1][3]=0.
    sols=[minimize(p.value_grad,z,jac=True,method='L-BFGS-B',bounds=[(None,None)]*3+[(-4,4)]+[(-12,12)]*12,
                   options=dict(maxiter=1000,gtol=1e-6,ftol=1e-13,maxls=40)) for z in starts]
    opt=min(sols,key=lambda o:o.fun);h=p.hessian(opt.x);eig=np.linalg.eigvalsh(h)
    grad=float(max(abs(opt.jac)));good=eig.min()>1e-7
    covariance=np.linalg.inv(h) if good else None
    lo=hi=None
    if good:
        sd=np.sqrt(np.diag(covariance))[:4]
        lo=opt.x[:4]-1.95996398454*sd;hi=opt.x[:4]+1.95996398454*sd
        lo[3]=np.tanh(lo[3]);hi[3]=np.tanh(hi[3]);lo=lo.tolist();hi=hi.tolist()
    beta,rho,c,_=p.unpack(opt.x)
    at_bound=abs(opt.x[3])>3.999 or max(abs(opt.x[4:]))>11.999
    return dict(success=bool(good and grad<.005 and not at_bound),estimate=np.r_[beta,rho].tolist(),
                lower=lo,upper=hi,seconds=time.perf_counter()-start,gradient_max=grad,
                hessian_eigenvalues=eig.tolist(),channel=c.tolist(),mode=opt.x.tolist(),
                covariance=covariance.tolist() if good else None,at_numerical_bound=bool(at_bound),
                starts=[dict(value=float(s.fun),gradient=float(max(abs(s.jac)))) for s in sols])


def job(path):
    d=json.loads(Path(path).read_text());assert d['source_hash']==source_hash()
    result=fit_full(simulate(**d['config'],seed=d['seed']),d['results']['joint'])
    out=Path('results/noisy_probit/full_channel')/d['case']/Path(path).name
    write_json(out,dict(case=d['case'],seed=d['seed'],truth=d['truth'],result=result,
                       input_signature=d['signature'],channel_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return result['success']

if __name__=='__main__':
    files=[str(p) for root,case in [('minimal','moderate'),('extension','correlated_errors'),('extension','heterogeneous_errors')]
           for p in sorted((Path('results/noisy_probit')/root/case).glob('seed*.json'))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,ok in enumerate(pool.map(job,files),1):
            if i%20==0:print(i,len(files),ok,flush=True)
