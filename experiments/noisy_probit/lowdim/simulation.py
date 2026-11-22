"""Sampling implementation independent of likelihood/CDF evaluation."""
from dataclasses import replace
import numpy as np
from scipy.special import ndtri
from ..model import Data
from ..experiment import simulate as old_simulate


def simulate(config,seed,audit_fraction):
    if config.get('legacy'):
        data=old_simulate(n=60,beta=(1.,-.5,2.),rho=.8,fp=.05,fn=.2,
                          noise=config['legacy'],seed=seed,audit_fraction=audit_fraction)
        return data
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    rx,ry,re,ra=streams;n=config['n'];m=n*(n-1)//2
    beta=np.asarray(config['beta']);rho=config['rho'];fp=config['fp'];fn=config['fn']
    x=rx.normal(size=(m,2,3));x[:,:,0]=1
    eps=ry.normal(size=(m,2));eps[:,1]=rho*eps[:,0]+np.sqrt(1-rho*rho)*eps[:,1]
    truth=(x@beta+eps>0).astype(int)
    errors=re.normal(size=(m,2))
    kappa=np.asarray(config['kappa'])
    if kappa.ndim:kappa=kappa[2*truth[:,0]+truth[:,1]]
    errors[:,1]=kappa*errors[:,0]+np.sqrt(1-kappa*kappa)*errors[:,1]
    flip=(errors<ndtri(np.where(truth==0,fp,fn))).astype(int)
    observed=np.bitwise_xor(truth,flip)
    audit=np.zeros(m,bool);audit[ra.permutation(m)[:int(round(audit_fraction*m))]]=True
    return Data(x,observed,truth,audit,beta,rho,fp,fn,n)


VALIDATION={
    'positive':dict(n=80,beta=[.4,-.8,1.3],rho=.6,fp=.08,fn=.18,kappa=.7,seed_start=710000),
    'independent':dict(n=80,beta=[.4,-.8,1.3],rho=.6,fp=.08,fn=.18,kappa=0.,seed_start=720000),
    'negative':dict(n=80,beta=[.4,-.8,1.3],rho=.6,fp=.08,fn=.18,kappa=-.5,seed_start=730000),
    'state_dependent':dict(n=80,beta=[.4,-.8,1.3],rho=.6,fp=.08,fn=.18,kappa=[.85,-.3,-.3,.3],seed_start=740000),
}
DEVELOPMENT={
    'legacy_independent':dict(legacy='iid',seed_start=202101),
    'legacy_correlated':dict(legacy='correlated',seed_start=202101),
}
