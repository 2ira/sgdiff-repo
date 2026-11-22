from dataclasses import replace
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
from ..model import channel,STATES,Posterior
from ..full_channel import FullChannel
from .model import copula_channel,LowDim,Full
from .simulation import simulate,VALIDATION


def test_channel_normalization_margins_independence_and_gradient():
    for fp,fn,kappa in [(.05,.2,0),(.08,.18,.7),(.2,.3,-.5),(.01,.1,.95)]:
        c,g=copula_channel(fp,fn,kappa)
        assert np.all(c>=0);assert np.allclose(c.sum(1),1,atol=2e-12)
        assert np.allclose(g.sum(1),0,atol=2e-10)
        for t in range(4):
            for j in range(2):
                expected=fp if STATES[t,j]==0 else 1-fn
                assert abs(c[t,STATES[:,j]==1].sum()-expected)<1e-12
        if kappa==0:assert np.allclose(c,channel(STATES,fp,fn)[0].T,atol=1e-12)
        numerical=approx_derivative(lambda x:copula_channel(*x)[0].ravel(),np.array([fp,fn,kappa]),method='3-point')
        assert np.allclose(g.reshape(16,3),numerical,atol=2e-7,rtol=2e-5)


def test_copula_cell_against_quadrature():
    fp,fn,kappa=.08,.18,.7;c,_=copula_channel(fp,fn,kappa)
    q1,q2=ndtri([fp,fn])
    reference=quad(lambda z:np.exp(-z*z/2)/np.sqrt(2*np.pi)*ndtr((q2-kappa*z)/np.sqrt(1-kappa*kappa)),
                   -np.inf,q1,epsabs=1e-12)[0]
    # true 01 and observed 10 imply two errors.
    assert abs(c[1,2]-reference)<1e-10


def test_posterior_gradient_nesting_and_truth_isolation():
    data=simulate(dict(VALIDATION['positive'],n=15),901,.2)
    for post,z in [(LowDim(data),np.r_[.5,-.3,1.1,.6,-1.3,-.5,.8]),
                   (LowDim(data,kappa_shape=2),np.r_[.5,-.3,1.1,.6,-1.3,-.5,-.8]),
                   (Full(data),np.r_[.5,-.3,1.1,.6,np.arange(12)*.1]),
                   (Full(data,shrink=True),np.r_[.5,-.3,1.1,.6,np.arange(12)*.1])]:
        value,g=post.value_grad(z)
        numerical=approx_derivative(lambda x:post.value_grad(x)[0],z,method='3-point').ravel()
        assert np.allclose(g,numerical,atol=3e-6,rtol=3e-5)
        original=data.truth.copy();data.truth[~data.audit]=1-data.truth[~data.audit]
        new=type(post)(data);new.__dict__.update({k:v for k,v in post.__dict__.items() if k not in ['allowed','data']})
        assert np.array_equal(new.value_grad(z)[1],g);assert new.value_grad(z)[0]==value
        data.truth[:]=original
    z=np.r_[.5,-.3,1.1,.6,-1.3,-.5]
    p=LowDim(data,dependent=False);old=Posterior(data)
    assert np.allclose(p.value_grad(z)[0],old.value_grad(z)[0])
    assert np.allclose(p.value_grad(z)[1],old.value_grad(z)[1])
    full=Full(data);oldfull=FullChannel(data);z=np.r_[z[:4],np.arange(12)*.1]
    assert np.allclose(full.value_grad(z)[0],oldfull.value_grad(z)[0])
    assert np.allclose(full.value_grad(z)[1],oldfull.value_grad(z)[1])


def test_nested_audit_budgets_and_sampling_matches_channel():
    cfg=dict(VALIDATION['positive'],n=650)
    data=simulate(cfg,904,.01);more=simulate(cfg,904,.05)
    assert np.array_equal(data.observed,more.observed)
    assert np.all(~data.audit|more.audit)
    c,_=copula_channel(cfg['fp'],cfg['fn'],cfg['kappa'])
    true=2*data.truth[:,0]+data.truth[:,1];obs=2*data.observed[:,0]+data.observed[:,1]
    for t in range(4):
        n=(true==t).sum();emp=np.bincount(obs[true==t],minlength=4)/n
        assert np.all(abs(emp-c[t])<6*np.sqrt(c[t]*(1-c[t])/n)+.0005)
