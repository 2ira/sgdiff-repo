import numpy as np
from dataclasses import replace
from scipy.optimize._numdiff import approx_derivative
from .model import StateChannel,state_channel
from .run import CASES
from ..lowdim.model import copula_channel
from ..lowdim.simulation import simulate
from ..model import STATES


def test_channel_nesting_margins_symmetry_and_gradients():
    v=np.array([.06,.22,.75,-.25,.2]);c,g=state_channel(*v[:2],v[2:])
    assert np.all(c>=0) and np.allclose(c.sum(1),1)
    assert np.allclose(g.sum(1),0)
    for t in range(4):
        for j in range(2):assert np.isclose(c[t,STATES[:,j]==1].sum(),v[0] if STATES[t,j]==0 else 1-v[1])
    swap=[0,2,1,3];assert np.allclose(c,c[np.ix_(swap,swap)])
    num=approx_derivative(lambda x:state_channel(*x[:2],x[2:])[0].ravel(),v,method='3-point')
    assert np.allclose(g.reshape(16,5),num,atol=2e-7,rtol=2e-5)
    for k in [-.7,0,.8]:
        cm,gm=copula_channel(.06,.22,k);cs,gs=state_channel(.06,.22,[k]*3)
        assert np.allclose(cm,cs);assert np.allclose(gm[:,:,2],gs[:,:,2:].sum(2))


def test_posterior_gradients_and_no_truth_leakage():
    data=simulate(dict(CASES['state'],n=15),10123,.2)
    z=np.r_[.1,.4,1.2,.4,-1.4,-.7,.8,-.3,.2]
    for shape in [1,2]:
        p=StateChannel(data,kappa_shape=shape);value,g=p.value_grad(z)
        num=approx_derivative(lambda x:p.value_grad(x)[0],z,method='3-point').ravel()
        assert np.allclose(g,num,atol=4e-6,rtol=3e-5)
        truth=data.truth.copy();truth[~data.audit]=1-truth[~data.audit]
        other=StateChannel(replace(data,truth=truth),kappa_shape=shape)
        assert value==other.value_grad(z)[0];assert np.array_equal(g,other.value_grad(z)[1])


def test_sampling_and_fresh_seed_ranges():
    cfg=dict(CASES['state'],n=600);d=simulate(cfg,10124,.01);more=simulate(cfg,10124,.05)
    assert np.array_equal(d.observed,more.observed);assert np.all(~d.audit|more.audit)
    c,_=state_channel(cfg['fp'],cfg['fn'],np.array(cfg['kappa'])[[0,1,3]])
    ti=2*d.truth[:,0]+d.truth[:,1];oi=2*d.observed[:,0]+d.observed[:,1]
    for t in range(4):
        n=(ti==t).sum();emp=np.bincount(oi[ti==t],minlength=4)/n
        assert np.all(abs(emp-c[t])<6*np.sqrt(c[t]*(1-c[t])/n)+.0005)
    ranges=[set(range(c['seed_start'],c['seed_start']+200)) for c in CASES.values()]
    assert all(min(r)>740199 for r in ranges)
    assert not any(ranges[i]&ranges[j] for i in range(3) for j in range(i))
