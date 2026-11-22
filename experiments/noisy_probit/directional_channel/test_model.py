from dataclasses import replace
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from .model import directional_channel,make_posterior
from .density import negative_log_density
from .run import CASES
from ..state_channel.model import state_channel
from ..lowdim.simulation import simulate
from ..model import STATES


def test_channel_nesting_and_derivatives():
    v=np.array([.06,.22,.75,-.65,.45,.2]);c,g=directional_channel(*v[:2],v[2:])
    assert c.min()>=0 and np.allclose(c.sum(1),1) and np.allclose(g.sum(1),0)
    for t in range(4):
        for j in range(2):assert np.isclose(c[t,STATES[:,j]==1].sum(),v[0] if STATES[t,j]==0 else 1-v[1])
    num=approx_derivative(lambda x:directional_channel(*x[:2],x[2:])[0].ravel(),v,method='3-point')
    assert np.allclose(num,g.reshape(16,6),atol=2e-7,rtol=2e-5)
    old,og=state_channel(.06,.22,[.75,-.25,.2]);new,ng=directional_channel(.06,.22,[.75,-.25,-.25,.2])
    assert np.allclose(old,new) and np.allclose(og[:,:,3],ng[:,:,3]+ng[:,:,4])
    assert not np.allclose(c,c[np.ix_([0,2,1,3],[0,2,1,3])])


def test_posterior_and_gradient_free_density():
    data=simulate(dict(CASES['asymmetric'],n=15),13700,.2);rng=np.random.default_rng(43)
    z=np.r_[.1,.4,1.2,.4,-1.4,-.7,.8,-.5,.4,.2]
    for kind in ['six','six_shrink']:
        post=make_posterior(data,kind);value,g=post.value_grad(z)
        num=approx_derivative(lambda x:post.value_grad(x)[0],z,method='3-point').ravel()
        assert np.allclose(g,num,atol=5e-6,rtol=3e-5)
        truth=data.truth.copy();truth[~data.audit]=1-truth[~data.audit]
        other=make_posterior(replace(data,truth=truth),kind)
        assert value==other.value_grad(z)[0]
    for kind in ['six','six_shrink','state','full','full_shrink']:
        post=make_posterior(data,kind)
        for _ in range(20):
            v=rng.normal(0,.6,post.dim)
            assert negative_log_density(post,v)==post.value_grad(v)[0]


def test_independent_generator_and_fresh_seeds():
    cfg=dict(CASES['asymmetric'],n=600);d=simulate(cfg,13701,.025);e=simulate(cfg,13701,.05)
    assert np.all(~d.audit|e.audit) and np.array_equal(d.observed,e.observed)
    c,_=directional_channel(cfg['fp'],cfg['fn'],cfg['kappa']);ti=d.truth@np.array([2,1]);oi=d.observed@np.array([2,1])
    for t in range(4):
        n=(ti==t).sum();emp=np.bincount(oi[ti==t],minlength=4)/n
        assert np.all(abs(emp-c[t])<6*np.sqrt(c[t]*(1-c[t])/n)+.0005)
    ranges=[set(range(c['seed_start'],c['seed_start']+400)) for c in CASES.values()]
    assert min(min(r) for r in ranges)>930199 and not ranges[0]&ranges[1]
