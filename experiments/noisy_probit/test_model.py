import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from .model import bvn_cdf,state_probabilities,channel,Posterior,STATES
from .experiment import simulate,fit,conditional_covariance


def test_cdf_against_independent_quadrature():
    for h,k in [(-3,-2),(-2,1),(.1,-.5),(0,0),(0,1),(2,3),(-5,2)]:
        for r in [-.95,-.5,0,.8,.95]:
            exact=quad(lambda z:np.exp(-z*z/2)/np.sqrt(2*np.pi)*ndtr((k-r*z)/np.sqrt(1-r*r)),
                       -np.inf,h,epsabs=1e-12,epsrel=1e-12)[0]
            assert abs(bvn_cdf(h,k,r)-exact)<2e-10,(h,k,r,exact,bvn_cdf(h,k,r))


def test_probabilities_sum_and_gradients():
    data=simulate(n=12)
    p,g=state_probabilities(data.x,data.beta,data.rho)
    assert np.allclose(p.sum(axis=1),1,atol=1e-12)
    assert np.allclose(g.sum(axis=1),0,atol=1e-12)
    theta=np.r_[data.beta,data.rho]
    for j in range(4):
        d=np.eye(4)[j]*1e-5
        a,_=state_probabilities(data.x,(theta+d)[:3],(theta+d)[3])
        b,_=state_probabilities(data.x,(theta-d)[:3],(theta-d)[3])
        assert np.allclose((a-b)/2e-5,g[:,:,j],atol=1e-8,rtol=1e-5)


def test_channel_is_stochastic_and_identity_without_noise():
    matrix,grad=channel(STATES,.05,.2)
    assert np.allclose(matrix.sum(axis=0),1)
    assert np.allclose(grad.sum(axis=0),0)
    assert np.array_equal(channel(STATES,0,0)[0],np.eye(4))


def test_joint_gradient_and_audited_likelihood():
    data=simulate(n=15,audit_fraction=.25)
    post=Posterior(data)
    z=np.r_[.7,-.3,1.7,np.arctanh(.65),-1.5,-.6]
    value,g=post.value_grad(z)
    for j in range(6):
        d=np.eye(6)[j]*1e-5
        numerical=(post.value_grad(z+d)[0]-post.value_grad(z-d)[0])/2e-5
        assert abs(numerical-g[j])<2e-6
    beta,rho,rates=post.decode(z)
    p,_=state_probabilities(data.x,beta,rho);w,_=channel(data.observed,*rates)
    logs,_=post.loglik_scores(z)
    state=2*data.truth[:,0]+data.truth[:,1]
    expected=(p*w).sum(axis=1)
    expected[data.audit]=(p*w)[np.arange(len(p))[data.audit],state[data.audit]]
    assert np.allclose(np.exp(logs),expected)


def test_unaudited_truth_does_not_affect_inference():
    from dataclasses import replace
    data=simulate(n=15)
    altered=data.truth.copy();altered[~data.audit]=1-altered[~data.audit]
    first,second=Posterior(data),Posterior(replace(data,truth=altered))
    z=np.r_[0.,0.,1.,.5,-1.,-.5]
    assert first.value_grad(z)[0]==second.value_grad(z)[0]
    assert np.array_equal(first.value_grad(z)[1],second.value_grad(z)[1])


def test_conditional_covariance_does_not_change_mean_and_is_smaller():
    result=fit(Posterior(simulate(n=30,audit_fraction=.1,seed=10)))
    assert result['success']
    conditional=conditional_covariance(result)
    assert conditional['estimate']==result['estimate'][:4]
    delta=np.asarray(result['covariance'])[:4,:4]-np.asarray(conditional['covariance'])
    assert np.linalg.eigvalsh(delta).min()>-1e-9
