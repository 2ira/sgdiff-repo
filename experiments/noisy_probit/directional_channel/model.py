"""Six measurement parameters: shared FP/FN and one kappa per true state."""
import time
import numpy as np
from scipy.special import expit,log_expit
from scipy.optimize import minimize
from ..model import state_probabilities
from ..lowdim.model import LowDim,Full,copula_channel
from ..state_channel.model import fit as baseline_fit,make_posterior as baseline_posterior

GROUP=np.arange(4)


def directional_channel(fp,fn,kappas):
    c,g=copula_channel(fp,fn,np.asarray(kappas)[GROUP,None])
    expanded=g[:,:,2,None]*(GROUP[:,None,None]==np.arange(4)[None,None,:])
    return c,np.concatenate([g[:,:,:2],expanded],axis=2)


class DirectionalChannel(LowDim):
    def __init__(self,data,kappa_shape=1.):
        super().__init__(data,kappa_shape=kappa_shape);self.dim=10

    def native(self,z):return np.r_[z[:3],np.tanh(z[3]),.5*expit(z[4:6]),np.tanh(z[6:10])]

    def measurement(self,z):return directional_channel(*(.5*expit(z[4:6])),np.tanh(z[6:10]))

    def value_grad(self,z):
        native=self.native(z);beta=native[:3];rho=native[3];rates=native[4:6]
        c,dc=self.measurement(z);p,dp=state_probabilities(self.data.x,beta,rho)
        w=c[:,self.obs_index].T*self.allowed;prob=(p*w).sum(1);denom=np.maximum(prob,1e-14)
        gtheta=(dp*w[:,:,None]).sum(1)/denom[:,None];gtheta[:,3]*=1-rho*rho
        gn=(p[:,:,None]*dc[:,self.obs_index,:].transpose(1,0,2)*self.allowed[:,:,None]).sum(1)/denom[:,None]
        gn[:,:2]*=rates*(1-2*rates);gn[:,2:]*=1-native[6:10]**2
        scores=np.column_stack([gtheta,gn]);scores[prob<1e-14]=0;grad=scores.sum(0)
        prior=-.5*beta@self.precision@beta;grad[:3]-=self.precision@beta
        prior+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3]);grad[3]-=2*rho
        a,b=self.error_prior;v=z[4:6]
        prior+=np.sum(a*log_expit(v)+b*log_expit(-v));grad[4:6]+=a-(a+b)*expit(v)
        prior+=self.kappa_shape*np.sum(log_expit(2*z[6:10])+log_expit(-2*z[6:10]))
        grad[6:10]-=2*self.kappa_shape*native[6:10]
        return -float(np.log(denom).sum()+prior),-grad


def make_posterior(data,kind):
    if kind=='six':return DirectionalChannel(data)
    if kind=='six_shrink':return DirectionalChannel(data,kappa_shape=2)
    return baseline_posterior(data,kind)


def fit(data,kind,warm=None):
    if kind not in ['six','six_shrink']:
        return baseline_fit(data,kind,warm)
    if warm is None:raise ValueError('Six fits require observed-data initialization')
    post=make_posterior(data,kind);started=time.perf_counter()
    starts=[np.r_[warm['mode'][:6],np.arctanh(k)] for k in [[0,0,0,0],[.7,-.3,.3,.2],[-.5,.3,-.3,.7]]]
    bounds=[(None,None)]*3+[(-4,4)]+[(-12,12)]*2+[(-6,6)]*4
    sols=[minimize(post.value_grad,z,jac=True,method='L-BFGS-B',bounds=bounds,
                   options=dict(maxiter=1000,gtol=1e-7,ftol=1e-13,maxls=40)) for z in starts]
    opt=min(sols,key=lambda s:s.fun);z=opt.x.copy();refinements=0
    for _ in range(6):
        value,g=post.value_grad(z)
        if max(abs(g))<1e-4:break
        h=post.hessian(z)
        if np.linalg.eigvalsh(h).min()<=1e-8:break
        step=np.linalg.solve(h,g);accepted=False
        for power in range(12):
            candidate=z-step/(2**power)
            inside=all((lo is None or v>lo) and (hi is None or v<hi) for v,(lo,hi) in zip(candidate,bounds))
            if inside and post.value_grad(candidate)[0]<value:
                z=candidate;refinements+=1;accepted=True;break
        if not accepted:break
    value,g=post.value_grad(z);h=post.hessian(z);eig=np.linalg.eigvalsh(h)
    good=bool(np.isfinite(eig).all() and eig.min()>1e-7);cov=np.linalg.inv(h) if good else None
    at_bound=any((lo is not None and v-lo<.001) or (hi is not None and hi-v<.001) for v,(lo,hi) in zip(z,bounds))
    lo=hi=None
    if good:
        sd=np.sqrt(np.diag(cov));lo=post.native(z-1.95996398454*sd).tolist();hi=post.native(z+1.95996398454*sd).tolist()
    return dict(success=bool(good and max(abs(g))<.005 and not at_bound),estimate=post.native(z).tolist(),lower=lo,upper=hi,
                mode=z.tolist(),covariance=cov.tolist() if good else None,channel=post.measurement(z)[0].tolist(),
                gradient_max=float(max(abs(g))),hessian_eigenvalues=eig.tolist(),at_numerical_bound=bool(at_bound),
                seconds=time.perf_counter()-started,objective=value,newton_steps=refinements,
                starts=[dict(value=float(s.fun),gradient_max=float(max(abs(s.jac))),optimizer_success=bool(s.success)) for s in sols])
