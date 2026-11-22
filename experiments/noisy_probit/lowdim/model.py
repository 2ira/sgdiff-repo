"""Gaussian-copula error channel, nested independence, and full-channel control.

E_k = 1[Z_k <= Phi^{-1}(r_{Y_k})], O_k = Y_k XOR E_k.
The correlation kappa concerns error-generating normals, NOT observed binary
errors and NOT the structural reciprocity rho. Both can have either sign.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri, ndtr, expit, logit, log_expit, logsumexp
from ..model import Posterior, STATES, bvn_cdf, state_probabilities


def copula_channel(fp,fn,kappa):
    """C[true,observed], derivatives in native (fp,fn,kappa) coordinates."""
    errors=np.bitwise_xor(STATES[:,None,:],STATES[None,:,:])
    signs=2*errors-1
    rates=np.where(STATES==0,fp,fn)
    quantile=ndtri(rates)
    h=signs[:,:,0]*quantile[:,0,None]
    k=signs[:,:,1]*quantile[:,1,None]
    signprod=signs[:,:,0]*signs[:,:,1]
    r=signprod*kappa;scale=np.sqrt(1-r*r)
    c=bvn_cdf(h,k,r)
    phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    dh=phi(h)*ndtr((k-r*h)/scale)
    dk=phi(k)*ndtr((h-r*k)/scale)
    dr=np.exp(-(h*h-2*r*h*k+k*k)/(2*(1-r*r)))/(2*np.pi*scale)
    dfirst=dh*signs[:,:,0]/phi(quantile[:,0,None])
    dsecond=dk*signs[:,:,1]/phi(quantile[:,1,None])
    gradients=[]
    for state in [0,1]:
        gradients.append(dfirst*(STATES[:,0,None]==state)+dsecond*(STATES[:,1,None]==state))
    gradients.append(dr*signprod)
    return c,np.stack(gradients,axis=2)


class LowDim(Posterior):
    def __init__(self,data,dependent=True,kappa_shape=1.,error_prior=(2.,8.)):
        super().__init__(data,error_prior=error_prior)
        self.dependent=dependent;self.dim=7 if dependent else 6
        self.kappa_shape=kappa_shape
        self.obs_index=2*data.observed[:,0]+data.observed[:,1]

    def native(self,z):
        values=np.r_[z[:3],np.tanh(z[3]),.5*expit(z[4:6])]
        return np.r_[values,np.tanh(z[6])] if self.dependent else values

    def measurement(self,z):
        a,b=.5*expit(z[4:6]);kap=np.tanh(z[6]) if self.dependent else 0.
        return copula_channel(a,b,kap)

    def value_grad(self,z):
        native=self.native(z);beta=native[:3];rho=native[3];rates=native[4:6]
        c,dc=self.measurement(z)
        p,dp=state_probabilities(self.data.x,beta,rho)
        w=c[:,self.obs_index].T*self.allowed
        prob=(p*w).sum(1);denom=np.maximum(prob,1e-14)
        gtheta=(dp*w[:,:,None]).sum(1)/denom[:,None]
        gtheta[:,3]*=1-rho*rho
        gn=(p[:,:,None]*dc[:,self.obs_index,:].transpose(1,0,2)*self.allowed[:,:,None]).sum(1)/denom[:,None]
        gn[:,:2]*=rates*(1-2*rates)
        if self.dependent:gn[:,2]*=1-native[6]**2
        else:gn=gn[:,:2]
        scores=np.column_stack([gtheta,gn]);scores[prob<1e-14]=0
        grad=scores.sum(0)
        logprior=-.5*beta@self.precision@beta
        grad[:3]-=self.precision@beta
        logprior+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3]);grad[3]-=2*rho
        a,b=self.error_prior;v=z[4:6]
        logprior+=np.sum(a*log_expit(v)+b*log_expit(-v));grad[4:6]+=a-(a+b)*expit(v)
        if self.dependent:
            # (kappa+1)/2 ~ Beta(shape,shape), including atanh Jacobian.
            logprior+=self.kappa_shape*(log_expit(2*z[6])+log_expit(-2*z[6]))
            grad[6]-=2*self.kappa_shape*native[6]
        return -float(np.log(denom).sum()+logprior),-grad


class Full(Posterior):
    def __init__(self,data,shrink=False):
        super().__init__(data);self.dim=16
        self.obs_index=2*data.observed[:,0]+data.observed[:,1]
        center=copula_channel(.1,.1,0)[0]
        self.alpha=4*center if shrink else np.full((4,4),.5)

    def measurement(self,z):
        logits=np.column_stack([z[4:].reshape(4,3),np.zeros(4)])
        logs=logits-logsumexp(logits,axis=1,keepdims=True)
        return np.exp(logs),logs

    def native(self,z):return np.r_[z[:3],np.tanh(z[3])]

    def value_grad(self,z):
        beta=z[:3];rho=np.tanh(z[3]);c,logs=self.measurement(z)
        p,dp=state_probabilities(self.data.x,beta,rho)
        w=c[:,self.obs_index].T*self.allowed
        prob=(p*w).sum(1);denom=np.maximum(prob,1e-14)
        responsibility=p*w/denom[:,None];responsibility[prob<1e-14]=0
        scores=(dp*w[:,:,None]).sum(1)/denom[:,None];scores[prob<1e-14]=0
        scores[:,3]*=1-rho*rho
        counts=np.column_stack([responsibility[self.obs_index==k].sum(0) for k in range(4)])
        gchannel=counts-counts.sum(1)[:,None]*c+self.alpha-self.alpha.sum(1)[:,None]*c
        value=np.log(denom).sum()-.5*beta@self.precision@beta
        value+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3])+(self.alpha*logs).sum()
        g=scores.sum(0);g[:3]-=self.precision@beta;g[3]-=2*rho
        return -float(value),-np.r_[g,gchannel[:,:3].ravel()]


def make_posterior(data,kind):
    if kind=='independent':return LowDim(data,dependent=False)
    if kind=='copula':return LowDim(data)
    if kind=='copula_shrink':return LowDim(data,kappa_shape=2.)
    if kind=='full':return Full(data)
    if kind=='full_shrink':return Full(data,shrink=True)
    raise ValueError(kind)


def fit(data,kind,warm=None):
    post=make_posterior(data,kind);started=time.perf_counter()
    theta=np.r_[ndtri(np.clip(data.observed.mean(),.02,.98)),0.,0.,np.arctanh(.3)] if warm is None else np.asarray(warm['mode'][:4])
    if isinstance(post,LowDim):
        rates=np.repeat(logit(.2),2) if warm is None else np.asarray(warm['mode'][4:6])
        starts=[np.r_[theta,rates]] if not post.dependent else [np.r_[theta,rates,np.arctanh(k)] for k in [0.,.7,-.5]]
        bounds=[(None,None)]*3+[(-4,4)]+[(-12,12)]*2+([(-6,6)] if post.dependent else [])
        if not post.dependent:
            other=starts[0].copy();other[3]=np.arctanh(.8);starts.append(other)
    else:
        ti=2*data.truth[:,0]+data.truth[:,1];counts=np.full((4,4),.5)
        np.add.at(counts,(ti[data.audit],post.obs_index[data.audit]),1)
        c=counts/counts.sum(1)[:,None]
        first=np.r_[theta,np.log(c[:,:3]/c[:,3,None]).ravel()]
        second=first.copy();second[3]=0.;starts=[first,second]
        bounds=[(None,None)]*3+[(-4,4)]+[(-12,12)]*12
    sols=[minimize(post.value_grad,z,jac=True,method='L-BFGS-B',bounds=bounds,
                   options=dict(maxiter=1000,gtol=1e-7,ftol=1e-13,maxls=40)) for z in starts]
    opt=min(sols,key=lambda s:s.fun);z=opt.x.copy();refinements=0
    # Identical deterministic Newton polish for every model, declared before validation.
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
    good=bool(np.isfinite(eig).all() and eig.min()>1e-7)
    cov=np.linalg.inv(h) if good else None
    at_bound=any((lo is not None and v-lo<.001) or (hi is not None and hi-v<.001) for v,(lo,hi) in zip(z,bounds))
    success=good and max(abs(g))<.005 and not at_bound
    lo=hi=None
    if good:
        sd=np.sqrt(np.diag(cov));lo=post.native(z-1.95996398454*sd).tolist();hi=post.native(z+1.95996398454*sd).tolist()
    return dict(success=bool(success),estimate=post.native(z).tolist(),lower=lo,upper=hi,
                mode=z.tolist(),covariance=cov.tolist() if good else None,channel=post.measurement(z)[0].tolist(),
                gradient_max=float(max(abs(g))),hessian_eigenvalues=eig.tolist(),at_numerical_bound=bool(at_bound),
                seconds=time.perf_counter()-started,objective=value,newton_steps=refinements,
                starts=[dict(value=float(s.fun),gradient_max=float(max(abs(s.jac))),optimizer_success=bool(s.success)) for s in sols])
