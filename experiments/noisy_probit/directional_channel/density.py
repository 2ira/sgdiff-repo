"""Gradient-free evaluation of the same numerical log density for MCMC."""
import numpy as np
from scipy.special import log_expit,expit
from ..model import bvn_cdf,SIGNS
from ..lowdim.model import Full


def negative_log_density(post,z):
    beta=z[:3];rho=np.tanh(z[3]);mu=post.data.x@beta
    p=bvn_cdf(mu[:,0,None]*SIGNS[:,0],mu[:,1,None]*SIGNS[:,1],rho*SIGNS[:,0]*SIGNS[:,1])
    c,aux=post.measurement(z);w=c[:,post.obs_index].T*post.allowed;prob=(p*w).sum(1);denom=np.maximum(prob,1e-14)
    if isinstance(post,Full):
        value=np.log(denom).sum()-.5*beta@post.precision@beta
        value+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3])+(post.alpha*aux).sum()
        return -float(value)
    prior=-.5*beta@post.precision@beta
    prior+=np.log(4)+log_expit(2*z[3])+log_expit(-2*z[3])
    a,b=post.error_prior;v=z[4:6];prior+=np.sum(a*log_expit(v)+b*log_expit(-v))
    prior+=post.kappa_shape*np.sum(log_expit(2*z[6:])+log_expit(-2*z[6:]))
    return -float(np.log(denom).sum()+prior)
