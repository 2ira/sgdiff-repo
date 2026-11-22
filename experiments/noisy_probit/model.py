"""Four-state collapsed dyad likelihood with audited true-edge observations.

Independent dyads and independent directional measurement errors are model
assumptions, not properties of arbitrary networks. All computations use float64.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, owens_t, expit, log_expit

STATES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
SIGNS = 2 * STATES - 1
SQRT2PI = np.sqrt(2 * np.pi)


def bvn_cdf(h, k, rho):
    """Deterministic bivariate CDF via Owen's T, including zero limits.

Tiny cancellation errors in vanishing tails are clipped at probability zero.
The likelihood uses a separately declared 1e-14 probability floor.
"""
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(rho, float))
    scale = np.sqrt(1 - rho * rho)
    hz, kz = np.abs(h) < 1e-12, np.abs(k) < 1e-12
    hs, ks = np.where(hz, 1., h), np.where(kz, 1., k)
    result = .5 * (ndtr(h) + ndtr(k)) - owens_t(h, (k / hs - rho) / scale) - owens_t(k, (h / ks - rho) / scale)
    result -= .25 * (1 - np.sign(h) * np.sign(k))
    result = np.where(hz, .5 * ndtr(k) + owens_t(k, rho / scale), result)
    result = np.where(kz, .5 * ndtr(h) + owens_t(h, rho / scale), result)
    return np.clip(result, 0., 1.)


def state_probabilities(x, beta, rho):
    """Return p[m,4] and derivatives w.r.t. native (beta,rho)."""
    mu = x @ beta
    h = mu[:, 0, None] * SIGNS[:, 0]
    k = mu[:, 1, None] * SIGNS[:, 1]
    r = rho * SIGNS[:, 0] * SIGNS[:, 1]
    p = bvn_cdf(h, k, r)
    scale = np.sqrt(1 - r * r)
    dh = np.exp(-h*h/2) / SQRT2PI * ndtr((k-r*h)/scale)
    dk = np.exp(-k*k/2) / SQRT2PI * ndtr((h-r*k)/scale)
    dr = np.exp(-(h*h-2*r*h*k+k*k)/(2*(1-r*r))) / (2*np.pi*scale)
    db = (dh * SIGNS[:, 0])[:, :, None] * x[:, 0, None, :] + (dk * SIGNS[:, 1])[:, :, None] * x[:, 1, None, :]
    gradient = np.concatenate([db, (dr*SIGNS[:, 0]*SIGNS[:, 1])[:, :, None]], axis=2)
    return p, gradient


def channel(observed, fp, fn):
    """P(observed two bits | each latent state), and native FP/FN derivatives."""
    obs = np.asarray(observed)[:, None, :]
    true = STATES[None, :, :]
    prob1 = np.where(true == 0, fp, 1-fn)
    factors = np.where(obs == 1, prob1, 1-prob1)
    derivative_fp = np.where(true == 0, 2*obs-1, 0.)
    derivative_fn = np.where(true == 1, 1-2*obs, 0.)
    weights = factors[:, :, 0] * factors[:, :, 1]
    d_fp = derivative_fp[:, :, 0]*factors[:, :, 1] + factors[:, :, 0]*derivative_fp[:, :, 1]
    d_fn = derivative_fn[:, :, 0]*factors[:, :, 1] + factors[:, :, 0]*derivative_fn[:, :, 1]
    return weights, np.stack([d_fp, d_fn], axis=2)


@dataclass
class Data:
    x: np.ndarray
    observed: np.ndarray
    truth: np.ndarray
    audit: np.ndarray
    beta: np.ndarray
    rho: float
    fp: float
    fn: float
    n: int


class Posterior:
    """Gaussian beta prior, uniform rho, independent scaled-Beta error priors.

Fit in z=(beta, atanh(rho), logit(2*fp), logit(2*fn)). Jacobians are
included. Error rates are individually below .5; this is an explicit prior
restriction stronger than fp+fn<1. Beta precision matches the report's g-prior.
"""
    def __init__(self, data, rates=None, audit=None, error_prior=(2., 8.)):
        self.data, self.rates, self.error_prior = data, rates, error_prior
        self.p = data.x.shape[-1]
        self.audit = data.audit if audit is None else np.asarray(audit, bool)
        self.dim = self.p + 1 + (2 if rates is None else 0)
        flat = data.x.reshape(-1, self.p)
        self.precision = flat.T @ flat / len(flat)
        self.allowed = np.ones((len(data.x), 4))
        self.allowed[self.audit] = (STATES[None,:,:] == data.truth[self.audit,None,:]).all(axis=2)

    def decode(self, z):
        beta, rho = np.asarray(z[:self.p]), np.tanh(z[self.p])
        rates = .5*expit(z[-2:]) if self.rates is None else np.asarray(self.rates)
        return beta, rho, rates

    def loglik_scores(self, z):
        beta, rho, rates = self.decode(z)
        p, dp = state_probabilities(self.data.x, beta, rho)
        w, dw = channel(self.data.observed, *rates)
        w = w * self.allowed
        prob = (p*w).sum(axis=1)
        # Numerically underflowing events are retained with a declared floor.
        denom = np.maximum(prob, 1e-14)
        d = (dp*w[:,:,None]).sum(axis=1)
        d[:,self.p] *= 1-rho*rho
        if self.rates is None:
            de = (p[:,:,None]*dw*self.allowed[:,:,None]).sum(axis=1)
            de *= rates*(1-2*rates)
            d = np.column_stack([d,de])
        # Gradient of the explicitly floored likelihood.
        d[prob < 1e-14] = 0.
        return np.log(denom), d/denom[:,None]

    def value_grad(self, z):
        beta, rho, rates = self.decode(z)
        logs, scores = self.loglik_scores(z)
        logprior = -.5 * beta @ self.precision @ beta
        grad = np.zeros(self.dim); grad[:self.p] = -self.precision @ beta
        eta = z[self.p]
        logprior += np.log(4.) + log_expit(2*eta) + log_expit(-2*eta)
        grad[self.p] = -2*rho
        if self.rates is None:
            a,b = self.error_prior; v = z[-2:]
            logprior += np.sum(a*log_expit(v)+b*log_expit(-v))
            grad[-2:] = a-(a+b)*expit(v)
        return -float(logs.sum()+logprior), -(scores.sum(axis=0)+grad)

    def hessian(self, z, step=1e-4):
        columns = []
        for j in range(self.dim):
            delta = np.zeros(self.dim); delta[j] = step*max(1.,abs(z[j]))
            columns.append((self.value_grad(z+delta)[1]-self.value_grad(z-delta)[1])/(2*delta[j]))
        h = np.column_stack(columns)
        return (h+h.T)/2
