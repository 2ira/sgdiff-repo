# Model, identification argument, and limits

## Separate structural and measurement dependence

For an unordered dyad d, let structural latent errors have correlation rho and
unit marginal variances. With mu_k=x_k' beta,
p_t(beta,rho)=Phi_2(s_1 mu_1,s_2 mu_2;s_1 s_2 rho), s_k=2t_k-1.

Let r_0=alpha (false positive) and r_1=gamma (false negative). Independently of
the structural errors, draw standard Gaussian measurement errors (Z_1,Z_2)
with correlation kappa. Define E_k=1[Z_k <= Phi^-1(r_{Y_k})] and O_k=Y_k XOR E_k.
For e=t XOR o and a_k=2e_k-1, the channel is

C_{t,o}=Phi_2(a_1 Phi^-1(r_{t_1}), a_2 Phi^-1(r_{t_2}); a_1 a_2 kappa).

This gives nonnegative row probabilities summing to one. The marginal error
rates are exactly alpha and gamma regardless of kappa. At kappa=0 it factorizes
into the independent error channel. Kappa is a latent Gaussian correlation;
the binary error correlation depends additionally on the row's marginal rates.
The channel assumes equal directional rates, a common dependence parameter
across true states, and no covariate dependence. The unrestricted 4x4 channel
does not impose these restrictions (but still assumes homogeneity in x).

An unaudited dyad contributes sum_t p_t C_{t,o}; an audited dyad with true t
contributes p_t C_{t,o}. An entire dyad is audited, including both directions.
The audit indicator is independent of truth and measurement conditional on
the fixed random audit design. No factor involving latent truth is used for
unaudited observations.

## A population identification argument under random auditing

This is a direct argument for the stated parametric model, not a new general
identification theorem or a finite-sample guarantee.

Assume (i) an audit probability q>0 independent of (Y,O,X); (ii) the gold labels
are correct; (iii) standard probit covariate identification conditions (the
relevant design support spans the coefficient space, and latent variances are
fixed at one); (iv) finite indices, interior error rates and correlations; and
(v) the homogeneous channel model is correct.

1. Random auditing identifies the joint population law of (Y,O,X), since its
   conditional law among audited dyads equals the population law.
2. The audited true-response margins Phi(x_k' beta) identify beta under the
   covariate-support assumption. Given beta, p_11=Phi_2(mu_1,mu_2;rho) is strictly
   increasing in rho because its derivative is the positive bivariate normal
   density. Thus the audited true dyad distribution identifies rho.
3. The audited error margins identify alpha and gamma. For example, in true
   state 00, the probability of two errors is Phi_2(Phi^-1(alpha),Phi^-1(alpha);kappa).
   This is strictly increasing in kappa for interior rates and correlations.
   With a positive-probability 00 stratum, kappa is therefore identified.

Other nondegenerate true-state strata can also identify kappa. This argument
does NOT imply that 32 or 79 audited dyads suffice in practice, or that rho and
kappa are well separated in a finite posterior. A vanishing audit fraction
requires additional asymptotic conditions, including sufficiently many audited
observations in informative true-state strata. No such rate theorem is derived.
Without audits, parametric identification may or may not hold depending on
covariate support and restrictions; we do not assert universal nonidentifiability.

## Why parsimony can lower variance and also cause bias

The unrestricted row-stochastic channel has 12 degrees of freedom. The copula
channel has 3. Pooling error rates and dependence across true states reduces the
number of quantities that scarce audited dyads must estimate. It also links
information from the unaudited observations. Those restrictions can be wrong.
The state-dependent validation scenario varies kappa by true state while keeping
the marginal rates fixed; no common-kappa channel can generally represent it.

The low-dimensional and full models have different prior geometries. Therefore
an empirical improvement is evidence about the complete inferential procedure,
not an isolated causal effect of parameter count. The prespecified alternative
priors check sensitivity but cannot establish information-equivalent priors.

## Approximation and estimation conventions

Optimization and Gaussian approximation are performed in beta, atanh(rho),
logit(2 alpha), logit(2 gamma), atanh(kappa), or row-logit coordinates. All native
prior Jacobians are included. A reported point estimate is the transformed joint
mode mapped to the native scale; it is neither a native MAP nor a posterior mean.
Intervals are transformed-coordinate Laplace intervals with full nuisance
covariance. Multi-chain checks use the same posterior and numerical bounds.
Sampling convergence does not establish model correctness or frequentist coverage.

## Prior-art boundary

Tang et al. (2015), DOI 10.1111/rssc.12081, already distinguish structural
response dependence and dependence in the misclassification mechanism using
internal validation. Roy and Banerjee (2009), DOI 10.1002/bimj.200800127, study
misclassified correlated binary outcomes through multivariate probit. Our
Gaussian-copula construction uses established probability tools. The contribution
evaluated here is a specific parsimonious dyad error channel and an independently
simulated audit-cost comparison, not a first general correlated-error model.
