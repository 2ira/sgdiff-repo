# Frozen plan: a five-parameter state-dependent error channel

2026-11-21. Internal prospective plan, not public preregistration. Written before
new validation data are generated or inspected. Earlier validation data are now
development data and will not be presented as new independent evidence.

## Model and fixed comparisons

Shared FP alpha and FN gamma; three latent Gaussian error correlations for true
states 00, 01/10, and 11. Thus 5 measurement parameters and 9 total parameters,
versus 3/7 for common-kappa and 12/16 for the unrestricted channel. The two
discordant states share dependence: this is a substantive directional-symmetry
restriction. All channels remain homogeneous in covariates.

Native priors: alpha,gamma ~ .5 Beta(2,8); each kappa ~ Uniform(-1,1), independently;
rho ~ Uniform(-1,1); report-style Gaussian beta prior. Full rows use independent
Dirichlet(.5,.5,.5,.5). All coordinate Jacobians are included. Priors are not
information-equivalent. Common L-BFGS and Newton-polish settings are inherited
from the prior frozen implementation. State starts are kappa=(0,0,0),(.7,-.3,.2),
(-.5,.3,.7), with beta/rho/rates initialized by the observed-data common-channel fit.
No fitting initialization uses unaudited true edges.

## Development

Previous failed state-dependent scenario: n=80, beta=(.4,-.8,1.3), rho=.6,
FP=.08,FN=.18, kappas=(.85,-.3,-.3,.3), seeds 740000--740199, audit 5%.
Reuse frozen common/full results and fit only the new state channel.
This check is explicitly exploratory, because these data motivated the model.

## Fresh validation

n=90 (4,005 dyads), beta=(-.2,.6,1.1), rho=.5, FP=.06,FN=.22.
200 independently simulated networks per scenario, disjoint fresh seed ranges:

| Case | Kappa by true state 00,01,10,11 | Seeds |
|---|---|---|
| state (primary) | .75,-.25,-.25,.20 | 910000--910199 |
| common (redundancy control) | .50,.50,.50,.50 | 920000--920199 |
| asymmetric (misspecification) | .75,-.65,.45,.20 | 930000--930199 |

Each network is analyzed at audit fractions 1%,2.5%,5%: 40,100,200 dyads,
revealing 80,200,400 true directed labels. Audits are nested across budgets;
all methods share observed data and audit sets. Separate RNG streams generate
covariates, truth, measurement errors, and audit order. Generation uses normal
draws, not channel likelihood evaluation. Total: 600 independent networks,
1,800 network-budget cells. No tuning or method selection on this validation set.

## Primary outcomes, margins, and inference

Primary parameter: structural reciprocity rho. Two primary contrasts in the
state scenario: State 5% / Full 5%, and State 2.5% / Full 5%.
Retains-performance rule, fixed before results: upper 97.5th percentile of paired
RMSE ratio <=1.10; lower two-sided 95% Wilson coverage bound >=.90 for State;
empirical numerical failures <=1%. Nominal interval coverage is 95%. These are
practical tolerances, not universal standards or a guarantee of exact calibration.
Use 10,000 paired network bootstrap samples with fixed seed 132017, preserving
methods and audit budgets. One-sided alpha=.025 per claim allocates .05 across
the two claims by Bonferroni; bootstrap calibration is approximate. Report all
ratios and bounds even if a criterion fails; no retroactive margin changes.

Also report rho bias, RMSE MCSE, width, coverage with Wilson intervals, beta
metrics, channel parameter estimates, and failure rates. Failed fits count as
noncoverage; all finite point estimates remain in RMSE. Numerical success:
positive Hessian min eigenvalue>1e-7, max gradient<.005, no boundary. Domains:
eta_rho in [-4,4], each eta_kappa in [-6,6], rate/row logits in [-12,12].
Point estimates are transformed joint modes mapped to native coordinates,
not native MAPs or posterior means. All nonprimary comparisons are descriptive.

## Prespecified sensitivity and diagnostics

In the state scenario at 2.5% and 5%, all 200 networks additionally fit:
- State-S: (kappa_g+1)/2 ~ Beta(2,2) independently for the three groups;
- Full-S: Dirichlet(4*C0_row), where C0 has independent FP=FN=.1.
These alternatives do not select a new main method or establish matched priors.

Four-chain posterior checks on the first primary-validation seed (910000),
2.5% audit: State and Full. Warmup 4,000, initially 12,000 retained draws per
chain; adaptive scale during warmup only. If max rank-split R-hat>=1.01 or
min bulk ESS<=400, extend the same dataset/target to 48,000 and, if needed,
96,000 retained draws per chain, retaining previous attempts. No sampling
estimator replaces frozen validation estimates. Check sensitivity to the
numerical domain only if diagnostics show material boundary proximity; do not
claim accuracy for the untruncated posterior merely from these checks.

## Scope and prior art

Random gold-label audits and informative true-state strata separate structural
and measurement probabilities in the population. This is not a finite-sample
precision theorem. Shared-node dependence, covariate-dependent or direction-
dependent marginal error rates, fallible audits, and real networks are omitted.
The asymmetric case deliberately violates only the shared 01/10 dependence
restriction while keeping marginal FP/FN fixed.
Correlated misclassification/internal validation already exists (Tang et al.,
2015, DOI 10.1111/rssc.12081; Roy & Banerjee, 2009, DOI 10.1002/bimj.200800127).
The contribution tested is this parsimonious dyad parameterization and its
repair/cost tradeoff, not a new general misclassification framework.
