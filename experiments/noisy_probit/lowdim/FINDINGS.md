# Low-dimensional dependent errors: results and claim boundaries

The Gaussian-copula channel uses three measurement parameters (FP, FN, kappa),
compared with twelve for the unrestricted 4x4 channel. Including beta and rho,
the fitted dimensions are 7 versus 16. It exactly nests independent errors at
kappa=0 and allows either sign of latent error dependence.

## Prospective independent simulation validation

The protocol and core source/configuration hashes were frozen before validation
outcomes were inspected. This was an internal plan, not a public preregistration.
Four new scenarios use 200 independent networks each, disjoint seed ranges,
n=80, new regression/reciprocity/error parameters, and nested 1%,2.5%,5% audits.
There are 800 independent networks, 2,400 network-budget cells, and 8,000 fitted
method results including prespecified prior sensitivities. No validation fit
failed the declared numerical checks. The earlier 200 development networks
are reported separately and are not counted as independent validation.

## Primary common-positive-dependence scenario

| Comparison | Copula RMSE | Full RMSE | Paired RMSE ratio (95% interval) |
|---|---:|---:|---|
| 5% versus 5% | 0.0669 | 0.0779 | 0.8585 [0.7908,0.9264] |
| 2.5% versus 5% | 0.0727 | 0.0779 | 0.9335 [0.8422,1.0274] |

The second comparison uses 79 versus 158 audited dyads: exactly half the audit
count. Its confidence interval includes a ratio of one, so superiority at half
budget is not established. Both comparisons meet the prespecified retention
rule: upper bound<=1.10, lower coverage bound>=.90, and numerical failures<=1%.
Copula coverage is 97.5% in both comparisons, with Wilson interval
[94.28%,98.93%]. Full at 5% has 96% coverage. Nominal coverage remains 95%.

The paired bootstrap resamples whole networks, keeping method and audit-budget
results together. The upper 97.5th percentile is used for each of two primary
claims, allowing a Bonferroni allocation of one-sided alpha=.025. Calibration
is approximate, and the retention margin is a practical tolerance fixed before
validation, not a universal scientific standard.

Prespecified alternative priors leave the primary scenario's results similar.
They do not make the full and low-dimensional priors information-equivalent.
The performance comparison is between complete inferential procedures rather
than a causal isolation of parameter-count reduction.

## Secondary scenarios and counterexample

At 5% audit:

| Scenario | Independent RMSE / coverage | Copula RMSE / coverage | Full RMSE / coverage |
|---|---|---|---|
| Independent errors | .0824 / 95.5% | .0855 / 96.0% | .0969 / 96.5% |
| Negative dependence | .1402 / 76.0% | .0873 / 95.0% | .1011 / 97.0% |
| State-dependent errors | .1862 / 16.5% | .1557 / 45.5% | .0829 / 97.5% |

The third scenario uses different latent measurement correlations in the four
true states. The full channel contains this law; the common-kappa restriction
generally does not. Copula bias in rho is +.1461 at 5%, compared with -.0119 for
Full. The substantial undercoverage occurs despite numerical convergence.
Some regression-coefficient coverages also fail under this misspecification.

The 1% Copula versus 5% Full comparison in the positive scenario has RMSE ratio
.9449 with descriptive interval [.848,1.048], but it is secondary and does not
establish a general minimum audit budget or an adaptive stopping policy.

## What the study supports

Under the tested homogeneous common-dependence mechanism, a three-parameter
channel retains full-channel performance and halves the audit count in a new
scenario. It improves on ignoring positive or negative measurement dependence.
When the common-dependence restriction is false, this efficiency gain need not
persist; the full channel can be substantially more reliable.

The contribution is a concrete dyad model and reproducible cost/accuracy evidence.
Correlated misclassification and internal validation already have substantial
prior literature, notably Tang et al. (2015), DOI 10.1111/rssc.12081. There is no
claim of first separating structural and measurement dependence.

## Remaining limitations

All validation is simulated, within a designed family of conditions. The four
scenarios share one new structural parameter configuration, rather than forming
a factorial study across network sizes, densities and structural correlations. No audited
real network, shared-node effects, fallible gold labels, covariate-dependent
errors, or optimized audit selection are evaluated. Point estimates are mapped
transformed-coordinate joint modes, not posterior means. Laplace intervals are
approximate; four fixed posteriors are checked by multi-chain sampling using
the declared numerical domains. These checks do not establish untruncated
posterior accuracy or model correctness. Both initial Full-channel sampling
runs failed the fixed diagnostics and were extended from 12,000 to 48,000 retained
draws per chain. All four final checks meet max rank-split R-hat<1.01 and min
bulk ESS>400. Initial runs are retained. About 0.67% of retained draws in each
Full-channel check have at least one row logit within one unit of its numerical
bound; this does not bound excluded posterior mass. Numerical-domain sensitivity
beyond these declared bounds has not been established.

A next model could allow a few state-specific dependence parameters. Because
the present stress-test result would inform that design, it would require new
validation data and a new frozen plan rather than recycling this holdout.
