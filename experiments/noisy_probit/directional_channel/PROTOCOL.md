# Prospective plan: six-parameter directional state channel

Date: 2026-11-22 (America/Indiana/Indianapolis). Internal prospective plan, not
public preregistration. Freeze source and design hashes before fresh simulation.
The earlier five-parameter study is development context only.

## Question and model

Allow kappa_00, kappa_01, kappa_10, kappa_11 to differ, retaining common FP alpha
and FN gamma. Six channel parameters / ten total; five-state-group baseline:
five / nine; unrestricted Full: twelve / sixteen. Equal discordant kappas nest
the previous five-parameter channel. Fixed margins plus four binary coupling
probabilities give the six-dimensional family. No direction-dependent marginal
error rates, covariate-dependent channels, shared-node dependence, or fallible
audits are included.

Priors, parameterization, Jacobians, Laplace intervals, optimizer tolerances,
and numerical success criteria match the previous frozen study. Each kappa is
independently uniform (-1,1); alpha,gamma independently .5 Beta(2,8); rho uniform;
beta has the existing normalized-design Gaussian prior. Full row priors are
Dirichlet(.5,.5,.5,.5). Priors are not information-equivalent.
Six starts use common-channel beta/rho/rate estimates plus kappas
(0,0,0,0), (.7,-.3,.3,.2), and (-.5,.3,-.3,.7). No initialization uses
unaudited truth. Each method includes the same common-fit initialization cost.
All old sources and results stay unchanged.

## Fresh design

400 independent networks per case (800 total), n=90 / 4,005 dyads,
beta=(-.2,.6,1.1), rho=.5, FP=.06, FN=.22. Reusing known parameter settings is
intentional independent replication of the specific prior stress test, not a
claim of a new data-generating family.

- asymmetric (primary): kappas=(.75,-.65,.45,.20), seeds 1010000--1010399.
- symmetric (redundancy control): kappas=(.75,-.25,-.25,.20),
  seeds 1020000--1020399.

Audit fractions .025 and .05 yield 100 and 200 audited dyads / 200 and 400 gold
directed labels. Audits are nested; methods share observations and audits.
Four RNG streams separate covariates, structural truth, measurement draws,
and audit order. Generation uses normal draws independently of the fitted CDF.
Fit Common as initializer, then Five, Six, Full in every network-budget cell.
1,600 network-budget cells are not independent replicates.

## Frozen comparisons

Primary estimand rho. Two primary retention claims, both in asymmetric truth:
Six(200)/Full(200), Six(100)/Full(200). Same margins as before:
paired-bootstrap RMSE-ratio upper 97.5th percentile <=1.10;
Six lower two-sided 95% Wilson coverage bound >=.90;
empirical numerical failure rate <=1%.
Use 10,000 paired network bootstrap draws, seed 132017. Two one-sided .025
allocations give an approximate Bonferroni allocation across the two RMSE
claims. Coverage nominally 95%; the .90 lower-bound gate is a practical tolerance.
Failed fits count as noncoverage; finite estimates remain in RMSE.
Report all primary outcomes irrespective of success. No margin or estimator
changes after inspection. All other comparisons are descriptive, including
Six/Five at equal budgets, symmetric controls, and nuisance-channel recovery.
Report rho bias, RMSE/MCSE, width, Wilson coverage, beta metrics, failure counts,
median runtime including initialization, and channel-entry RMSE
sqrt(mean over networks and 16 channel cells of (C_hat-C_true)^2).

## Sensitivity and diagnostics

Every primary-case network at both budgets additionally fits Six-S, with
(kappa_t+1)/2 ~ Beta(2,2), and Full-S, with Dirichlet(4*C0_row), where C0 uses
independent FP=FN=.1. These do not select a replacement primary estimator.
Total unique fits: 6,400 main/initializer fits plus 1,600 sensitivities = 8,000.

Fixed diagnostic dataset: first primary seed 1010000, audit .025. Four-chain
Metropolis checks for Six and Full, 4,000 warmup / initially 12,000 retained
draws. Extend to 48,000 then 96,000 if max rank-split R-hat >=1.01 or min bulk
ESS <=400. Keep all attempts, and never replace the validation estimator.
Use a gradient-free evaluation of exactly the same log density; verify equality
against the analytic-gradient implementation before validation. Default numerical
domains match fitting: rho atanh +/-4, kappa atanh +/-6, rate/row logits +/-12.
If retained draws approach bounds, report proximity and assess enlarged domains
where material. No claims about an untruncated posterior follow automatically.

## Publication

Publish only this synthetic-data research extension and its dependencies to
`yichao_dev`, based on current remote main. Include protocol, all fit records
(compressed), machine-readable summaries, posterior diagnostics, editable
English slides/report, and reproduction instructions. Exclude the supplied
third-party PDF, unrelated workspace experiments, local credentials, caches,
and machine-specific absolute paths from published result metadata.
