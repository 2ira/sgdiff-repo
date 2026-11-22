# Frozen evaluation plan: low-dimensional dependent measurement errors

Date: 2026-11-21 (America/Indianapolis). Written before viewing validation results.
This is an internal prospective analysis plan, not a publicly registered study.
Development can use the previous n=60 data; the validation seeds below have never
been used in that study. No model or threshold will be selected using validation outcomes.

## Model fixed in advance

For true state t=(Y1,Y2), define error indicators Ek=1[Zk <= Phi^-1(r_tk)] with
r0=FP and r1=FN. (Z1,Z2) is standard bivariate normal with correlation kappa,
independent of the structural network latent errors. Ok=Yk XOR Ek.
The 4x4 channel is a deterministic four-state probability calculation. Kappa=0
exactly nests independent errors. Kappa is NOT the Pearson correlation of binary
errors; rho is the distinct latent reciprocity parameter.

Measurement parameter counts: independent=2, copula=3, unrestricted=12.
Total dimensions, including three beta coefficients and rho: 6,7,16.
Common priors: report-style Gaussian beta prior; rho uniform(-1,1). Independent
and copula FP/FN ~ 0.5 Beta(2,8); kappa uniform(-1,1). Full-channel rows use
Dirichlet(.5,.5,.5,.5). Priors differ and are not claimed to be information-matched.
Jacobian factors are included in transformed-coordinate inference.

## Development and independent validation

Development: legacy independent and shared-uniform correlated errors, n=60,
beta=(1,-.5,2), rho=.8, FP=.05,FN=.2; seeds 202101--202200, q=.05.
These data have already influenced the choice of research direction.

Validation: n=80, beta=(.4,-.8,1.3), rho=.6, FP=.08,FN=.18. Four scenarios:

| Scenario | Error dependence | Seeds (200 networks each) |
|---|---|---|
| positive (primary) | common kappa=.7 | 710000--710199 |
| independent | common kappa=0 | 720000--720199 |
| negative | common kappa=-.5 | 730000--730199 |
| state_dependent | kappa=(.85,-.3,-.3,.3) for true states 00,01,10,11 | 740000--740199 |

Separate random streams generate covariates, true edges, measurement errors, and
audit order. Each network is fit at q=1%,2.5%,5%: 32,79,158 audited dyads.
Audits are nested across budgets and reveal both directions. All methods use the
same observed network and audit set. Different scenarios use disjoint seeds.
There are 800 independent validation networks and 2,400 network-budget cells,
not 2,400 independent networks. State-dependent errors deliberately violate the
common-kappa restriction but are inside the full homogeneous 4x4 family.

## Primary questions and criteria

The primary parameter is rho. Two primary contrasts in the positive scenario:
(A) copula at 5% versus full at 5%; (B) copula at 2.5% versus full at 5%.
We operationalize "retains performance" as an upper confidence bound <=1.10
for the RMSE ratio and a lower Wilson bound >=.90 for copula interval coverage,
with <=1% numerical failures. These are practical tolerances, not universal
scientific standards; 95% remains nominal coverage. A 10% relative RMSE margin
was selected before validation and will also be reported continuously.

Use paired resampling of NETWORKS (10,000 bootstrap replicates, fixed seed),
preserving methods and budgets together. The 97.5th percentile is the upper
one-sided bound for each primary RMSE ratio; each claim uses alpha=.025 to allow
Bonferroni control for two primary claims. Wilson lower bounds likewise use
two-sided 95% limits (one-sided 97.5%). This is approximate Monte Carlo inference,
not a theorem of exact family-wise coverage. No extrapolation to arbitrary budgets.
Coverage counting failed fits as noncoverage is primary; conditional coverage is
also shown. RMSE uses all finite returned point estimates, and missing estimates
invalidate the comparison rather than being silently omitted.

All other scenario/method/budget comparisons are descriptive secondary analyses.
Also report bias, width, MCSE, Wilson intervals, beta errors, estimated kappa,
rho-kappa posterior correlation, numerical failures, and local Hessian diagnostics.
Point estimates are transformed-coordinate posterior modes mapped to native
scales, NOT native MAPs or posterior means. No model selection is performed.

## Numerical checks and fixed diagnostics

Before validation: channel normalization and margins, exact independence nesting,
independent numerical quadrature, analytic gradients, no unaudited-truth leakage,
and independent simulation frequency checks. Three low-dimensional starts
(kappa=0,.7,-.5), two independent/full starts. Common L-BFGS tolerance and a
deterministic bounded Newton polish. Success: Hessian min eigenvalue>1e-7,
max absolute gradient<.005, and no numerical boundary. eta_rho in [-4,4],
eta_kappa in [-6,6]; error/row logits in [-12,12]. All failures retained.
Development-only pilot (three legacy seeds) hit the initial eta_kappa bound of 4
under perfectly shared errors. This bound was expanded to 6 BEFORE validation;
the pilot is retained separately and is not pooled with the final runs.

Four-chain posterior checks: positive and state_dependent scenario, first seed,
q=2.5%, for copula and full. Prewarm 4,000, retain 12,000 draws/chain; adapt during
warmup only. Require rank-split R-hat<1.01 and bulk ESS>400. Failed diagnostics
trigger longer sampling without selecting another dataset; report all attempts.
Sampling is an accuracy diagnostic, not a new estimator in the validation table.

## Prior sensitivity fixed in advance

In the primary positive scenario at q=2.5% and 5% (all 200 networks), add:
- copula_shrink: (kappa+1)/2 ~ Beta(2,2), with FP/FN priors unchanged;
- full_shrink: each row Dirichlet(4*C0_row), where C0 is the independent channel
  with FP=FN=.1 (the prior mean rates of the low-dimensional model).
This is a sensitivity analysis, NOT a claim that these priors are equivalent.
Use unchanged primary criteria; sensitivity results do not select a new winner.

## Claim boundaries and updated prior art

Correlated binary outcomes with misclassification and internal validation are
already studied, including Roy & Banerjee (2009), DOI 10.1002/bimj.200800127,
and Tang et al. (2015), DOI 10.1111/rssc.12081. Tang et al. explicitly allow
correlation in the measurement process as well as in the true response process.
Lin et al. (online 2025; 2026 issue), DOI 10.1080/02664763.2025.2593319, further
study internal validation for correlated outcomes. These complement the previous
network-specific review. We do NOT claim first separation of the two correlations.
The contribution tested here is the dyad-specific parsimonious channel and its
empirical audit-cost tradeoff relative to an unrestricted channel.

Primary sources consulted:
- https://pubmed.ncbi.nlm.nih.gov/19588453/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4440592/ (indexed full-text methods)
- https://pubmed.ncbi.nlm.nih.gov/26005223/
- https://www.tandfonline.com/doi/abs/10.1080/02664763.2025.2593319

No real-data validation, global finite-sample identification guarantee, general
network dependence, or optimal audit design is claimed.
