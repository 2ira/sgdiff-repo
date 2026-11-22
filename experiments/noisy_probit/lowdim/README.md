# Parsimonious dependent measurement-error channels

The study compares independent (2 measurement parameters), Gaussian-copula (3),
and unrestricted row-stochastic (12) channels in reciprocal probit networks.
The total fitted dimensions are 6, 7, and 16. Existing results and Chinese slides
from the preceding study are preserved.

Read `PROTOCOL.md` for the prospective internal analysis plan and `THEORY.md`
for model restrictions and the population identification argument. The study
does not claim the first model separating response and measurement dependence.

## Reproduction

Run from the repository root. This study used Python 3.12.15, NumPy 2.3.1,
SciPy 1.17.0 and CPU float64. No GPU or external data are required.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest experiments/noisy_probit/test_model.py experiments/noisy_probit/test_components.py experiments/noisy_probit/lowdim/test_model.py experiments/noisy_probit/lowdim/test_analysis.py -q
.venv/bin/python -m experiments.noisy_probit.lowdim.run --stage development --workers 4
.venv/bin/python -m experiments.noisy_probit.lowdim.run --stage validation --workers 12
.venv/bin/python -m experiments.noisy_probit.lowdim.analysis --root results/noisy_probit/lowdim/development
.venv/bin/python -m experiments.noisy_probit.lowdim.analysis --root results/noisy_probit/lowdim/validation
```

The runner checks an immutable `LOCK.json` and all saved record signatures.
Worker count may change on resumption; models, priors, configurations, seeds,
budgets and fitting code may not. Each record is written atomically.
Validation initially used 6 workers and resumed with 12 using the same lock.
This scheduling change does not alter any estimator or random stream.

Development uses 100 previously used networks in each of two legacy settings.
Validation uses 800 independent new networks (four disjoint 200-seed ranges),
each analyzed at three nested audit budgets: 2,400 network-budget cells.
There are 7,200 main validation fits plus 800 prespecified prior-sensitivity fits.
The 200 development records contain another 600 fits.

## Fixed posterior checks

Run both `copula` and `full` for the positive seed 710000 and state-dependent
seed 740000, at 2.5% audit. For example:

```bash
.venv/bin/python -m experiments.noisy_probit.lowdim.mcmc --input results/noisy_probit/lowdim/validation/positive/q0.025/seed710000.json --kind copula --output results/noisy_probit/lowdim/mcmc/positive_copula
.venv/bin/python -m experiments.noisy_probit.lowdim.mcmc --input results/noisy_probit/lowdim/validation/positive/q0.025/seed710000.json --kind full --output results/noisy_probit/lowdim/mcmc/positive_full
```

Default: 4 chains, 4,000 warmup and 12,000 retained draws per chain. If diagnostics
fail, increase retained draws with `--draws 48000` (or longer) and use an
`_extended` output suffix. Keep every initial attempt. A longer run uses the
same random stream and includes the original prefix; attempts are not pooled
as if independent. The export chooses the longest run of each fixed dataset,
even if its diagnostics fail, and reports the status explicitly.

Numerical domains are shared with optimization. Diagnostics establish neither
untruncated-posterior accuracy nor correctness of a misspecified channel.
No sampled estimator replaces the frozen validation-table point estimates.

## English slides

After all four posterior checks and any required extensions:

```bash
.venv/bin/python -m experiments.noisy_probit.lowdim.export
bash deliverables/noisy_probit_lowdim_english/build.sh
```

The deck is editable Beamer/TeX. All plots and tables are generated from verified
outputs. `TECTONIC_BIN=/path/to/tectonic` overrides the bundled compiler location;
a complete XeLaTeX/Beamer/pgfplots installation can also compile the source.

## Interpretation and provenance

- Primary tests use paired network bootstrap intervals, a fixed 10% RMSE margin,
  and a 90% lower coverage-bound tolerance; 95% remains nominal coverage.
- Failed fits count as noncoverage. All finite returned estimates contribute
  to RMSE; failures are reported rather than dropped.
- All estimates in validation tables are transformed joint modes mapped to
  native scales. MCMC means are separate approximation diagnostics.
- The full and low-dimensional priors are not equivalent; sensitivity checks
  do not eliminate this distinction.
- State-dependent errors were specified before validation and intentionally
  violate the common-kappa restriction.
- `smoke/` and `smoke_final/` are development-only pilot records. The first
  pilot exposed an overly restrictive kappa numerical bound; it was fixed
  before the final source/configuration lock, without inspecting holdout data.
- Validation results must not be reused to select a new channel and then be
  described as untouched validation for that channel.

No commits, external publication, or modifications to the user's original PDF
were made as part of this research task.
