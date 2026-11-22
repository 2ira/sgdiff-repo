# Six-parameter directional error channel

This is a **separate reciprocal-probit measurement-error study**, not an SGDiff
training benchmark. It relaxes the five-parameter channel's assumption that
true states 01 and 10 share the same latent error correlation. Marginal FP/FN
rates remain shared. The main scientific outcomes and limitations appear in the
[English report](../../../deliverables/noisy_probit_directional_english/report.pdf)
and [slides](../../../deliverables/noisy_probit_directional_english/slides.pdf).

The [frozen protocol](PROTOCOL.md) defines 800 independent networks, two nested
audit budgets, and two primary retention comparisons. Earlier results motivated
the extension but are not counted as fresh evidence. The same parameter settings
with fresh seeds deliberately test replication of the prior stress mechanism.

## Environment

Python 3.12, NumPy 2.3.1, SciPy 1.17.0. CPU only. From repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r experiments/noisy_probit/directional_channel/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Read or reproduce the published results

- `results/noisy_probit/directional_channel/validation/summary.csv`: all structural
  coefficient outcomes, coverage bounds, Monte Carlo errors, and failure counts.
- `validation/channel.csv`: conditional-channel recovery and descriptive timings.
- `validation/summary.json`: paired primary and descriptive contrasts, nuisance
  estimates, and lock provenance.
- `fit_records.jsonl.gz`: every network-budget fit record, including full joint
  covariance, initialization outcomes, point estimates, intervals, and hashes.
- `mcmc/*/result.json` and `posterior_draws.zip`: all diagnostic attempts and draws.

Restore the compact archives before re-analysis:

```sh
.venv/bin/python -m experiments.noisy_probit.directional_channel.archive extract
.venv/bin/python -m experiments.noisy_probit.directional_channel.analysis
.venv/bin/python -m experiments.noisy_probit.directional_channel.audit
```

The archive extractor rejects unexpected paths, duplicates, and conflicting
existing records. Per-network source/design signatures and regenerated data
hashes permit integrity checks. Raw files are ignored by Git to keep the branch
compact; their lossless archives are tracked.

For a fresh simulation, move `results/noisy_probit/directional_channel/validation`
aside, then run:

```sh
.venv/bin/python -m experiments.noisy_probit.directional_channel.run --workers 12
```

Existing compatible records resume automatically. Changed frozen sources or
configuration are rejected. The default design uses 400 networks per case.
Record counts and analysis unit matter: 1,600 network-budget cells are nested
within 800 independent networks, with 8,000 unique method fits including common
initialization and prior sensitivity. No failures are silently discarded.

Run numerical tests:

```sh
.venv/bin/python -m pytest experiments/noisy_probit/test_model.py experiments/noisy_probit/test_components.py experiments/noisy_probit/lowdim/test_model.py experiments/noisy_probit/lowdim/test_analysis.py experiments/noisy_probit/state_channel/test_model.py experiments/noisy_probit/directional_channel/test_model.py -q
```

## Posterior diagnostics

The fixed input is
`results/noisy_probit/directional_channel/validation/asymmetric/q0.025/seed1010000.json`.
For each of `six` and `full`, run the sampler with 12,000 retained draws and
extend to 48,000, then 96,000, only when the frozen diagnostic rule requires it:

```sh
.venv/bin/python -m experiments.noisy_probit.directional_channel.mcmc --input results/noisy_probit/directional_channel/validation/asymmetric/q0.025/seed1010000.json --kind six --draws 12000 --output results/noisy_probit/directional_channel/mcmc/six_12000
```

Use matching names for each kind/draw count. Earlier attempts remain archived.
For the conditional numerical-domain check, use Full with `--logit-bound 20` and
an output name beginning `domain_full_`. This check does not replace the main
fits or alter the frozen comparisons. `mcmc/summary.json` records the final
outcomes and boundary proximity. `diagnostics.py` regenerates that audit.

## Editable English artifacts

The [artifact directory](../../../deliverables/noisy_probit_directional_english)
contains LaTeX sources, generated tables, and a build script. It uses Beamer and
article with Tectonic 0.17.0. After restoring/analyzing results:

```sh
PYTHON_BIN=.venv/bin/python TECTONIC_BIN=/absolute/path/to/tectonic bash deliverables/noisy_probit_directional_english/build.sh
```

Set `PYTHON_BIN` to an absolute path if launching outside the repository root.
The build script generates tables directly from saved results. PDFs need no TeX
installation to read. Source/data hashes in `MANIFEST.sha256` can be checked with
`sha256sum -c results/noisy_probit/directional_channel/MANIFEST.sha256`.

The supplied third-party group report is not redistributed. The code implements
the framework independently and includes its numerical-model dependencies.
