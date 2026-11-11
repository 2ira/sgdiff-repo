# Results

Experiment outputs land here: `*.json` metric files written by
`sgdiff evaluate/transfer --output`, and `*.log` transcripts. Everything except
this file is git-ignored, because results should be regenerated
(`bash scripts/run_all_experiments.sh`) rather than trusted from the repository.

See `docs/reproduction.md` for the numbers measured on this machine and how
they compare to the paper.

`benchmark_<dataset>.md` is a **tracked** file: it is the data-identical
comparison produced by `scripts/run_benchmark.py` and is committed next to its
JSON (split / sub-graph fingerprints + per-model metrics). Protocol and caveats:
`docs/benchmark.md`.

`ablation_<dataset>.md` is the likelihood audit produced by
`scripts/run_ablation.py` (condition shuffle, DRNL shuffle, single-step and
single-stream scoring, randomized-condition retrain), together with the learned
fusion weights and the per-stream variance decomposition. The `_full` variant
is the same audit on the full training split (`--profile full --no-retrain
--tag full`). Interpretation: `docs/ablation.md`.
