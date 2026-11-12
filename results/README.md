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

`robustness_<dataset>.md` is the RF / DICE / EA evasion table produced by
`scripts/run_robustness.py`. Tagged variants: `_robust` (five-model comparison with the RGCN baselines), `_rgcn1000`
(RGCN at a 1000-epoch budget) and `_purify2` (the 2025-2026 purifiers GDDM /
GPR-GAE, paired defended vs undefended cells; this is the current reference). Interpretation: `docs/robustness.md` §5.

`ablation_<dataset>.md` is the likelihood audit produced by
`scripts/run_ablation.py` (condition shuffle, DRNL shuffle, single-step and
single-stream scoring, randomized-condition retrain), together with the learned
fusion weights and the per-stream variance decomposition. The `_full` variant
is the same audit on the full training split (`--profile full --no-retrain
--tag full`). Interpretation: `docs/ablation.md`.

`probe_<dataset>.md` is the mechanism attribution produced by
`scripts/run_probe.py`: targeted deletion / insertion / feature-shuffle probes,
the property heuristics scored on the same perturbed graphs, the component
ablations, and what the sub-graph extractor could see. Interpretation:
`docs/attribution.md`.

`deep_<dataset>.md` is the layer/operator/edge-kind + repair audit produced by
`scripts/run_deep_probe.py` (layer-wise linear probes and representation drift,
propagation-operator substitution under the trained weights, triadic-vs-bridging
deletion, CN/AA edge completion, feature smoothing, CN-prior blending, and the
k = 2 sub-graph retrain). Interpretation: `docs/attribution.md` §7.

`redesign_<dataset>.md` is the component-level redesign grid produced by
`scripts/run_redesign.py`: depth (1/2/3 hops), normalisation (sym / row),
self-loops (on / off), input augmentation (raw / structure-smoothed / +2-hop
structural descriptors) and robust training (30 % edge dropout), each retrained
from scratch and re-probed with the same suite. Interpretation:
`docs/attribution.md` §7.6.

`operating_<dataset>.md` answers *where* the damage lives
(`scripts/run_operating_point.py`, fed by the per-pair dumps of
`scripts/run_score_dump.py` in `results/scores/`): AP / hit@100 / TPR@1%FPR next
to AUC, class-wise score drops, Kendall shift, and clean-defined strata
(CN bucket, degree tertile, and the pairs whose evidence was deleted).

`adaptive_<dataset>.md` replaces the fixed probes with a greedy score-feedback
edge deletion at a matched budget (`scripts/run_adaptive_attack.py`), which is the
lower-bound check on every "the model is fragile to X" claim.

`calibration_<dataset>.md` tests whether the scoring rule can be fixed without
retraining (`scripts/run_evidence_calibration.py`): four corrections — global
null-model residual, parametric null subtraction, an evidence gate
(`CN = 0`) and stratified percentile ranking — each with the clean-accuracy cost
it pays. Interpretation: `docs/attribution.md` §7.9 (all four fail, and why).

`probe_report.html` is the same content rendered for reading — one self-contained
page (styled tables + inline SVG, no CDN, works offline), built by
`scripts/make_probe_report.py` from the `probe_*.json`, `deep_*.json` and
`redesign_*.json` files.
