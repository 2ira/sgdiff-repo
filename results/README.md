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

`significance_<dataset>.md` gives every probe damage a **paired bootstrap**
interval over the 1000 scored pairs plus a one-sided p
(`scripts/run_bootstrap_ci.py`) — the small AUC differences stop being claims.

`gated_<dataset>.md` is the evidence-routed repair: two sub-graph models (k = 1
and k = 2) with a per-pair router (`scripts/run_gated_repair.py`). It buys the
k = 2 ceiling and inherits the k = 2 deletion gap, because the gate fires on 93 %
of pairs after a 50 % deletion. Three additions make that verdict testable rather
than anecdotal: a `CN ≤ τ` threshold sweep (`τ = 0, 1, 2`, plus "always k = 2")
shows no operating point beats plain k = 1; paired-bootstrap intervals on every
arm difference turn the −0.067 / −0.064 / −0.051 regrets into intervals that
exclude 0; and a completion-reach count shows that rebuilding the deleted edges
with the CN prior restores evidence for only 3.7-4.2 % of the `CN = 0` pairs it is
meant to rescue.

`splice_<dataset>.md` is the counterfactual gate §7.11 asked for
(`scripts/run_splice_gate.py`): each pair is scored twice by the k = 1 model, once
plain and once with a synthetic shared neighbour spliced into its sub-graph. The
rise `Δ` fires on 97-100 % of pairs and does not rank per-pair damage (AUC
0.45-0.53), so the quantile gates only interpolate between k = 1 and "always
k = 2" — gates on state fire too often, gates on response fire on everyone.

`probe_report.html` is the whole repository rendered for reading — one self-contained
page (styled tables + inline SVG, no CDN, works offline), built by
`scripts/make_probe_report.py`. Section 00 covers the reproduction layer and is
computed from `summary_indomain_*`, `benchmark_*`, `ablation_*`, `robustness_*`
(including `_purify2`), `transfer_matrix_*` and `summary_lowdata_cora`, so those
numbers cannot drift from their JSON. Sections 01-15 render the probe chain from
`probe_*.json`, `deep_*.json`, `redesign_*.json`, `operating_*.json`,
`adaptive_*.json`, `calibration_*.json`, `significance_*.json`, `gated_*.json` and
`splice_*.json`. Section 14 renders the paired bootstrap intervals, the
gated-routing experiment and the splice gate; the conclusion cards are numbered
from whatever artifacts are present, so rebuilding after a new run keeps the
numbering consistent.
