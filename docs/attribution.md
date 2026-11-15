# Mechanism attribution: which graph property does each model read?

> **Withdrawn (2026-11-15).** The entire probe chain documented here
> (`run_probe.py`, `run_deep_probe.py`, `run_redesign.py`,
> `run_operating_point.py`, `run_adaptive_attack.py`,
> `run_evidence_calibration.py`, `run_bootstrap_ci.py`, `run_gated_repair.py`,
> `run_splice_gate.py` and `make_probe_report.py`) was executed against the
> old, non-faithful SGDiff implementation (2026-11-13 audit). All artifacts
> (`results/probe_*.md`, `deep_*.md`, `redesign_*.md`, `operating_*.md`,
> `adaptive_*.md`, `calibration_*.md`, `significance_*.md`, `gated_*.md`,
> `splice_*.md`, `probe_report.html`, `results/scores/`) have been deleted.
> None of the old verdicts apply to `sgdiff-paper-v2`.

What remains valid is the tooling itself: `sgdiff/data/perturb.py` and the
probe scripts can be re-run against `sgdiff-paper-v2` checkpoints
(`runs/sgdiff-paper-v2/`) if mechanism attribution for the corrected
implementation is needed:

```bash
python scripts/run_probe.py --dataset cora --profile paper --seeds 0 1 2
```

The fresh reproduction results live in
[`reproduction.md`](reproduction.md) §0 and [`robustness.md`](robustness.md).
