# Likelihood ablations: is the score really conditioned?

> **Withdrawn (2026-11-15).** All ablation results on this page were produced
> by the old, non-faithful implementation (2026-11-13 audit: incorrect
> categorical diffusion/posteriors, a non-Eq. (11) feature denoiser,
> independent noise between classes, non-paper training schedule). The
> artifacts (`results/ablation_cora*.md`) have been deleted. None of the old
> verdicts apply to `sgdiff-paper-v2`.

`scripts/run_ablation.py` implements the check that GCCM (arXiv:2605.05689,
2026) argues every conditional-diffusion predictor needs: **does the reported
accuracy come from the conditional likelihood, or from machinery around it?**

Everything except the ablated quantity is held fixed: same checkpoint, same
evaluation pairs as `scripts/run_benchmark.py`, same fusion weights, same
seeds. The module is `sgdiff/engine/ablation.py`. The audit grid: condition
shuffle, DRNL shuffle, single-step and single-stream scoring, and
randomized-condition retraining, plus the learned fusion weights and the
per-stream variance decomposition.

**Status: not yet rerun under `sgdiff-paper-v2`.** To regenerate against the
corrected implementation (checkpoints live in `runs/sgdiff-paper-v2/`):

```bash
python scripts/run_ablation.py --dataset cora --profile paper \
    --config configs/sgdiff_cora.yaml
```
