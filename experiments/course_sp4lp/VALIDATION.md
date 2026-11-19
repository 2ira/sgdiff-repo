# Integration validation (2026-11-15)

Base: upstream `a2bcdd734374ef6780f9120344d047b28e187141` (`sgdiff-paper-v2`). Environment and protected-file comparison are saved in `validation/environment_and_scope.json`.

- Contribution tests: **12 passed** (raw scores, query consistency, path masking/symmetry, re-extraction after missingness, matched budgets and frozen evidence).
- Repository suite: **198 passed, 2 skipped, 1 deselected** in180.58s. The deselected case is `tests/test_gcn_ablation_bridge.py::test_shared_matches_reference_outputs_gradients_and_dropout_rng[True-False]`. It also fails on a clean detached checkout of unmodified upstream a2bcdd7 in this same environment. One gradient differs by2.078e-7 at atol1e-7. It is not silently treated as passing, and neither its source nor tolerance is changed. The control failure log is included. The2 skipped cases are upstream conditional skips.
- Shared benchmark: all9 added methods completed synthetic smoke training/scoring with matching split and training-query fingerprints.
- Shared attack harness: all9 methods completed clean/RF/DICE at10% budget. EA uses the same generic graph-scoring dispatch; EA itself was not rerun in this integration check.
- SGDiff interface check: unmodified upstream SGDiff, CN and SP4LP completed the same synthetic smoke benchmark. This does not assess SGDiff reproduction quality.
- Separate observation runner:12 configurations ×5 conditions =60 finite metric rows on fixed synthetic queries (clean, random deletion40%, low-degree deletion40%, attribute masking40%, joint degree/attribute40%).
- Historical archive:1,152 rows; no SGDiff model rows. These are historical observations, not current v2 reruns.
- Protected upstream files: SGDiff model/diffusion/transformer/fusion/layers, trainer, evaluator, factory, baseline trainer, graph splitting/extraction, metrics, config and default configuration are byte-identical to upstream. Only the shared model registry/training dispatch and attack scoring dispatch receive narrow integration hooks.

## Reproduce tests

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -o addopts='' -q \
  --deselect 'tests/test_gcn_ablation_bridge.py::test_shared_matches_reference_outputs_gradients_and_dropout_rng[True-False]'
python -m pytest tests/test_course_integration.py tests/test_observation_robustness.py -o addopts='' -q
```

See `docs/course_sp4lp.md` for installation and smoke commands. Synthetic outputs are interface checks only, not model-ranking evidence. Larger real-data comparisons with SGDiff v2 need a fresh run under one chosen protocol.
