# Results

Experiment outputs land here. **Everything on this page under
`sgdiff-paper-v2/` was produced by the corrected implementation
(`METHOD_VERSION = sgdiff-paper-v2`) at the `paper` profile** (Table 4
exactly) on 2026-11-14/15, 8x NVIDIA H20. The old-implementation artifacts
that previously sat at the root of this directory were deleted on 2026-11-15
(2026-11-13 audit: the earlier implementation was not faithful; its records
were not paper-faithful evidence).

## Layout

```
sgdiff-paper-v2/
  {dataset}_seed{K}_paper.json     in-domain grid (5 datasets x 10 seeds)
  transfer_{src}_seed{K}_paper.json  RQ1 zero-shot transfer (5 sources x 10 seeds)
  lowdata_cora_seed{K}_paper.json   RQ2 1% training data (10 seeds)
  attack_cora_seed{K}_paper.json    RQ3 SGDiff under RF/DICE/EA (3 seeds)
  robustness_cora__v2_seed{K}.{json,md}  RQ3 all 7 models under attack (3 seeds)
  baselines_cora.json               GCN/GAT/SAGE/VGAE/SEAL on the same split
  summary_indomain_paper.json       merged in-domain summary (per-invocation)
  logs/                             per-run transcripts
```

Checkpoints and per-run metrics: `runs/sgdiff-paper-v2/` (the `_bak` cora
directories are the in-domain originals, kept because RQ2/RQ3 runs use their
own run names).

Every JSON embeds a `protocol` block (loss modes, DRNL details, feature
handling, sub-graph marginals, schedule status) and `method_version`, so any
number can be traced to its exact protocol. The detailed tables and
interpretation live in `docs/reproduction.md` §0, `docs/robustness.md`,
`docs/benchmark.md` and `docs/findings.md`.

To regenerate: see `docs/reproduction.md` §3 (single commands) or
`scripts/repro_queue.sh` / `scripts/rq_queue.sh` (the parallel dispatchers
used for the 2026-11-14/15 grid).
