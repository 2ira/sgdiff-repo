# Measured experiment archive

Completed on 2026-11-18: Cora/Citeseer × three author models × three components
per model × seeds 0/1/2 = **54 training configurations and 756 evaluations**.
Protocol: `new-link-models-evasion-v2-common-batch512`.

- [Chinese findings](FINDINGS.zh-CN.md) and [complete tables/curves](report.md).
- `per_seed.csv`, `summary.csv`: AUC/AP, strict global-pool Hits@100,
  separately named precision@100 and clean-to-attack drops.
- `paired_components*.csv`: paired full-minus-variant performance and
  variant-minus-full degradation; positive values have different meanings,
  as defined in the column names and report.
- `strata*.csv`, `score_correlations.csv`: clean-graph candidate strata with
  sample counts, and label-separated descriptive correlations.
- `learned_parameters.csv`, `attention_per_pair.csv`: learned depth/branch
  parameters and sampled attention behavior, documented in `introspection.md`.
- `run_manifest.json`, `example_config.json`, `data_manifest.json`,
  `dataset_statistics.json`, `runtime.json`: all configurations, source/data
  fingerprints, hardware and dependency versions.
- `training_summary.csv`: actual selected/stopped epochs and allocated memory.
  Timing excludes graph/PPR preparation. Parameter counts include registered
  author parameters that can be unused in a particular forward path.
- `audit.json`, `protocol_audit.json`, `replay_audit.json`,
  `mechanism_audit.json`, `ppr_reference_check.json`, `verification.json`:
  completed-grid checks, protected targets and exact budgets, independent
  checkpoint prediction replay and backend/component checks.
- `robustness_auc.png` / `.pdf`: standalone scientific figures;
  `checksums.sha256` fingerprints this curated archive.

Checkpoints, raw datasets, PPR caches, complete prediction arrays and per-epoch
histories remain in the external experiment output directory. They are excluded
from Git. Regeneration commands and the local/ACP job payload are in the
[experiment README](../README.md).

The common batch size retains visible training edges after target masking;
`protocol_revision.json` records the exclusion of preliminary configurations
that used author-specific batch sizes. Only fixed graph caches and splits were
reused; all 54 published model weights were trained from scratch under the final
protocol. These results do not replace original-paper benchmark tables or make
historical SGDiff attack runs directly comparable.
