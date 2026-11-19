# Historical local results, not SGDiff v2 reruns

2 datasets ×3 seeds ×12 non-SGDiff configurations ×16 conditions =1,152 metric rows. The folder date is an archive identifier for the earlier local investigation. SGDiff rows and audit patches are excluded. Exact historical query arrays and manifests are preserved.

These results used the old independent implementation environment and train-only context in every phase; they must not be pooled with current sgdiff-paper-v2 tables. Current upstream changed splitting/preprocessing/metrics, and the shared benchmark uses train+validation context at test. Re-run all compared methods under one protocol for a new table. Original source snapshots were local bde4aac/0cb69ff with protocol fixes; the full original environment/checkpoints remain on the original local checkout. This archive alone is not a complete executable reproduction.

Main metrics are AUROC/AP. Historical precision_at_100 randomly breaks boundary ties in expectation; upstream legacy hit@100 retains input order. Budgets:4,000 total training queries,512 validation,1,024 test; SP4LP width64, up to200 epochs; all perturbation scoring freezes weights.

Conditions: clean; random/low-degree deletion20/40%; RF/DICE10/25/50%; whole-node attribute masking20/40%; joint edge+attribute40%; exploratory no-new-isolates deletion40%. The last control was added after observing initial results. CN and feature similarity directly connect to course work; AA/PA are classic extensions. SP4LP ablations were independently retrained. These are not the SP4LP authors benchmark results.
