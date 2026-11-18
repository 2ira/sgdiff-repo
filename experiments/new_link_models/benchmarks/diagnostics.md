# Mechanism diagnostics

Strata use common-neighbor count, minimum endpoint degree and feature cosine from the clean train+validation graph. Each attack is scored on the same candidate membership. Per-stratum AUC/AP require both classes; sample counts are reported. AP depends on stratum prevalence.

`score_correlations.csv` reports clean score Spearman correlations separately for positive and negative candidates, reducing label-mixture confounding. A missing rho indicates a constant attribute/score. These are descriptive associations, not causal feature attributions. Read them alongside retrained component deltas and frozen probes.
