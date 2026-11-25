# Part 2 node-rank AUROC, aligned with Katz / HL-GNN

This replaces the earlier sparse degree-only diagnostic. It uses **the same scoring code and definitions** as Jiatong's [`node-rank-auc-v1` protocol](https://github.com/2ira/sgdiff-repo/blob/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/node-rank-auc/PROTOCOL.zh-CN.md), read from commit `bef967467abd3fe825c08e243b3f7c7b6a2f250b`. No CN, GCN, or SGDiff model was retrained: the analysis reuses the already audited per-pair predictions.

## Definition

For each dataset and seed, compute Degree and PageRank on the **clean `train_pos` graph**. PageRank uses alpha 0.85, uniform teleport/dangling distribution, tolerance `1e-12`, and at most 1000 iterations. Ties use `SHA256("node-rank-v1|dataset|seed|node_id")`, exactly as in the teammate's evaluator. For each integer k=0,...,100, take `ceil(N*k/100)` highest ranked nodes and select original test pairs with **at least one** endpoint among them. The cohort remains fixed for every model and attack. Compute pooled AUROC on the retained positive and negative pairs, without resampling. k=0 is undefined; k=100 reproduces the original full-test AUROC.

The protocol, three datasets (Cora, CiteSeer, PubMed), seeds 0–2, 14 attack conditions, and pinned native evaluator revision `95083da808e500c02da458e205f4b861ceb80871` match the team's Katz / HL-GNN work. We compared **all six split arrays for each of the nine dataset/seed combinations** against Jiatong's published `katz-replay-inputs.zip`; all were exactly equal. The selected-cohort counts also match her released key points (for example, CiteSeer Degree Top 10%: minimum 295 positive and 128 negative pairs). We evaluate seven Part 2 variants: CN; GCN, no-neighbors, no-attributes; SGDiff fusion, structure, feature. The original 882 evaluations yield **178,164 per-seed curve rows** and **59,388 three-seed summary rows**. `aggregation_audit.json` checks every original global AUC. The source scorer's nine `audit.json` files also verify score arrays and k=100.

## Files

- `summary.csv`: mean and sample SD of AUROC for every model, condition, rank type, and k, plus minimum positive/negative counts, support flag, and class coverage.
- `delta_summary.csv`: within-seed attack minus clean AUROC, then mean and sample SD. Negative means degradation.
- `key_points.csv`: k=1,5,10,20,50,100 for a compact table.
- `per_seed_curves.zip`: all nine per-seed curves, paired differences, rankings, and audits.
- `replay_inputs.zip`: the nine original split arrays, 882 original score archives, and portable manifests with ordered-query fingerprints and expected full AUC.
- `clean_node_rank_auc.pdf`, `hub_delete_50_node_rank_auc.pdf`: Degree and PageRank curves for CN, GCN, and SGDiff; `test_pair_coverage.pdf` shows how many positive and negative pairs each percentile retains. PNG versions are included.

## Recompute

Extract `replay_inputs.zip` into a fresh directory. Run `scripts/part2_node_rank_peer_evaluate.py` on each `inputs/manifests/*.json`, using a distinct empty output directory for each dataset/seed. Then run `scripts/part2_peer_rank_summarize.py --results <output-root>` and `scripts/part2_peer_rank_report.py --results <output-root>`. `scripts/part2_peer_rank_manifests.py` rebuilds the manifests from the original local peer run, and `scripts/part2_peer_rank_package.py` builds both archives. These steps use CPU only; training checkpoints and CUDA are not needed.

The axes describe **node importance**, not rank among model-predicted links. Small-k cohorts can have few negative test pairs; dots in the plots mark any seed with fewer than 20 positives or negatives. Differences between Top k% and global AUROC also reflect which candidate pairs entered the subset, so they are not causal effects of changing the model.
