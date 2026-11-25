Superseded by [the fully aligned Degree and PageRank analysis](../part2-node-rank-aligned/README.md).

# Part 2: Top-degree node cohorts

This is a post-hoc analysis of the **saved** Part 2 peer-protocol predictions. No model was retrained. It covers CN, the three GCN configurations, and SGDiff fusion/structure/feature readouts on Cora, CiteSeer, and PubMed, with seeds 0–2 and all 14 conditions.

For each dataset and seed, rank nodes by degree in the **clean training-positive graph** (ties: ascending node ID). Select the first `ceil(k% × number of nodes)` for k = 1, 5, 10, 20, 50, 100. A held-out candidate pair enters the cohort if **either** endpoint is selected. The pair labels and model scores come from the existing peer evaluation; the cohort does not change across models or perturbations. We report AUROC and AP for that cohort, plus positive and negative counts. `summary.csv` is the mean and sample SD over three seeds. `per_seed.csv` retains the individual values. At k = 100%, AUROC exactly reproduces the original result for each model and condition.

The clean and 50% hub-priority deletion curves are in the PDF/PNG files; all conditions are in the CSVs. The top 1% subsets have few negatives in Cora and CiteSeer, so interpret their seed variation cautiously. This is **degree-conditioned AUROC**, not precision among the model's highest-scoring predicted pairs. The latter would require a different cohort definition.

Rebuild from repository root:

```powershell
D:\Tools\Anaconda\anaconda3\python.exe scripts\part2_top_degree.py `
  --repo . --output results\part2-top-degree
```

The report inserts `report_fragment.tex` as `deliverables/part2_report/generated/top_degree.tex` and copies the two PDFs into `deliverables/part2_report/figures`.
