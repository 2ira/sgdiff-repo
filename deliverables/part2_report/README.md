# Part 2 report build

The report text, tables, and figures are derived from the audited aligned result table. The asset builder rejects incomplete grids, duplicate cells, non-finite metrics, and graph-hash mismatches.

From the repository root:

```powershell
D:\Tools\Anaconda\anaconda3\python.exe scripts\build_part2_report_assets.py `
  --results results\part2-peer-cn-gcn-sgdiff-cora-citeseer\per_seed.csv `
  --datasets cora citeseer `
  --ncnc-results E:\codex\lin-eval\experiments\new_link_models\benchmarks\per_seed.csv `
  --report deliverables\part2_report

tectonic --outdir deliverables\part2_report deliverables\part2_report\main.tex
```

`main.tex` and `references.bib` are the editable sources. The generated TeX fragments, source data extract, figures, and final PDF are retained so that the submitted version can be inspected without rerunning training.
