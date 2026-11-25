"""Paired hub-conditioned link prediction metrics from saved peer predictions.

Ranks nodes by degree in the clean training graph (never by attacked graph or
test labels). A test pair belongs to a top-k% cohort if either endpoint is in
the selected node set. The same cohort is reused across models and attacks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

KS = (1, 5, 10, 20, 50, 100)
MAIN = ("cn", "gcn", "sgdiff")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = repo / "results/part2-peer-cn-gcn-sgdiff-all/per_seed.csv"
    with source.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    with (source.parent / "provenance.json").open() as stream:
        num_nodes = {row["dataset"]: int(row["num_nodes"]) for row in json.load(stream)}
    eval_root = repo / "runs/part2-peer/evaluation"
    split_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for source_row in source_rows:
        dataset, seed = source_row["dataset"], int(source_row["seed"])
        key = dataset, seed
        if key not in split_cache:
            with np.load(eval_root / "splits" / f"{dataset}_seed{seed}.npz") as split:
                pos, neg = split["test_pos"].copy(), split["test_neg"].copy()
                train = split["train_pos"]
                n_nodes = num_nodes[dataset]
                degree = np.bincount(train.ravel(), minlength=n_nodes)
            # Descending degree, ascending node id for a deterministic tie-break.
            ranking = np.lexsort((np.arange(n_nodes), -degree))
            split_cache[key] = pos, neg, ranking
        pos, neg, ranking = split_cache[key]
        pairs = np.vstack((pos, neg))
        labels = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
        prediction = eval_root / "runs" / f"{dataset}_{source_row['model']}_s{seed}" / f"predictions_{source_row['condition']}.npz"
        with np.load(prediction) as saved:
            scores = np.r_[saved["positive"], saved["negative"]]
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid prediction vector: {prediction}")
        full_auc = roc_auc_score(labels, scores)
        if not np.isclose(full_auc, float(source_row["auc"]), atol=1e-12):
            raise ValueError(f"Full AUROC does not match published peer result: {prediction}")
        for k in KS:
            count = math.ceil(len(ranking) * k / 100)
            chosen = np.zeros(len(ranking), dtype=bool)
            chosen[ranking[:count]] = True
            selected = chosen[pairs].any(axis=1)
            y, s = labels[selected], scores[selected]
            npos = int(y.sum())
            nneg = len(y) - npos
            rows.append(dict(dataset=dataset, seed=seed, model=source_row["model"],
                             condition=source_row["condition"], top_node_percent=k,
                             selected_nodes=count, positive=npos, negative=nneg,
                             auc=roc_auc_score(y, s) if npos and nneg else "",
                             ap=average_precision_score(y, s) if npos else ""))
    fields = list(rows[0])
    with (out / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        gkey = row["dataset"], row["model"], row["condition"], row["top_node_percent"]
        grouped.setdefault(gkey, []).append(row)
    summary = []
    for (dataset, model, condition, k), group in sorted(grouped.items()):
        auc = np.array([float(x["auc"]) for x in group if x["auc"] != ""])
        ap = np.array([float(x["ap"]) for x in group if x["ap"] != ""])
        summary.append(dict(dataset=dataset, model=model, condition=condition,
                            top_node_percent=k, valid_auc_seeds=len(auc),
                            auc_mean=auc.mean() if len(auc) else "",
                            auc_sd=auc.std(ddof=1) if len(auc) > 1 else "",
                            ap_mean=ap.mean() if len(ap) else "",
                            ap_sd=ap.std(ddof=1) if len(ap) > 1 else "",
                            positive_mean=np.mean([x["positive"] for x in group]),
                            negative_mean=np.mean([x["negative"] for x in group])))
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for condition, stem in (("clean", "clean"), ("delete_hub_0.5", "hub_delete_50")):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
        for axis, dataset in zip(axes, ("cora", "citeseer", "pubmed")):
            for model in MAIN:
                cells = [x for x in summary if x["dataset"] == dataset and
                         x["model"] == model and x["condition"] == condition]
                x = np.array([r["top_node_percent"] for r in cells])
                y = np.array([r["auc_mean"] for r in cells], dtype=float)
                sd = np.array([r["auc_sd"] for r in cells], dtype=float)
                axis.plot(x, y, marker="o", label=model.upper() if model != "sgdiff" else "SGDiff")
                axis.fill_between(x, y - sd, y + sd, alpha=.13)
            axis.set_title(dataset.capitalize())
            axis.set_xticks(KS)
            axis.set_xlabel("Top k% nodes by train degree")
            axis.grid(alpha=.2)
        axes[0].set_ylabel("AUROC on pairs incident to selected nodes")
        axes[-1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_top_degree_auc.pdf")
        fig.savefig(out / f"{stem}_top_degree_auc.png", dpi=180)
        plt.close(fig)

    lookup = {(r["dataset"], r["model"], r["condition"], r["top_node_percent"]): r for r in summary}
    table = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Model & Top 5\% & Top 20\% & All nodes \\", r"\midrule"]
    for dataset in ("cora", "citeseer", "pubmed"):
        for model in MAIN:
            vals = [lookup[(dataset, model, "clean", k)]["auc_mean"] for k in (5, 20, 100)]
            table.append(f"{dataset.capitalize()} & {model.upper() if model != 'sgdiff' else 'SGDiff'} & " +
                         " & ".join(f"{v:.3f}" for v in vals) + r" \\")
        table.append(r"\midrule" if dataset != "pubmed" else r"\bottomrule")
    table += [r"\end{tabular}",
              r"\caption{Clean AUROC on test pairs incident to nodes with the highest clean training degree. Values average three seeds. The 100\% column is the original full-test AUROC.}",
              r"\label{tab:topdegree}", r"\end{table}"]
    tex = [r"\subsection{Performance around high-degree nodes}",
           r"Following the team's degree-stratified diagnostic, we rank nodes by degree in each seed's clean training graph. For a given $k$, we select the top $k\%$ nodes, breaking degree ties by node ID, and evaluate the original held-out test pairs with at least one endpoint in that set. The node set and test pairs remain fixed across all models and perturbations. We use $k\in\{1,5,10,20,50,100\}$, and report AUROC and AP only when the selected cohort contains both classes. All 14 conditions, seven model variants, and three seeds are retained in the accompanying CSV files; the main plots show clean and 50\% hub-priority edge deletion.",
           *table,
           r"\begin{figure}[H]", r"\centering",
           r"\includegraphics[width=\linewidth]{clean_top_degree_auc.pdf}",
           r"\caption{Clean AUROC as the high-degree node cohort expands. Shading shows standard deviation across three seeds.}",
           r"\label{fig:topdegreeclean}", r"\end{figure}",
           r"\begin{figure}[H]", r"\centering",
           r"\includegraphics[width=\linewidth]{hub_delete_50_top_degree_auc.pdf}",
           r"\caption{The same fixed cohorts under 50\% hub-priority edge deletion.}",
           r"\label{fig:topdegreehubdelete}", r"\end{figure}",
           r"The top 1\% cohorts are highly imbalanced, especially in Cora and CiteSeer, and their curves vary substantially across seeds. Cohort AUROC is conditional on node degree; it is not the precision of the model's highest-scoring predicted links. These observations alone do not establish that SGDiff's subgraph node cap causes its hub behavior."]
    (out / "report_fragment.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} per-seed rows and {len(summary)} summary rows to {out}")


if __name__ == "__main__":
    main()
