"""Figures and report text for Part 2 using Jiatong's exact node-rank output."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = (("cn", "CN"), ("gcn", "GCN"), ("sgdiff", "SGDiff"))
DATASETS = ("cora", "citeseer", "pubmed")
RANKS = ("degree", "pagerank")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    root = a.results.resolve()
    df = pd.read_csv(root / "summary.csv")
    assert len(df) == 3 * 7 * 14 * 2 * 101
    assert df.n_seeds_present.eq(3).all()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"cn": "#126e82", "gcn": "#d98c10", "sgdiff": "#6b4c9a"}
    for condition, stem in (("clean", "clean"), ("delete_hub_0.5", "hub_delete_50")):
        fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.2), sharex=True, sharey=True)
        for i, dataset in enumerate(DATASETS):
            for j, rank in enumerate(RANKS):
                ax = axes[i, j]
                for model, label in MODELS:
                    rows = df[(df.dataset == dataset) & (df.variant == model) &
                              (df.condition == condition) & (df["rank"] == rank)].sort_values("k")
                    assert len(rows) == 101
                    x = rows.k.to_numpy()
                    y = rows["mean"].to_numpy(dtype=float)
                    sd = rows.sd.to_numpy(dtype=float)
                    valid = rows.n_seeds_valid.eq(3).to_numpy() & (x > 0)
                    ax.plot(x[valid], y[valid], color=colors[model], label=label, lw=1.5)
                    ax.fill_between(x[valid], np.maximum(0, y[valid] - sd[valid]),
                                    np.minimum(1, y[valid] + sd[valid]), color=colors[model], alpha=.11)
                    low = valid & rows.n_low_support.gt(0).to_numpy()
                    ax.scatter(x[low], y[low], color=colors[model], s=8, alpha=.7)
                ax.set(title=f"{dataset.capitalize()} | {rank.title()}", xlim=(0, 100), ylim=(0, 1))
                ax.grid(alpha=.16)
                if i == 2:
                    ax.set_xlabel("Top k% nodes")
                if j == 0:
                    ax.set_ylabel("Incident-pair AUROC")
        axes[0, 0].legend(frameon=False, loc="lower right")
        fig.suptitle(f"{condition}: node-importance-conditioned link prediction\n"
                     "Mean ± sample SD across 3 seeds; dots mark low-support cohorts", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(root / f"{stem}_node_rank_auc.pdf")
        fig.savefig(root / f"{stem}_node_rank_auc.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for rank, linestyle in (("degree", "-"), ("pagerank", "--")):
            rows = df[(df.dataset == dataset) & (df.variant == "cn") &
                      (df.condition == "clean") & (df["rank"] == rank)].sort_values("k")
            for field, color, label in (("mean_pos_coverage", "#126e82", "positive"),
                                        ("mean_neg_coverage", "#c04747", "negative")):
                ax.plot(rows.k, rows[field], color=color, ls=linestyle, lw=1.5,
                        label=f"{rank} {label}")
        ax.set(title=dataset.capitalize(), xlim=(0, 100), ylim=(0, 1), xlabel="Top k% nodes")
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Fraction of original test pairs retained")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "test_pair_coverage.pdf")
    fig.savefig(root / "test_pair_coverage.png", dpi=180)
    plt.close(fig)

    lookup = {(r.dataset, r.variant, r["rank"], int(r.k)): r for _, r in
              df[df.condition.eq("clean")].iterrows()}
    tex = [r"\subsection{Node-importance-conditioned AUROC}",
           r"To align with the team's node-rank AUROC protocol, we used the same pinned evaluator on our saved predictions. For each dataset and seed, Degree and PageRank are computed on the clean training-positive graph; tied ranks are resolved by SHA256 of the dataset, seed and node ID. At each integer $k=0,\ldots,100$, we retain original test pairs with at least one endpoint among the top $\lceil Nk/100\rceil$ nodes. All models and perturbations use the same fixed cohort. AUROC is pooled over retained positive and negative pairs; $k=0$ is undefined and $k=100$ exactly reproduces the original test AUROC. Three seed estimates are summarized with their sample standard deviation. Points with fewer than 20 positives or 20 negatives are flagged as low support.",
           r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{llccc}",
           r"\toprule", r"Dataset & Model & Degree Top 10\% & PageRank Top 10\% & All nodes \\",
           r"\midrule"]
    for dataset in DATASETS:
        for model, label in MODELS:
            d = lookup[(dataset, model, "degree", 10)]["mean"]
            pr = lookup[(dataset, model, "pagerank", 10)]["mean"]
            full = lookup[(dataset, model, "degree", 100)]["mean"]
            tex.append(f"{dataset.capitalize()} & {label} & {d:.3f} & {pr:.3f} & {full:.3f}" + r" \\")
        tex.append(r"\midrule" if dataset != "pubmed" else r"\bottomrule")
    tex += [r"\end{tabular}",
            r"\caption{Clean AUROC conditioned on high-importance nodes, averaged over three seeds. The full-test column is the original AUROC.}",
            r"\label{tab:noderank}", r"\end{table}"]
    for stem, caption in (("clean", "Clean"), ("hub_delete_50", r"50\% hub-priority edge deletion")):
        tex += [r"\begin{figure}[H]", r"\centering",
                rf"\includegraphics[width=0.94\linewidth]{{{stem}_node_rank_auc.pdf}}",
                rf"\caption{{{caption} AUROC by Degree and PageRank cohort. Shading is the sample SD across seeds; dots denote any low-support seed.}}",
                r"\end{figure}"]
    tex += [r"\begin{figure}[H]", r"\centering",
            r"\includegraphics[width=\linewidth]{test_pair_coverage.pdf}",
            r"\caption{Positive and negative test-pair coverage at each node-percentile cutoff. Different class coverage changes the selected cohort's difficulty.}",
            r"\end{figure}",
            r"These are degree- or PageRank-conditioned results, not model-score top-$k$ precision. Subset AUROC changes with the test cohort and its positive/negative composition; it does not by itself show that a predictor improved. Full per-seed curves, paired attack-minus-clean differences, support counts and audit records accompany the report."]
    (root / "report_fragment.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")
    print("Wrote aligned figures and report fragment", flush=True)


if __name__ == "__main__":
    main()
