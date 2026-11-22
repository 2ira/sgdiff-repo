"""Build the Part 2 report tables, narrative, and scientific figures.

The script intentionally refuses partial grids so that no draft report can be
mistaken for the final aligned comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "pubmed")
MODELS = (
    "cn",
    "gcn",
    "gcn_no_neighbors",
    "gcn_no_attributes",
    "sgdiff",
    "sgdiff_structure",
    "sgdiff_feature",
)
PRIMARY = ("cn", "gcn", "sgdiff")
CONDITIONS = (
    "clean",
    "rf_0.1", "rf_0.25", "rf_0.5",
    "dice_0.1", "dice_0.25", "dice_0.5",
    "ea_0.1", "ea_0.25", "ea_0.5",
    "delete_0.5", "delete_hub_0.5", "insert_0.5", "feature_shuffle",
)
LABEL = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "cn": "CN",
    "gcn": "GCN",
    "gcn_no_neighbors": "GCN no neighbors",
    "gcn_no_attributes": "GCN no attributes",
    "sgdiff": "SGDiff",
    "sgdiff_structure": "SGDiff structure",
    "sgdiff_feature": "SGDiff feature",
    "clean": "Clean",
    "rf_0.5": "RF 50%",
    "dice_0.5": "DICE 50%",
    "ea_0.5": "EA 50%",
    "delete_0.5": "Delete 50%",
    "delete_hub_0.5": "Hub delete 50%",
    "insert_0.5": "Insert 50%",
    "feature_shuffle": "Feature shuffle",
}
COLORS = {"cn": "#4C78A8", "gcn": "#F58518", "sgdiff": "#54A24B"}


def validate(df: pd.DataFrame) -> None:
    expected = len(DATASETS) * len(MODELS) * 3 * len(CONDITIONS)
    if len(df) != expected:
        raise ValueError(f"partial grid: found {len(df)} rows, expected {expected}")
    for column in ("dataset", "seed", "model", "condition", "auc", "ap", "hits100", "precision100"):
        if column not in df:
            raise ValueError(f"missing column: {column}")
    if set(df.dataset) != set(DATASETS):
        raise ValueError(f"dataset mismatch: {sorted(set(df.dataset))}")
    if set(df.model) != set(MODELS):
        raise ValueError(f"model mismatch: {sorted(set(df.model))}")
    if set(df.seed.astype(int)) != {0, 1, 2}:
        raise ValueError(f"seed mismatch: {sorted(set(df.seed))}")
    if set(df.condition) != set(CONDITIONS):
        raise ValueError(f"condition mismatch: {sorted(set(df.condition))}")
    if not np.isfinite(df[["auc", "ap", "hits100", "precision100"]].to_numpy()).all():
        raise ValueError("non-finite metric detected")
    hashes = df["graph_hash_matches"].astype(str).str.lower()
    if not hashes.isin({"1", "true"}).all():
        raise ValueError("at least one attacked graph hash does not match the peer asset")
    counts = df.groupby(["dataset", "model", "seed", "condition"]).size()
    if len(counts) != expected or not (counts == 1).all():
        raise ValueError("grid contains duplicate or missing cells")


def stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "model", "condition"], sort=False)
        .agg(
            auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
            ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
            hits_mean=("hits100", "mean"), hits_sd=("hits100", "std"),
            precision_mean=("precision100", "mean"), precision_sd=("precision100", "std"),
        )
        .reset_index()
    )


def tex_escape(value: str) -> str:
    return (value.replace("\\", r"\textbackslash{}")
                 .replace("_", r"\_")
                 .replace("%", r"\%")
                 .replace("&", r"\&")
                 .replace("#", r"\#"))


def pm(mean: float, sd: float) -> str:
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def lookup(summary: pd.DataFrame, dataset: str, model: str, condition: str) -> pd.Series:
    row = summary[(summary.dataset == dataset) & (summary.model == model) & (summary.condition == condition)]
    if len(row) != 1:
        raise ValueError(f"missing summary row: {(dataset, model, condition)}")
    return row.iloc[0]


def write_tables(summary: pd.DataFrame, target: Path) -> None:
    lines = [
        "\\subsection{Clean link prediction}",
        "Table~\\ref{tab:clean} reports all four clean metrics. Values are mean $\\pm$ sample standard deviation over three seeds.",
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Model & AUROC & AP & Hits@100 & Precision@100 \\\\",
        "\\midrule",
    ]
    for di, dataset in enumerate(DATASETS):
        for model in PRIMARY:
            row = lookup(summary, dataset, model, "clean")
            lines.append(
                f"{LABEL[dataset]} & {LABEL[model]} & {pm(row.auc_mean, row.auc_sd)} & "
                f"{pm(row.ap_mean, row.ap_sd)} & {pm(row.hits_mean, row.hits_sd)} & "
                f"{pm(row.precision_mean, row.precision_sd)} \\\\"
            )
        if di != len(DATASETS) - 1:
            lines.append("\\addlinespace")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Clean performance under the aligned evaluator.}",
        "\\label{tab:clean}",
        "\\end{table}",
        "",
        "\\subsection{Strong structural perturbations}",
        "Table~\\ref{tab:strong} gives AUROC at the 50\\% budgets and the change from the matched clean mean.",
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Model & RF 50\\% & DICE 50\\% & EA 50\\% \\\\",
        "\\midrule",
    ]
    for di, dataset in enumerate(DATASETS):
        for model in PRIMARY:
            clean = lookup(summary, dataset, model, "clean").auc_mean
            cells = []
            for condition in ("rf_0.5", "dice_0.5", "ea_0.5"):
                attacked = lookup(summary, dataset, model, condition).auc_mean
                cells.append(f"{attacked:.3f} ({attacked-clean:+.3f})")
            lines.append(f"{LABEL[dataset]} & {LABEL[model]} & " + " & ".join(cells) + " \\\\")
        if di != len(DATASETS) - 1:
            lines.append("\\addlinespace")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{AUROC under the strongest RF, DICE, and EA settings. Parentheses show the change from clean.}",
        "\\label{tab:strong}",
        "\\end{table}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ncnc_reference(path: Path) -> pd.DataFrame:
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    peer = pd.read_csv(path)
    required = {"dataset", "model", "component", "seed", "attack", "ratio", "auc"}
    if not required.issubset(peer.columns):
        raise ValueError(f"NCNC reference is missing columns: {sorted(required - set(peer.columns))}")
    peer = peer[(peer.model == "ncnc") & (peer.component == "full")].copy()
    peer = peer[peer.dataset.isin(("cora", "citeseer"))]
    peer["condition"] = np.where(
        peer.attack.isin(("clean", "feature_shuffle")),
        peer.attack,
        peer.attack + "_" + peer.ratio.map(lambda value: f"{value:g}"),
    )
    expected = 2 * 3 * len(CONDITIONS)
    counts = peer.groupby(["dataset", "seed", "condition"]).size()
    if (len(peer) != expected or len(counts) != expected or not (counts == 1).all()
            or set(peer.seed.astype(int)) != {0, 1, 2}
            or set(peer.condition) != set(CONDITIONS)):
        raise ValueError("NCNC reference does not contain the complete aligned Cora/CiteSeer grid")
    peer.attrs.update(source=str(path.resolve()), source_sha256=source_sha256)
    return peer


def write_ncnc_comparison(summary: pd.DataFrame, peer: pd.DataFrame, generated: Path) -> None:
    peer[["dataset", "seed", "condition", "auc", "ap", "hits100", "precision100"]].to_csv(
        generated / "ncnc_reference_per_seed.csv", index=False
    )
    (generated / "ncnc_reference_provenance.json").write_text(
        json.dumps({
            "peer_commit": "95083da808e500c02da458e205f4b861ceb80871",
            "source": peer.attrs["source"],
            "source_sha256": peer.attrs["source_sha256"],
            "selection": "model=ncnc, component=full, datasets=cora/citeseer",
            "rows": len(peer),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    peer_summary = (
        peer.groupby(["dataset", "condition"])
        .auc.agg(["mean", "std"])
        .reset_index()
    )
    shown = ("clean", "rf_0.5", "dice_0.5", "ea_0.5", "feature_shuffle")
    lines = [
        "\\subsection{CN compared with the team's NCNC result}",
        "Table~\\ref{tab:cn-ncnc} uses Junxiong's full NCNC rows from the same pinned evaluator. "
        "The team reference covers Cora and CiteSeer, so PubMed is not extrapolated.",
        "\\begin{table}[H]",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Model & Clean & RF 50\\% & DICE 50\\% & EA 50\\% & Feature shuffle \\\\",
        "\\midrule",
    ]
    for di, dataset in enumerate(("cora", "citeseer")):
        cn_values = []
        ncnc_values = []
        for condition in shown:
            cn = lookup(summary, dataset, "cn", condition)
            ref = peer_summary[(peer_summary.dataset == dataset) & (peer_summary.condition == condition)]
            if len(ref) != 1:
                raise ValueError(f"missing NCNC reference row: {(dataset, condition)}")
            ref = ref.iloc[0]
            cn_values.append(pm(cn.auc_mean, cn.auc_sd))
            ncnc_values.append(pm(ref["mean"], ref["std"]))
        lines.append(f"{LABEL[dataset]} & CN & " + " & ".join(cn_values) + " \\\\")
        lines.append(f"{LABEL[dataset]} & NCNC & " + " & ".join(ncnc_values) + " \\\\")
        if di == 0:
            lines.append("\\addlinespace")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        "\\caption{AUROC for parameter-free CN and the team's learned NCNC model under matched data and perturbations.}",
        "\\label{tab:cn-ncnc}",
        "\\end{table}",
        "NCNC is substantially stronger on the clean graph and under structural edits. "
        "Feature shuffling removes much of that advantage because NCNC uses learned node representations, "
        "whereas CN ignores attributes and therefore reproduces its clean score exactly. This does not make "
        "CN the stronger overall predictor; it identifies the source of its invariance.",
    ]
    (generated / "ncnc_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    peer.to_csv(generated / "ncnc_reference_per_seed.csv", index=False)


def write_narrative(summary: pd.DataFrame, generated: Path) -> dict:
    findings: dict[str, object] = {"clean": {}, "worst": {}, "ablations": {}}
    clean_sentences = []
    robustness_sentences = []
    for dataset in DATASETS:
        clean_rows = [lookup(summary, dataset, model, "clean") for model in PRIMARY]
        winner_idx = int(np.argmax([row.auc_mean for row in clean_rows]))
        winner = PRIMARY[winner_idx]
        findings["clean"][dataset] = {
            model: float(lookup(summary, dataset, model, "clean").auc_mean) for model in PRIMARY
        }
        clean_sentences.append(
            f"On {LABEL[dataset]}, {LABEL[winner]} has the highest clean AUROC among the three primary models "
            f"({clean_rows[winner_idx].auc_mean:.3f})."
        )
        for model in PRIMARY:
            clean = lookup(summary, dataset, model, "clean").auc_mean
            candidates = summary[(summary.dataset == dataset) & (summary.model == model) & (summary.condition != "clean")]
            worst = candidates.loc[candidates.auc_mean.idxmin()]
            findings["worst"][f"{dataset}:{model}"] = {
                "condition": str(worst.condition),
                "auc": float(worst.auc_mean),
                "drop": float(worst.auc_mean - clean),
            }
        robust_model = max(
            PRIMARY,
            key=lambda model: min(
                lookup(summary, dataset, model, condition).auc_mean
                for condition in CONDITIONS if condition != "clean"
            ),
        )
        floor = min(
            lookup(summary, dataset, robust_model, condition).auc_mean
            for condition in CONDITIONS if condition != "clean"
        )
        robustness_sentences.append(
            f"Across the 13 perturbed conditions on {LABEL[dataset]}, {LABEL[robust_model]} has the highest worst-case AUROC ({floor:.3f})."
        )

    discussion = [
        "\\subsection{Main comparison}",
        " ".join(clean_sentences),
        "",
        "\\subsection{Robustness}",
        " ".join(robustness_sentences),
        "Figure~\\ref{fig:attacks} separates attack budget from attack type, while Figure~\\ref{fig:heatmap} reports matched clean-to-attack changes. Absolute attacked AUROC and degradation should be read together: a low-capacity model can show a small drop simply because its clean score is already low.",
    ]
    (generated / "discussion.tex").write_text("\n".join(discussion) + "\n", encoding="utf-8")

    ablation_lines = ["\\subsection{GCN inputs}"]
    for dataset in DATASETS:
        full = lookup(summary, dataset, "gcn", "clean").auc_mean
        no_n = lookup(summary, dataset, "gcn_no_neighbors", "clean").auc_mean
        no_x = lookup(summary, dataset, "gcn_no_attributes", "clean").auc_mean
        findings["ablations"][f"{dataset}:gcn"] = {
            "full": float(full), "no_neighbors": float(no_n), "no_attributes": float(no_x)
        }
        ablation_lines.append(
            f"For {LABEL[dataset]}, removing neighbor aggregation changes clean AUROC by {no_n-full:+.3f}, "
            f"while removing node attributes changes it by {no_x-full:+.3f}."
        )
    ablation_lines += ["", "\\subsection{SGDiff streams}"]
    for dataset in DATASETS:
        full = lookup(summary, dataset, "sgdiff", "clean").auc_mean
        structure = lookup(summary, dataset, "sgdiff_structure", "clean").auc_mean
        feature = lookup(summary, dataset, "sgdiff_feature", "clean").auc_mean
        findings["ablations"][f"{dataset}:sgdiff"] = {
            "full": float(full), "structure": float(structure), "feature": float(feature)
        }
        best_stream = "structure" if structure >= feature else "feature"
        best_value = max(structure, feature)
        ablation_lines.append(
            f"For {LABEL[dataset]}, the stronger single SGDiff readout is {best_stream}-only "
            f"({best_value:.3f} AUROC), compared with {full:.3f} for learned fusion."
        )
    ablation_lines.append(
        "These readouts share one trained checkpoint and one set of diffusion evidence. They identify which stream the fitted model uses at test time; they do not estimate the performance of independently retrained single-stream models."
    )
    (generated / "ablation_text.tex").write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")
    return findings


def clean_plot(summary: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.5 * len(DATASETS), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(PRIMARY))
    for ax, dataset in zip(axes, DATASETS):
        means = [lookup(summary, dataset, model, "clean").auc_mean for model in PRIMARY]
        sds = [lookup(summary, dataset, model, "clean").auc_sd for model in PRIMARY]
        ax.bar(x, means, yerr=sds, capsize=3, color=[COLORS[m] for m in PRIMARY], width=0.68)
        ax.set_title(LABEL[dataset])
        ax.set_xticks(x, [LABEL[m] for m in PRIMARY])
        ax.set_ylim(0.45, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUROC")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"clean_auroc.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def attack_plot(summary: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(10.4, 2.85 * len(DATASETS) + 0.6), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    ratios = (0.0, 0.1, 0.25, 0.5)
    for row, dataset in enumerate(DATASETS):
        for col, attack in enumerate(("rf", "dice", "ea")):
            ax = axes[row, col]
            for model in PRIMARY:
                means, sds = [], []
                for ratio in ratios:
                    condition = "clean" if ratio == 0 else f"{attack}_{ratio:g}"
                    value = lookup(summary, dataset, model, condition)
                    means.append(value.auc_mean)
                    sds.append(value.auc_sd)
                ax.errorbar(ratios, means, yerr=sds, marker="o", linewidth=1.8,
                            capsize=2, color=COLORS[model], label=LABEL[model])
            if row == 0:
                ax.set_title(attack.upper())
            if col == 0:
                ax.set_ylabel(f"{LABEL[dataset]}\nAUROC")
            if row == len(DATASETS) - 1:
                ax.set_xlabel("Edge-edit budget")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"attack_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_plot(summary: pd.DataFrame, figures: Path) -> None:
    shown = ("rf_0.5", "dice_0.5", "ea_0.5", "delete_0.5", "delete_hub_0.5", "insert_0.5", "feature_shuffle")
    rows, matrix = [], []
    for dataset in DATASETS:
        for model in PRIMARY:
            clean = lookup(summary, dataset, model, "clean").auc_mean
            rows.append(f"{LABEL[dataset]} - {LABEL[model]}")
            matrix.append([lookup(summary, dataset, model, condition).auc_mean - clean for condition in shown])
    values = np.asarray(matrix)
    limit = max(0.05, float(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xticks(np.arange(len(shown)), [LABEL[c] for c in shown], rotation=28, ha="right")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > 0.58 * limit else "black"
            ax.text(j, i, f"{values[i,j]:+.3f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label="AUROC change from clean")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"robustness_heatmap.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def ablation_plot(summary: pd.DataFrame, figures: Path) -> None:
    conditions = ("clean", "rf_0.5", "dice_0.5", "ea_0.5", "feature_shuffle")
    families = (
        ("GCN", ("gcn", "gcn_no_neighbors", "gcn_no_attributes")),
        ("SGDiff", ("sgdiff", "sgdiff_structure", "sgdiff_feature")),
    )
    palette = ("#4C78A8", "#F58518", "#54A24B")
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(3.75 * len(DATASETS), 6.9), sharey=True)
    axes = np.asarray(axes).reshape(2, len(DATASETS))
    x = np.arange(len(conditions))
    width = 0.24
    for row, (family, models) in enumerate(families):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            for index, (model, color) in enumerate(zip(models, palette)):
                means = [lookup(summary, dataset, model, condition).auc_mean for condition in conditions]
                ax.bar(x + (index - 1) * width, means, width=width, label=LABEL[model], color=color)
            ax.set_title(f"{family}: {LABEL[dataset]}")
            ax.set_xticks(x, [LABEL[c].replace(" 50%", "\n50%") for c in conditions], rotation=20, ha="right")
            ax.set_ylim(0.35, 1.0)
            ax.grid(axis="y", alpha=0.22)
            if col == 0:
                ax.set_ylabel("AUROC")
            if col == len(DATASETS) - 1:
                ax.legend(
                    fontsize=8,
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"ablations.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global DATASETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/part2-peer-cn-gcn-sgdiff/per_seed.csv"))
    parser.add_argument("--report", type=Path, default=Path("deliverables/part2_report"))
    parser.add_argument("--ncnc-results", type=Path, required=True,
                        help="Junxiong's aligned per-seed table for the full NCNC reference")
    parser.add_argument("--datasets", nargs="+", choices=("cora", "citeseer", "pubmed"),
                        default=list(DATASETS),
                        help="Datasets that must form the complete report grid")
    args = parser.parse_args()
    DATASETS = tuple(args.datasets)
    df = pd.read_csv(args.results)
    validate(df)
    summary = stats(df)
    generated = args.report / "generated"
    figures = args.report / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary.to_csv(generated / "part2_summary.csv", index=False)
    write_tables(summary, generated / "results_tables.tex")
    write_ncnc_comparison(summary, ncnc_reference(args.ncnc_results), generated)
    findings = write_narrative(summary, generated)
    clean_plot(summary, figures)
    attack_plot(summary, figures)
    heatmap_plot(summary, figures)
    ablation_plot(summary, figures)
    (generated / "analysis_summary.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    print(f"Built report assets from {len(df)} audited cells")


if __name__ == "__main__":
    main()
