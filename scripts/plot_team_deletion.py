"""Plot paired deletion effects from completed team-study seed directories."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    names = {"cn": "Common Neighbors", "katz": "Katz", "feature_cosine": "Feature cosine",
             "gae": "GCN–GAE", "fsd": "FSD + GAE", "pencil": "PENCIL-style",
             "sp4lp": "SP4LP", "sgdiff": "SGDiff v2"}
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4.7))
    for ax, dataset in zip(axes, ["cora", "citeseer"]):
        diffs = []
        for seed in (0, 1, 2):
            rows = json.loads((root / dataset / f"seed_{seed}" / "metrics.json").read_text())
            auc = {(r["model"], r["condition"]): r["auc"] for r in rows}
            diffs.append([100 * (auc[m, "degree_delete_50"] - auc[m, "random_delete_50"]) for m in names])
        y = np.arange(len(names))
        ax.axvline(0, color="#687680", lw=1)
        for dy, values in zip([-.13, 0., .13], diffs):
            ax.scatter(values, y + dy, s=23, facecolors="white", edgecolors="#518990", zorder=2)
        ax.scatter(np.mean(diffs, axis=0), y, marker="D", s=34, color="#123e51", zorder=3)
        ax.set_title({"cora": "Cora", "citeseer": "CiteSeer"}[dataset], loc="left")
        ax.grid(axis="x", color="#dddddd", lw=.5)
        ax.set_yticks(y, names.values())
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.supxlabel("AUROC difference in percentage points (low-degree deletion minus uniform deletion)")
    fig.suptitle("Removing 50% of visible edges", x=.02, ha="left", weight="bold")
    fig.text(.02, .91, "Open circles show paired seeds. Diamonds show their mean.")
    fig.tight_layout(rect=(0, .04, 1, .9))
    fig.savefig(root / "paired_deletion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
