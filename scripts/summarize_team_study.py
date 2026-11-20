"""Replay stored predictions, verify paired controls, then aggregate within dataset."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.team_study import COMPONENTS, PRIMARY, study_plan
from sgdiff.engine.benchmark import pairs_fingerprint


def load_run(folder, dataset, seed):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = json.loads((folder / "metrics.json").read_text())
    plan = study_plan()
    models = [*PRIMARY, *(v for vs in COMPONENTS.values() for v in vs)]
    expected = {(m, c.name) for m in models for c in plan}
    by_key = {(r["model"], r["condition"]): r for r in rows}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError(f"incomplete or duplicate cells in {folder}")
    if (manifest["dataset"], manifest["seed"], manifest["profile"]) != (dataset, seed, folder.parents[1].name):
        raise ValueError(f"unexpected run identity in {folder}")
    with np.load(folder / "queries_and_scores.npz") as a:
        for split, signature in manifest["queries"].items():
            if pairs_fingerprint(a[f"{split}_pairs"], a[f"{split}_labels"]) != signature:
                raise ValueError(f"query mismatch in {folder}")
        labels = a["test_labels"]
        for cond in plan:
            name = cond.name
            info = manifest["conditions"][name]
            if info["deleted"] + info["inserted"] != round(info["clean_edges"] * cond.edge_ratio):
                raise ValueError(f"edit budget mismatch: {folder}/{name}")
            for model in models:
                score = a[f"{name}__{model}"]
                recorded = by_key[model, name]
                if (recorded["dataset"], recorded["seed"]) != (dataset, seed):
                    raise ValueError(f"metric identity mismatch in {folder}")
                np.testing.assert_allclose([roc_auc_score(labels, score), average_precision_score(labels, score)],
                                           [recorded["auc"], recorded["ap"]], rtol=0, atol=1e-12)
            if not cond.feature_ratio:
                np.testing.assert_array_equal(a[f"{name}__fsd"], a[f"{name}__gae"])
                np.testing.assert_array_equal(a[f"{name}__feature_cosine"], a["clean__feature_cosine"])
            if not cond.edge_ratio:
                for m in ("katz", "cn", "katz_h3", "katz_h2", "sp4lp_featureless"):
                    np.testing.assert_allclose(a[f"{name}__{m}"], a[f"clean__{m}"], rtol=1e-6, atol=1e-6)
            for metric in ("auc", "ap"):
                np.testing.assert_allclose(by_key["cn", name][metric], by_key["katz_h2", name][metric])
            if name.startswith("joint_"):
                edge_condition = name.removeprefix("joint_")
                for m in ("katz", "cn"):
                    np.testing.assert_array_equal(a[f"{name}__{m}"], a[f"{edge_condition}__{m}"])
                mask = f"feature_mask_{round(100 * cond.feature_ratio)}"
                np.testing.assert_array_equal(a[f"{name}__feature_cosine"], a[f"{mask}__feature_cosine"])
    return by_key


def save_table(rows, path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    return np.mean(values), np.std(values, ddof=1) if len(values) > 1 else ""


def summarize(root, datasets, seeds):
    if len(set(seeds)) != len(seeds) or len(set(datasets)) != len(datasets):
        raise ValueError("duplicate seed or dataset")
    summary, comparisons = [], []
    for dataset in datasets:
        runs = [load_run(root / dataset / f"seed_{s}", dataset, s) for s in seeds]
        for model, condition in runs[0]:
            row = dict(dataset=dataset, model=model, condition=condition, n=len(seeds))
            for metric in ("auc", "ap"):
                values = [r[model, condition][metric] for r in runs]
                drops = [r[model, "clean"][metric] - v for r, v in zip(runs, values)]
                row[metric], row[f"{metric}_sd"] = describe(values)
                row[f"{metric}_drop"], row[f"{metric}_drop_sd"] = describe(drops)
            summary.append(row)
            parent = next((m for m, parts in COMPONENTS.items() if model in parts), None)
            if parent:
                row = dict(dataset=dataset, full=parent, variant=model, condition=condition, n=len(seeds))
                for metric in ("auc", "ap"):
                    diff = [r[parent, condition][metric] - r[model, condition][metric] for r in runs]
                    row[metric + "_full_minus_variant"], row[metric + "_difference_sd"] = describe(diff)
                    row[metric + "_positive_seeds"] = sum(v > 0 for v in diff)
                comparisons.append(row)
    save_table(summary, root / "summary.csv")
    save_table(comparisons, root / "components.csv")
    print(f"verified {len(summary) * len(seeds)} cells, wrote {len(comparisons)} paired contrasts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="profile directory, e.g. results/team-study/local")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    summarize(args.root, args.datasets, args.seeds)
