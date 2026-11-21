"""Independently replay recorded scores; fail on any missing requested cell."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.reproduction import COMPONENTS, OWNED, PRIMARY, conditions, digest, expanded


def historical(dataset):
    sources = [Path(f"results/part2-aligned/cuda-local/robustness_{dataset}_observed_cuda.json")]
    root = Path("results/sgdiff-paper-v2")
    if dataset == "cora":
        sources.extend(sorted(root.glob("robustness_cora__v2_seed*.json")))
    else:
        sources.append(root / "robustness_citeseer__v2.json")
    values = {}
    for source in sources:
        for row in json.loads(source.read_text())["runs"]:
            name = row["model"]
            if name != "sgdiff" and not name.startswith(("katz", "sp4lp")):
                continue
            cond = "clean" if row["variant"] == "clean" else f"rq3_{row['attack']}_{round(100*row['ratio'])}"
            values[row["seed"], name, cond] = row["metrics"]
    return values


def read_seed(root, dataset, seed, names, archived):
    base = root / dataset / f"seed_{seed}"
    bank = base / "graphs"
    manifest = json.loads((bank / "manifest.json").read_text())
    if set(manifest["conditions"]) != {c.name for c in conditions()}:
        raise ValueError(f"incomplete graph bank: {bank}")
    assert manifest["identity"]["dataset"] == dataset and manifest["identity"]["seed"] == seed
    assert manifest["test_sha256"] == digest(bank / "test.npz")
    with np.load(bank / "test.npz") as data:
        labels = data["labels"]
    result = {}
    for cond, receipt in manifest["conditions"].items():
        assert digest(bank / f"{cond}.npz") == receipt["file_sha256"]
        for trained in expanded(names):
            meta = json.loads((base / trained / "recipe.json").read_text())
            if meta["test"] != manifest["test_sha256"]:
                raise ValueError(f"test cohort mismatch: {base}/{trained}")
            expected = {trained, *COMPONENTS["sgdiff"]} if trained == "sgdiff" else {trained}
            with np.load(base / trained / f"{cond}.npz") as data:
                if set(data.files) != expected | {"graph_sha"} or data["graph_sha"].item() != receipt["file_sha256"]:
                    raise ValueError(f"score or graph mismatch: {base}/{trained}/{cond}")
                for name in expected:
                    scores = data[name]
                    row = dict(dataset=dataset, seed=seed, model=name, condition=cond,
                               auc=roc_auc_score(labels, scores), ap=average_precision_score(labels, scores))
                    old = archived.get((seed, name, cond))
                    if old:
                        row.update(archived_auc=old["auc"], auc_minus_archived=row["auc"]-old["auc"],
                                   archived_ap=old["ap"], ap_minus_archived=row["ap"]-old["ap"])
                    result[name, cond] = row
    for (model, cond), row in result.items():
        for metric in ("auc", "ap"):
            row[f"{metric}_drop"] = result[model, "clean"][metric] - row[metric]
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, keys, lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def aggregate(root, datasets, seeds, names, out):
    out.mkdir(parents=True, exist_ok=True)
    per_seed, summary, components = [], [], []
    for dataset in datasets:
        archived = historical(dataset)
        runs = [read_seed(root, dataset, seed, names, archived) for seed in seeds]
        for run in runs:
            per_seed.extend(run.values())
        for name, cond in sorted(runs[0]):
            row = dict(dataset=dataset, model=name, condition=cond, n=len(seeds))
            for stat in ("auc", "ap", "auc_drop", "ap_drop", "archived_auc", "auc_minus_archived",
                         "archived_ap", "ap_minus_archived"):
                if all(stat in r[name, cond] for r in runs):
                    values = [r[name, cond][stat] for r in runs]
                    row[stat] = np.mean(values)
                    if len(seeds) > 1:
                        row[stat+"_sd"] = np.std(values, ddof=1)
            summary.append(row)
            for full, variants in COMPONENTS.items():
                if name not in variants:
                    continue
                comp = dict(dataset=dataset, full=full, variant=name, condition=cond, n=len(seeds))
                for metric in ("auc","ap"):
                    diffs = [r[full, cond][metric]-r[name, cond][metric] for r in runs]
                    comp[metric+"_full_minus_variant"] = np.mean(diffs)
                    comp[metric+"_positive_seeds"] = sum(val > 0 for val in diffs)
                    if len(seeds) > 1:
                        comp[metric+"_difference_sd"] = np.std(diffs, ddof=1)
                components.append(comp)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "summary.csv", summary)
    if components:
        write_csv(out / "components.csv", components)
    print(f"recomputed {len(per_seed)} cells on {datasets} seeds={seeds}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["cora","citeseer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--models", nargs="+", choices=PRIMARY, default=list(OWNED))
    p.add_argument("--root", type=Path, default=Path("results/paired-reproduction"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.datasets)) != len(a.datasets):
        p.error("datasets and seeds must be unique")
    aggregate(a.root, a.datasets, a.seeds, a.models, a.out)
