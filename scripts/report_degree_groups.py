"""Paired AUROC/AP within fixed strata of minimum clean training endpoint degree."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.factory import load_split
from sgdiff.engine.reproduction import COMPONENTS, OWNED, PRIMARY, conditions, digest, expanded
from sgdiff.engine.team_study import settings


def rows(dataset, seed, names, root, data_root):
    base = root / dataset / f"seed_{seed}"
    graph, split = load_split(settings(dataset, seed, "reference", data_root, "cpu"))
    degree = np.bincount(split.train_pos.ravel(), minlength=graph.num_nodes)
    bank = base / "graphs"
    manifest = json.loads((bank / "manifest.json").read_text())
    if digest(bank / "test.npz") != manifest["test_sha256"]:
        raise ValueError(f"changed test cohort: {bank}")
    with np.load(bank / "test.npz") as test:
        y = test["labels"]
        min_degree = degree[test["pairs"]].min(1)
    groups = {"0-1": min_degree <= 1, "2-3": (min_degree >= 2) & (min_degree <= 3), "4+": min_degree >= 4}
    clean = {}
    for c in conditions():
        ref = manifest["conditions"][c.name]["file_sha256"]
        for model in expanded(names):
            variants = [model, *COMPONENTS["sgdiff"]] if model == "sgdiff" else [model]
            with np.load(base / model / f"{c.name}.npz") as scores:
                if scores["graph_sha"].item() != ref:
                    raise ValueError(f"wrong graph: {dataset}/{seed}/{model}/{c.name}")
                for variant in variants:
                    for label, mask in groups.items():
                        target, values = y[mask], scores[variant][mask]
                        positive = int(target.sum())
                        result = dict(dataset=dataset, seed=seed, model=variant, condition=c.name,
                                      min_train_endpoint_degree=label, positive=positive, negative=len(target)-positive)
                        if 0 < positive < len(target):
                            for metric, fn in (("auc",roc_auc_score), ("ap",average_precision_score)):
                                value = fn(target, values)
                                key = (variant,label,metric)
                                if c.name == "clean":
                                    clean[key] = value
                                result[metric] = value
                                result[metric+"_drop"] = clean[key] - value
                        yield result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["cora","citeseer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--models", nargs="+", choices=PRIMARY, default=list(OWNED))
    p.add_argument("--root", type=Path, default=Path("results/paired-reproduction"))
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    with a.out.open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=("dataset","seed","model","condition","min_train_endpoint_degree",
                             "positive","negative","auc","auc_drop","ap","ap_drop"), lineterminator="\n")
        writer.writeheader()
        for d in a.datasets:
            for s in a.seeds:
                writer.writerows(rows(d,s,a.models,a.root,a.data_root))
