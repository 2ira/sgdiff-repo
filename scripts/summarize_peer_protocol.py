"""Aggregate paired, per-seed peer-protocol results; keep graph mismatch flags visible."""
import argparse
from pathlib import Path

import pandas as pd


def summarize(table):
    df = pd.read_csv(table / "per_seed.csv")
    key = ["dataset", "model", "condition", "seed"]
    datasets = set(df.dataset)
    is_extension = datasets == {"pubmed"}
    expected_cells = 198 if is_extension else 504
    if not is_extension and datasets != {"cora", "citeseer"}:
        raise ValueError("unexpected data sets")
    if (len(df) != expected_cells or df.duplicated(key).any()
            or df.groupby(key[:-1]).size().ne(3).any() or set(df.seed) != {0,1,2}):
        raise ValueError("incomplete or duplicated peer evaluation grid")
    metrics = ["auc", "ap", "hits100", "precision100"]
    full = df[df.model == df.family]
    paired = df.merge(full[["dataset", "family", "seed", "condition", *metrics]],
                      on=["dataset","family","seed","condition"], suffixes=("", "_full"), validate="many_to_one")
    clean = df[df.condition == "clean"][["dataset","model","seed",*metrics]]
    paired = paired.merge(clean, on=["dataset","model","seed"], suffixes=("", "_clean"), validate="many_to_one")
    fields = []
    for m in metrics:
        paired[m+"_drop"] = paired[m+"_clean"] - paired[m]
        paired[m+"_full_minus_variant"] = paired[m+"_full"] - paired[m]
        fields.extend([m, m+"_drop", m+"_full_minus_variant"])
    groups = paired.groupby(["dataset","family","model","condition"])
    out = groups[fields].agg(["mean","std"])
    out.columns = ["_".join(k) for k in out.columns]
    if not is_extension:
        out.insert(0, "matching_peer_graphs", groups.graph_hash_matches.sum())
    out.insert(0, "seeds", groups.seed.count())
    out.to_csv(table / "summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", type=Path)
    summarize(p.parse_args().table)
