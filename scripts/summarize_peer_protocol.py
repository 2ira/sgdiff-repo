"""Aggregate paired, per-seed peer-protocol results; keep graph mismatch flags visible."""
import argparse
from pathlib import Path

import pandas as pd


def summarize(table):
    df = pd.read_csv(table / "per_seed.csv")
    key = ["dataset", "model", "condition", "seed"]
    expected_conditions = {
        "clean", "rf_0.1", "rf_0.25", "rf_0.5", "dice_0.1", "dice_0.25", "dice_0.5",
        "ea_0.1", "ea_0.25", "ea_0.5", "delete_0.5", "delete_hub_0.5", "insert_0.5",
        "feature_shuffle",
    }
    if (df.empty or df.duplicated(key).any() or set(df.seed) != {0, 1, 2}
            or set(df.condition) != expected_conditions
            or df.groupby(key[:-1]).size().ne(3).any()):
        raise ValueError("incomplete or duplicated peer evaluation grid")
    expected = len(set(df.dataset)) * len(set(df.model)) * 3 * len(expected_conditions)
    if len(df) != expected:
        raise ValueError("the data/model/seed/condition grid is not rectangular")
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
    if "graph_hash_matches" in paired:
        out.insert(0, "matching_peer_graphs", groups.graph_hash_matches.sum())
    out.insert(0, "seeds", groups.seed.count())
    out.to_csv(table / "summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", type=Path)
    summarize(p.parse_args().table)
