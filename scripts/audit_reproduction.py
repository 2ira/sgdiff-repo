"""Audit the shared observations and query cohorts independently of model scores."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.factory import load_split
from sgdiff.engine.reproduction import archive, digest, json_file
from sgdiff.engine.team_study import settings, study_datasets


def audit_dataset(dataset, seed, root, data_root):
    base = root / dataset / f"seed_{seed}" / "graphs"
    manifest = json.loads((base / "manifest.json").read_text())
    raw, split = load_split(settings(dataset, seed, "reference", data_root, "cpu"))
    rows, counts, arrays = [], {}, {}
    candidate = np.load(base / "test.npz")
    pairs, labels = candidate["pairs"], candidate["labels"]
    for profile in ("paper-rq3", "reference"):
        cfg = settings(dataset, seed, profile, data_root, "cpu")
        ds = study_datasets(raw, split, cfg, profile)
        np.testing.assert_array_equal(pairs, ds["test"].pairs)
        np.testing.assert_array_equal(labels, ds["test"].labels)
        for stage in ("train", "val", "test"):
            d = ds[stage]
            counts[f"{profile}_{stage}"] = dict(total=len(d), positives=int(d.labels.sum()))
            if stage != "test":
                arrays[f"{profile}_{stage}_pairs"] = d.pairs
                arrays[f"{profile}_{stage}_labels"] = d.labels
    archive(base / "training_queries.npz", **arrays)
    json_file(base / "training_counts.json", dict(counts=counts, sha256=digest(base / "training_queries.npz")))
    src, dst = pairs.T
    classes = (("positive", labels == 1), ("negative", labels == 0))
    forbidden = {tuple(sorted(e)) for e in np.vstack((split.test_pos, split.test_neg, split.val_neg))}
    clean = set(map(tuple, np.load(base / "clean.npz")["edges"]))
    for name, receipt in manifest["conditions"].items():
        path = base / f"{name}.npz"
        if digest(path) != receipt["file_sha256"]:
            raise ValueError(f"changed graph: {path}")
        with np.load(path) as view:
            edges, missing = view["edges"], view["missing"]
        es = set(map(tuple, edges))
        added, deleted = len(es - clean), len(clean - es)
        legacy = name.startswith("rq3_")
        assert (added, deleted) == (receipt["visible_inserted"], receipt["visible_deleted"])
        if not legacy:
            assert not es.intersection(forbidden)
            assert added + deleted == round(len(clean) * receipt["requested_edge_ratio"])
        assert len(np.unique(missing)) == round(raw.num_nodes * receipt["requested_feature_ratio"])
        if name.startswith("joint_"):
            with np.load(base / f"{name.removeprefix('joint_')}.npz") as structure:
                np.testing.assert_array_equal(edges, structure["edges"])
            with np.load(base / f"feature_mask_{name.rsplit('_',1)[1]}.npz") as features:
                np.testing.assert_array_equal(missing, features["missing"])
        u, v = edges.T
        a = sparse.csr_matrix((np.ones(2*len(u), dtype=np.int32),
                               (np.r_[u,v], np.r_[v,u])), shape=(raw.num_nodes, raw.num_nodes))
        deg = np.asarray(a.sum(axis=1)).ravel()
        cn = np.asarray(a[src].multiply(a[dst]).sum(axis=1)).ravel()
        _, group = connected_components(a, directed=False)
        row = dict(dataset=dataset, seed=seed, condition=name, nodes=raw.num_nodes,
                   clean_edges=len(clean), visible_edges=len(edges), deleted=deleted, inserted=added,
                   masked_nodes=len(missing), isolated_fraction=np.mean(deg == 0),
                   test_neg_in_context=receipt["test_neg_in_context"])
        for category, take in classes:
            row.update({f"{category}_endpoint_min_degree": deg[pairs[take]].min(1).mean(),
                        f"{category}_common_neighbors": cn[take].mean(),
                        f"{category}_isolated_endpoint_fraction": (deg[pairs[take]].min(1)==0).mean(),
                        f"{category}_disconnected_fraction": (group[src[take]]!=group[dst[take]]).mean()})
        rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/paired-reproduction"))
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", nargs="+", default=["cora","citeseer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [row for d in args.datasets for s in args.seeds
            for row in audit_dataset(d, s, args.root, args.data_root)]
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
