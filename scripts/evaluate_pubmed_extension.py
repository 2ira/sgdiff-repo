"""PubMed extension of the peer protocol, using the eleven non-EA conditions."""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components

from evaluate_peer_protocol import REVISION, experiments
from sgdiff.data.graph import GraphData
from sgdiff.engine.baseline_trainer import _edge_index, node_feature_matrix
from sgdiff.engine.factory import load_split
from sgdiff.engine.reproduction import (
    archive,
    checkpoint_path,
    digest as file_digest,
    expanded,
    restore,
    training,
)
from sgdiff.engine.team_study import settings
from sgdiff.models.baselines.sp4lp import path_store
from sgdiff.vendor.sp4lp.paths import PathStore


def fixed_query_paths(graph, pairs):
    """Sorted CSR BFS matches the vendor's sorted-neighbor BFS on protected pairs."""
    adj = graph.adjacency()
    adj.sort_indices()
    if np.any(np.asarray(adj[pairs[:, 0], pairs[:, 1]])):
        raise ValueError("PubMed test candidate present in context")
    _, components = connected_components(adj, directed=False)
    by_source = defaultdict(list)
    paths, reaches = {}, {}
    for u, v in pairs:
        u, v = int(u), int(v)
        reaches[u, v] = components[u] == components[v]
        if reaches[u, v]:
            by_source[u].append(v)
        else:
            paths[u, v] = [u, v]
    for u, targets in by_source.items():
        _, parent = breadth_first_order(adj, u, directed=True)
        for v in targets:
            reverse = [v]
            while reverse[-1] != u:
                reverse.append(int(parent[reverse[-1]]))
            paths[u, v] = reverse[::-1]
    result = PathStore.from_dicts(paths, reaches, max_len=32)
    sample = pairs[np.linspace(0, len(pairs) - 1, 24, dtype=int)]
    reference = path_store(graph, sample)
    for pair in map(tuple, sample):
        i, j = result.key[pair], reference.key[pair]
        assert result.reach[i] == reference.reach[j]
        np.testing.assert_array_equal(
            result.data[result.offs[i] : result.offs[i + 1]],
            reference.data[reference.offs[j] : reference.offs[j + 1]],
        )
    return result


@torch.no_grad()
def score(model, name, graph, pairs, paths):
    if name.startswith("katz"):
        return model.scores(graph, pairs)
    model.store = paths
    x = torch.from_numpy(node_feature_matrix(graph))
    z = model.encode(x, _edge_index(graph, "cpu"), graph.num_nodes)
    return np.concatenate(
        [
            model.score_pairs(z, torch.from_numpy(pairs[i : i + 512])).numpy()
            for i in range(0, len(pairs), 512)
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    if (
        subprocess.check_output(["git", "-C", str(a.peer), "rev-parse", "HEAD"], text=True).strip()
        != REVISION
    ):
        p.error("unexpected peer version")
    experiments.__path__.append(str((a.peer / "experiments").resolve()))
    from experiments.new_link_models.protocol import digest, metrics, perturb, split_data
    from experiments.new_link_models.run import variants

    a.tables.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    names = list(expanded(("katz", "sp4lp")))
    grid = [(k, r) for k, r in variants() if k != "ea"]
    assert len(grid) == 11
    rows, sources = [], []
    for seed in a.seeds:
        cfg = settings("pubmed", seed, "reference", str(a.data_root), "cpu")
        graph, original = load_split(cfg)
        split, fingerprint = split_data(graph, seed, a.output)
        for k in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            np.testing.assert_array_equal(getattr(original, k), getattr(split, k))
        training(cfg, "pubmed", seed, ["sp4lp"], a.weights, "cpu", None)
        base = GraphData(
            "pubmed",
            graph.num_nodes,
            np.vstack((split.train_pos, split.val_pos)),
            x=graph.x,
            y=graph.y,
        )
        pairs = np.vstack((split.test_pos, split.test_neg))
        npos = len(split.test_pos)
        out = a.output / f"pubmed_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        archive(out / "queries.npz", pairs=pairs, positive_count=npos)
        models = {}
        for name in names:
            ckpt = checkpoint_path(a.weights, "pubmed", seed, name)
            models[name] = restore(name, cfg, graph, ckpt, "cpu", None)
            sources.append(
                dict(
                    dataset="pubmed",
                    seed=seed,
                    model=name,
                    peer_commit=REVISION,
                    split=fingerprint,
                    weights=file_digest(ckpt) if name.startswith("sp4lp") else "parameter-free",
                    positive=digest(split.test_pos),
                    negative=digest(split.test_neg),
                    features=digest(graph.x),
                    classes=digest(graph.y),
                )
            )
        for attack, ratio in grid:
            tag = attack if attack in ("clean", "feature_shuffle") else f"{attack}_{ratio:g}"
            if attack == "feature_shuffle":
                permutation = np.random.default_rng(seed + 7183).permutation(graph.num_nodes)
                attacked = GraphData(
                    base.name, base.num_nodes, base.edges, x=base.x[permutation], y=base.y
                )
            else:
                permutation = np.empty(0, dtype=int)
                attacked, _ = perturb(base, split, attack, ratio, seed)
            edge_hash = digest(attacked.edges)
            archive(out / f"graph_{tag}.npz", edges=attacked.edges, permutation=permutation)
            paths = fixed_query_paths(attacked, pairs)
            predictions = {}
            for name, model in models.items():
                v = score(model, name, attacked, pairs, paths)
                assert v.shape == (len(pairs),) and np.isfinite(v).all()
                predictions[name] = v
                vals = metrics(v[:npos], v[npos:])
                rows.append(
                    dict(
                        dataset="pubmed",
                        seed=seed,
                        family="sp4lp" if name.startswith("sp4lp") else "katz",
                        model=name,
                        condition=tag,
                        positives=npos,
                        negatives=len(v) - npos,
                        graph_sha256=edge_hash,
                        **{k: vals[k] for k in ("auc", "ap", "hits100", "precision100")},
                    )
                )
            archive(out / f"scores_{tag}.npz", **predictions)
            print(seed, tag, flush=True)
    with (a.tables / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (a.tables / "provenance.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
