"""PubMed extension of the peer protocol, through Junxiong's native evaluate() and the published GPU EA caches."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components

from evaluate_peer_protocol import REVISION, FrozenPredictor, bind_peer
from sgdiff.data.graph import GraphData
from sgdiff.engine.baseline_trainer import _edge_index, node_feature_matrix
from sgdiff.engine.factory import load_split
from sgdiff.engine.reproduction import (
    archive,
    checkpoint_path,
    digest as file_digest,
    expanded,
    restore,
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


class PubMedPredictor(FrozenPredictor):
    """Expose model-only scoring to the peer evaluator's context/score interface."""

    def __init__(self, model, is_path_model, pairs, paths, digest):
        super().__init__(model)
        self.is_path_model = is_path_model
        self.pairs = pairs
        self.paths = paths
        self.digest = digest
        # Katz shares source-node walk counts across the evaluator's 512-pair batches.
        self.n = 19717
        query_keys = pairs[:, 0] * self.n + pairs[:, 1]
        self.order = np.argsort(query_keys)
        self.sorted_keys = query_keys[self.order]

    @torch.no_grad()
    def context(self, graph):
        if not self.is_path_model:
            return self.model.scores(graph, self.pairs)[self.order]
        key = self.digest(graph.edges)
        if key not in self.paths:
            self.paths[key] = fixed_query_paths(graph, self.pairs)
        self.model.store = self.paths[key]
        x = torch.from_numpy(node_feature_matrix(graph))
        return self.model.encode(x, _edge_index(graph, "cpu"), graph.num_nodes)

    def score(self, context, pairs):
        if self.is_path_model:
            return self.model.score_pairs(context, pairs)
        keys = pairs[:, 0].numpy() * self.n + pairs[:, 1].numpy()
        idx = np.searchsorted(self.sorted_keys, keys)
        np.testing.assert_array_equal(self.sorted_keys[idx], keys)
        return torch.from_numpy(context[idx])


def stage_ea(cache_dir, target, seed, base_hash, manifest):
    for ratio in (0.1, 0.25, 0.5):
        name = f"pubmed_s{seed}_ea_{ratio:g}.npz"
        src = cache_dir / name
        expected = manifest[name]["sha256"]
        if file_digest(src) != expected:
            raise ValueError(f"EA checksum mismatch for {src}")
        with np.load(src) as cache:
            if str(cache["base_sha256"]) != base_hash:
                raise ValueError(f"EA clean graph mismatch for {src}")
        dest = target / name
        if dest.exists():
            if file_digest(dest) != expected:
                raise ValueError(f"Conflicting cached EA graph at {dest}")
        else:
            copyfile(src, dest)


def evaluate_pubmed(a):
    protocol, native = bind_peer(a.peer)
    digest, split_data, metrics = protocol.digest, protocol.split_data, protocol.metrics
    evaluator_hash = file_digest(Path(native.__file__))
    budgets = json.loads((a.ea_cache / "manifest.json").read_text())
    manifest = {row["file"]: row for row in budgets["graphs"]}
    expected_cache_files = {f"pubmed_s{s}_ea_{r:g}.npz" for s in range(3) for r in (0.1, 0.25, 0.5)}
    if set(manifest) != expected_cache_files or budgets["peer_commit"] != REVISION:
        raise ValueError("EA cache manifest does not match the peer protocol")
    a.tables.mkdir(parents=True, exist_ok=True)
    attacks = a.output / "attacks"
    attacks.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rows, sources = [], []
    for seed in range(3):
        cfg = settings("pubmed", seed, "reference", str(a.data_root), "cpu")
        graph, original = load_split(cfg)
        split, fingerprint = split_data(graph, seed, a.output)
        for k in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            np.testing.assert_array_equal(getattr(original, k), getattr(split, k))
        base = GraphData("pubmed", graph.num_nodes, np.vstack((split.train_pos, split.val_pos)))
        stage_ea(a.ea_cache, attacks, seed, digest(base.edges), manifest)
        pairs = np.vstack((split.test_pos, split.test_neg))
        out = a.output / f"pubmed_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        archive(out / "queries.npz", pairs=pairs, positive_count=len(split.test_pos))
        paths, by_condition = {}, defaultdict(dict)
        for name in expanded(("katz", "sp4lp")):
            family = "sp4lp" if name.startswith("sp4lp") else "katz"
            ckpt = checkpoint_path(a.weights, "pubmed", seed, name)
            model = restore(name, cfg, graph, ckpt, "cpu", None)
            adapter = PubMedPredictor(model, family == "sp4lp", pairs, paths, digest)
            run = a.output / "runs" / f"pubmed_{name}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            source = dict(
                dataset="pubmed",
                seed=seed,
                model=name,
                peer_commit=REVISION,
                evaluator="experiments.new_link_models.run.evaluate",
                evaluator_sha256=evaluator_hash,
                ea_manifest_sha256=file_digest(a.ea_cache / "manifest.json"),
                split=fingerprint,
                weights=file_digest(ckpt) if family == "sp4lp" else "parameter-free",
                positive=digest(split.test_pos),
                negative=digest(split.test_neg),
                features=digest(graph.x),
                classes=digest(graph.y),
            )
            receipt = run / "receipt.json"
            if any(run.glob("evaluation_*.json")) and (
                not receipt.exists() or json.loads(receipt.read_text()) != source
            ):
                raise ValueError(f"Conflicting evaluation receipt at {run}")
            receipt.write_text(json.dumps(source, indent=2) + "\n")
            sources.append(source)
            args = SimpleNamespace(
                dataset="pubmed",
                seed=seed,
                model=family,
                variant=name,
                output=a.output,
                eval_batch_size=512,
            )
            records = native.evaluate(adapter, graph, split, args, run)
            assert len(records) == len(native.variants()) == 14
            for r in records:
                attack, ratio = r["attack"], r["ratio"]
                tag = attack if attack in ("clean", "feature_shuffle") else f"{attack}_{ratio:g}"
                with np.load(run / f"predictions_{tag}.npz") as pred:
                    pos, neg = pred["positive"], pred["negative"]
                assert len(pos) == len(split.test_pos) and len(neg) == len(split.test_neg)
                assert metrics(pos, neg) == r["metrics"]
                edge_hash = r["statistics"]["graph_sha256"]
                if tag.startswith("ea_"):
                    assert edge_hash == manifest[f"pubmed_s{seed}_{tag}.npz"]["graph_sha256"]
                if not by_condition[tag]:
                    perm = np.empty(0, dtype=int)
                    if attack == "feature_shuffle":
                        edges = base.edges
                        perm = np.random.default_rng(seed + 7183).permutation(graph.num_nodes)
                    else:
                        with np.load(attacks / f"pubmed_s{seed}_{tag}.npz") as cached:
                            edges = cached["edges"]
                    assert digest(edges) == edge_hash
                    archive(out / f"graph_{tag}.npz", edges=edges, permutation=perm)
                by_condition[tag][name] = np.r_[pos, neg]
                rows.append(
                    dict(
                        dataset="pubmed",
                        seed=seed,
                        family=family,
                        model=name,
                        condition=tag,
                        positives=len(pos),
                        negatives=len(neg),
                        graph_sha256=edge_hash,
                        **{k: r["metrics"][k] for k in ("auc", "ap", "hits100", "precision100")},
                    )
                )
        for tag, pred in by_condition.items():
            archive(out / f"scores_{tag}.npz", **pred)
    with (a.tables / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (a.tables / "provenance.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ea-cache", type=Path, required=True)
    evaluate_pubmed(p.parse_args())
