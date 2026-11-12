#!/usr/bin/env python
"""Deep attribution: which *layer*, which *operator*, and can it be repaired?

``scripts/run_probe.py`` localises the damage to an input channel (attributes vs
structure vs evidence supply).  This script goes one level down and asks three
follow-up questions, each with a falsifiable measurement:

1. **Which layer?**  With the weights frozen, every message-passing layer is
   activated on the clean graph and on each probe graph.  Two numbers per layer:
   the cosine drift of the representation (how far it moved) and the AUC of a
   *linear probe trained on the clean layer* (how much link signal is still
   linearly readable there).  If a probe's damage is already visible in layer 1,
   the encoder never got the information; if only layer 2 collapses, the damage
   enters through the second hop.
2. **Which operator?**  The same trained weights are run with the propagation
   operator swapped: symmetric ``1/√(d_i d_j)`` (as trained), row-mean
   ``1/d_i``, unnormalised sum, and symmetric without self-loops.  This tests the
   "normalisation absorbs the perturbation" claim directly instead of inferring
   it from the architecture.
3. **Which *kind* of edge?**  Deletion is split by triadic closure: edges whose
   endpoints share a neighbour (``triadic``, the CN evidence) versus edges that
   close no triangle (``bridging``, reachability only).  Same 50 % budget each.

Then it tries to *repair* the models, i.e. turns the attribution into a causal
test:

* ``complete_cn`` / ``complete_aa``: after deletion, add the top-k missing edges
  ranked by a common-neighbour prior (k = number of deleted edges).  If the
  sub-graph family is really starved of evidence, this should restore most of it;
  if the GCN family does not read CN, it should stay flat.
* ``smooth_features``: after ``feature_shuffle``, rebuild features from the
  structure (neighbour averaging).  If the GCN family's damage is the attribute
  channel, this should restore part of it.
* ``hybrid_cn``: add a CN prior to the frozen model's scores.  This measures how
  much of the loss is exactly "the CN signal the model failed to keep".

Protocol matches ``run_probe.py``: one clean fit per model, then every variant on
frozen weights, identical scored pairs, test positives withheld.

    python scripts/run_deep_probe.py --dataset cora --profile cpu --seeds 0
    python scripts/run_deep_probe.py --dataset usair --profile cpu --seeds 0
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.graph import GraphData  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.models.baselines import node_pair as node_pair_module  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

VARIANT_LABEL = {
    "clean": "clean",
    "del@50%": "del",
    "del_hub@50%": "del_hub",
    "ins@50%": "ins",
    "ins_hub@50%": "ins_hub",
    "ins_homo@50%": "ins_homo",
    "ins_hetero@50%": "ins_hetero",
    "feature_shuffle": "feat_shuffle",
    "del_tri@50%": "del_tri",
    "del_bridge@50%": "del_bridge",
}
OPERATORS = ("sym", "row", "sum", "noloop")


# --------------------------------------------------------------------------- #
# Graph surgery: edge kinds, completion, feature smoothing
# --------------------------------------------------------------------------- #
def _keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.minimum(e[:, 0], e[:, 1]) * num_nodes + np.maximum(e[:, 0], e[:, 1])


def edge_kinds(graph) -> dict[str, np.ndarray]:
    """Split the edges into triadic (CN > 0) and bridging (CN == 0)."""
    adj = graph.adjacency().tocsr()
    two_hop = (adj @ adj).tocsr()
    edges = np.asarray(graph.edges, dtype=np.int64)
    cn = np.asarray(two_hop[edges[:, 0], edges[:, 1]]).ravel()
    return {"triadic": edges[cn > 0], "bridging": edges[cn == 0]}


def delete_edges(graph, edges: np.ndarray, keep_ratio: float, seed: int = 0) -> GraphData:
    """Delete ``1 - keep_ratio`` of a given edge subset (the rest is untouched)."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    drop_n = int(round((1.0 - keep_ratio) * len(edges)))
    if drop_n == 0:
        return graph
    drop = edges[rng.choice(len(edges), size=drop_n, replace=False)]
    drop_keys = np.sort(_keys(drop, graph.num_nodes))
    full = np.asarray(graph.edges, dtype=np.int64)
    keep = full[~np.isin(_keys(full, graph.num_nodes), drop_keys)]
    return GraphData(
        name=f"{graph.name}-del-{int(round((1 - keep_ratio) * 100))}",
        num_nodes=graph.num_nodes,
        edges=keep,
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "deep": "delete_subset"},
    )


def complete_edges(graph, budget: int, mode: str = "cn", forbidden: set[int] | None = None) -> GraphData:
    """Add the ``budget`` highest-scoring missing edges under a CN / AA prior."""
    if budget <= 0:
        return graph
    adj = graph.adjacency().tocsr().astype(np.float32).tocoo()
    adj = sp.csr_matrix((np.ones_like(adj.data), (adj.row, adj.col)), shape=adj.shape)
    if mode == "aa":
        deg = np.asarray(adj.sum(1)).ravel()
        weight = 1.0 / np.log(np.maximum(deg, 2.0) + 1.0)
        b = sp.diags(np.sqrt(weight)) @ adj
        score_mat = (b @ b.T).tocsr()
    else:
        score_mat = (adj @ adj).tocsr()

    existing = set(_keys(np.asarray(graph.edges), graph.num_nodes).tolist())
    forbidden = forbidden or set()
    coo = score_mat.tocoo()
    upper = coo.row < coo.col
    rows, cols = coo.row[upper], coo.col[upper]
    scores = coo.data[upper]
    keys = np.minimum(rows, cols) * graph.num_nodes + np.maximum(rows, cols)
    keep = np.array([k not in existing and k not in forbidden for k in keys], dtype=bool)
    rows, cols, scores, keys = rows[keep], cols[keep], scores[keep], keys[keep]
    if scores.size == 0:
        return graph
    order = np.argsort(-scores, kind="stable")[:budget]
    new_edges = np.stack([np.minimum(rows[order], cols[order]), np.maximum(rows[order], cols[order])], axis=1)
    return GraphData(
        name=f"{graph.name}-complete-{mode}",
        num_nodes=graph.num_nodes,
        edges=np.vstack([np.asarray(graph.edges, dtype=np.int64), new_edges.astype(np.int64)]),
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "deep": f"complete_{mode}"},
    )


def smooth_features(graph, rounds: int = 2, alpha: float = 0.5) -> GraphData:
    """Rebuild node features from the structure: ``X ← (1-α)X + α P X``."""
    if not graph.has_features:
        raise ValueError("smooth_features needs node features")
    adj = graph.adjacency().tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    prop = sp.diags(1.0 / deg) @ adj
    x = np.asarray(graph.x, dtype=np.float32)
    for _ in range(rounds):
        x = (1.0 - alpha) * x + alpha * (prop @ x)
    return GraphData(
        name=f"{graph.name}-smooth{rounds}",
        num_nodes=graph.num_nodes,
        edges=np.asarray(graph.edges),
        x=x,
        y=graph.y,
        meta={**graph.meta, "deep": "smooth_features"},
    )


# --------------------------------------------------------------------------- #
# Layer-level measurements
# --------------------------------------------------------------------------- #
def layer_modules(model) -> list:
    encoder = getattr(model, "encoder", None)
    return list(getattr(encoder, "lins", []) or [])


def layer_activations(model, x_t: torch.Tensor, edge_index, num_nodes: int, mods: list) -> list[np.ndarray]:
    """Per-layer output on one graph (forward hooks; no change to the model)."""
    if not mods:
        return []
    captured: list[torch.Tensor | None] = [None] * len(mods)
    handles = [
        mod.register_forward_hook(lambda _m, _i, out, idx=i: captured.__setitem__(idx, out.detach()))
        for i, mod in enumerate(mods)
    ]
    try:
        with torch.no_grad():
            model.encode(x_t, edge_index, num_nodes)
    finally:
        for handle in handles:
            handle.remove()
    return [t.float().cpu().numpy() if t is not None else None for t in captured]


def _unit(z: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norm, 1e-8)


def ridge_probe(z: np.ndarray, train_pairs: np.ndarray, y_train: np.ndarray, eval_pairs: np.ndarray, lam: float = 1.0):
    """Linear read-out on the elementwise product ``z_u ⊙ z_v`` (a mini decoder)."""
    z = _unit(np.asarray(z, dtype=np.float64))
    prod = z[train_pairs[:, 0]] * z[train_pairs[:, 1]]
    mu, sigma = prod.mean(0), prod.std(0) + 1e-8
    zs = (prod - mu) / sigma
    gram = zs.T @ zs + lam * np.eye(zs.shape[1])
    w = np.linalg.solve(gram, zs.T @ (y_train - y_train.mean()))
    zc = (_unit(z[eval_pairs[:, 0]] * z[eval_pairs[:, 1]]) - mu) / sigma
    return zc @ w


def sample_train_pairs(graph, split, seed: int, limit: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Train pairs for the linear probe: given positives + random negatives."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(split.train_pos, dtype=np.int64).reshape(-1, 2)
    if len(pos) > limit:
        pos = pos[rng.choice(len(pos), size=limit, replace=False)]
    existing = set(_keys(np.asarray(graph.edges), graph.num_nodes).tolist())
    neg = []
    while len(neg) < len(pos):
        u, v = rng.integers(0, graph.num_nodes, size=2)
        if u == v:
            continue
        key = int(min(u, v) * graph.num_nodes + max(u, v))
        if key in existing:
            continue
        existing.add(key)
        neg.append((int(min(u, v)), int(max(u, v))))
    pairs = np.vstack([pos, np.asarray(neg, dtype=np.int64)])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return pairs, labels


# --------------------------------------------------------------------------- #
# Operator substitution: same weights, different propagation
# --------------------------------------------------------------------------- #
@contextmanager
def propagation(mode: str):
    """Swap ``sparse_normalize`` for the duration of one forward pass."""
    original = node_pair_module.sparse_normalize

    def patched(edge_index, num_nodes, add_self_loops: bool = True):
        return _normalize(edge_index, num_nodes, add_self_loops, mode)

    node_pair_module.sparse_normalize = patched
    try:
        yield
    finally:
        node_pair_module.sparse_normalize = original


def _normalize(edge_index, num_nodes: int, add_self_loops: bool, mode: str):
    row, col = edge_index[0], edge_index[1]
    if mode == "noloop":
        add_self_loops = False
    if add_self_loops:
        loop = torch.arange(num_nodes, device=row.device)
        row = torch.cat([row, loop])
        col = torch.cat([col, loop])
    ones = torch.ones(row.numel(), device=row.device, dtype=torch.float32)
    deg = torch.zeros(num_nodes, device=row.device).index_add_(0, row, ones).clamp(min=1.0)
    if mode in ("sym", "noloop"):
        weights = deg.pow(-0.5)[row] * deg.pow(-0.5)[col]
    elif mode == "row":
        weights = deg.pow(-1.0)[row]
    else:  # sum: no normalisation at all
        weights = ones
    return row, col, weights


def score_with_operator(kind: str, model, cfg, graph, pairs, device, seed: int, features, mode: str):
    with propagation(mode):
        return rp.score_model(kind, model, cfg, graph, pairs, device, seed, features=features)


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def _score(kind, model, cfg, graph, pairs, device, seed, features) -> np.ndarray:
    return rp.score_model(kind, model, cfg, graph, pairs, device, seed, features=features)


def run(cfg, models, seeds, deep_variants, device="cpu", max_pairs=500, epochs=None):
    epochs = epochs or {}
    registry = rp.build_registry(cfg)
    layers_raw: dict[tuple[str, str, int], list[dict]] = {}
    ops_raw: dict[tuple[str, str, int, str], dict] = {}
    kinds_raw: dict[tuple[str, str, int], dict] = {}
    repairs_raw: dict[tuple[str, str, int], dict] = {}
    base_raw: dict[tuple[str, str, int], dict] = {}
    refs_raw: dict[tuple[str, str, int], dict] = {}
    defence_raw: dict[tuple[str, str, int], dict] = {}
    skipped: list[dict] = []

    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        cfg_seed.train.device = str(device)
        set_seed(int(seed))

        graph, split = rp.load_split(cfg_seed)
        datasets = rp.prepare_datasets(graph, split, cfg_seed)
        pos, neg = rp._cap_pairs(split, max_pairs)
        pairs = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()
        forbidden = {int(min(u, v) * graph.num_nodes + max(u, v)) for u, v in np.asarray(pairs)}
        clean = graph.drop_edges(test_edges)
        clean_features = rp.node_feature_matrix(clean)
        probe_train_pairs, probe_train_labels = sample_train_pairs(graph, split, int(seed))

        graphs: dict[str, GraphData] = {"clean": clean}
        for variant in deep_variants:
            try:
                graphs[variant] = PERTURBATIONS[variant](graph, int(seed)).drop_edges(test_edges)
            except ValueError as exc:
                skipped.append({"seed": int(seed), "variant": variant, "reason": str(exc)})

        # ---- edge-kind deletion: same 50 % budget, different subpopulation
        kinds = edge_kinds(clean)
        kind_graphs = {
            "del_tri@50%": delete_edges(clean, kinds["triadic"], 0.5, int(seed) + 1),
            "del_bridge@50%": delete_edges(clean, kinds["bridging"], 0.5, int(seed) + 2),
        }
        graphs.update(kind_graphs)

        # ---- repairs on the inference graph (no retraining, no test leakage)
        repair_graphs: dict[str, GraphData] = {}
        for variant in ("del@50%", "del_hub@50%"):
            if variant not in graphs:
                continue
            removed = clean.num_edges - graphs[variant].num_edges
            for mode in ("cn", "aa"):
                repair_graphs[f"{variant}|complete_{mode}"] = complete_edges(
                    graphs[variant], removed, mode, forbidden
                )
        if "feature_shuffle" in graphs:
            for rounds in (1, 2):
                repair_graphs[f"feature_shuffle|smooth{rounds}"] = smooth_features(graphs["feature_shuffle"], rounds)

        for name in models:
            spec = registry[name]
            started = time.perf_counter()
            set_seed(int(seed))
            model, _metrics, _params = rp.fit_model(
                spec,
                cfg_seed,
                graph,
                split,
                datasets,
                int(seed),
                sgdiff_epochs=epochs.get("sgdiff"),
                subgraph_epochs=epochs.get("subgraph", 40),
                node_pair_epochs=epochs.get("node_pair", 300),
                device=device,
                run_root=f"runs/deep/{cfg.data.name}",
            )
            model.eval()
            mods = layer_modules(model) if spec.kind in rp.NODE_PAIR_KINDS else []
            base: dict[str, float] = {}

            def feats_for(variant: str):
                if variant.startswith("feature_shuffle") and spec.kind in rp.NODE_PAIR_KINDS:
                    return rp.node_feature_matrix(graphs[variant])
                return clean_features

            # ---- 1) layer probes
            if mods:
                clean_acts = None
                for variant, eval_graph in graphs.items():
                    if variant not in ("clean",) + tuple(deep_variants):
                        continue
                    x = feats_for(variant)
                    x_t = torch.as_tensor(x, device=device, dtype=torch.float32)
                    edge_index = rp._edge_index(eval_graph, device)
                    torch.manual_seed(int(seed))
                    acts = layer_activations(model, x_t, edge_index, eval_graph.num_nodes, mods)
                    if variant == "clean":
                        clean_acts = acts
                        base_auc = summarize_scores(labels, _score(spec.kind, model, cfg_seed, eval_graph, pairs, device, int(seed), x))["auc"]
                        base["clean"] = float(base_auc)
                        continue
                    cells = []
                    for index, (act, ref) in enumerate(zip(acts, clean_acts or [])):
                        if act is None or ref is None:
                            continue
                        drift = 1.0 - float(np.mean(np.sum(_unit(act) * _unit(ref), axis=1)))
                        scores = ridge_probe(act, probe_train_pairs, probe_train_labels, pairs)
                        cells.append(
                            {
                                "layer": index + 1,
                                "drift": drift,
                                "probe_auc": float(summarize_scores(labels, scores)["auc"]),
                            }
                        )
                    layers_raw.setdefault((name, variant, int(seed)), []).extend(
                        [{"model": name, "variant": variant, "seed": int(seed), **cell} for cell in cells]
                    )

            # ---- 2) operator substitution (works without hooks: VGAE has one layer)
            if spec.kind in rp.NODE_PAIR_KINDS and name in ("gcn", "sage", "vgae"):
                for variant in ("clean", "del@50%", "ins_hub@50%", "feature_shuffle"):
                    if variant not in graphs:
                        continue
                    eval_graph = graphs[variant]
                    x = feats_for(variant)
                    for mode in OPERATORS:
                        try:
                            scores = score_with_operator(
                                spec.kind, model, cfg_seed, eval_graph, pairs, device, int(seed), x, mode
                            )
                        except Exception as exc:  # a shape mismatch must not kill the run
                            skipped.append({"seed": int(seed), "variant": f"{variant}|{mode}", "reason": str(exc)[:80]})
                            continue
                        ops_raw[(name, variant, int(seed), mode)] = {
                            "auc": float(summarize_scores(labels, scores)["auc"])
                        }

            # ---- 3) edge-kind deletion on all models
            for variant in ("del_tri@50%", "del_bridge@50%"):
                eval_graph = graphs[variant]
                scores = _score(spec.kind, model, cfg_seed, eval_graph, pairs, device, int(seed), clean_features)
                base_metric = float(summarize_scores(labels, scores)["auc"])
                kinds_raw.setdefault((name, variant, int(seed)), {})["auc"] = base_metric
                kinds_raw[(name, variant, int(seed))]["edges_before"] = len(kinds["triadic" if "tri" in variant else "bridging"])
                kinds_raw[(name, variant, int(seed))]["edges_after"] = eval_graph.num_edges

            # ---- 4) repairs
            if "clean" not in base:
                base["clean"] = float(
                    summarize_scores(labels, _score(spec.kind, model, cfg_seed, clean, pairs, device, int(seed), clean_features))["auc"]
                )
            base_raw.setdefault((name, "clean", int(seed)), {})["auc"] = base["clean"]
            for variant in deep_variants:
                if variant not in graphs or variant == "feature_shuffle":
                    continue
                base_metric = float(
                    summarize_scores(labels, _score(spec.kind, model, cfg_seed, graphs[variant], pairs, device, int(seed), clean_features))["auc"]
                )
                base_raw.setdefault((name, variant, int(seed)), {})["auc"] = base_metric
            for key, eval_graph in repair_graphs.items():
                variant, repair = key.split("|")
                features = clean_features
                if variant.startswith("feature_shuffle") and spec.kind in rp.NODE_PAIR_KINDS:
                    features = rp.node_feature_matrix(eval_graph)
                scores = _score(spec.kind, model, cfg_seed, eval_graph, pairs, device, int(seed), features)
                repairs_raw.setdefault((name, key, int(seed)), {})["auc"] = float(
                    summarize_scores(labels, scores)["auc"]
                )
            if "feature_shuffle" in graphs:
                base_raw.setdefault((name, "feature_shuffle", int(seed)), {})["auc"] = float(
                    summarize_scores(
                        labels,
                        _score(spec.kind, model, cfg_seed, graphs["feature_shuffle"], pairs, device, int(seed),
                               feats_for("feature_shuffle")),
                    )["auc"]
                )

            # ---- 5) hybrid: frozen model score + a CN prior
            if "del@50%" in graphs:
                model_scores = _score(spec.kind, model, cfg_seed, graphs["del@50%"], pairs, device, int(seed), clean_features)
                heur = rp.heuristic_scores(
                    graphs["del@50%"], pairs, clean_features if graphs["del@50%"].has_features else None
                )
                for key in ("cn", "aa", "deg"):
                    if key in heur:
                        refs_raw.setdefault((key, "del@50%", int(seed)), {})["auc"] = float(
                            summarize_scores(labels, heur[key])["auc"]
                        )
                z_model = (model_scores - model_scores.mean()) / (model_scores.std() + 1e-9)
                for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
                    cn = heur["cn"]
                    z_cn = (cn - cn.mean()) / (cn.std() + 1e-9)
                    auc = float(summarize_scores(labels, z_model + lam * z_cn)["auc"])
                    repairs_raw.setdefault((name, f"del@50%|hybrid_cn{lam:g}", int(seed)), {})["auc"] = auc

            # ---- 6) model-side repair: richer sub-graphs (k = 2) for the sub-graph family
            if spec.kind == "subgraph":
                cfg_deep = copy.deepcopy(cfg_seed)
                if hasattr(cfg_deep, "subgraph"):
                    cfg_deep.subgraph.hops = 2
                datasets_k2 = rp.prepare_datasets(graph, split, cfg_deep)
                set_seed(int(seed))
                model_k2, _m, _p = rp.fit_model(
                    spec,
                    cfg_deep,
                    graph,
                    split,
                    datasets_k2,
                    int(seed),
                    subgraph_epochs=epochs.get("subgraph", 40),
                    device=device,
                    run_root=f"runs/deep/{cfg.data.name}/k2",
                )
                model_k2.eval()
                for variant in ("clean", "del@50%", "del_hub@50%", "ins_hub@50%"):
                    if variant not in graphs:
                        continue
                    scores = _score(spec.kind, model_k2, cfg_deep, graphs[variant], pairs, device, int(seed), clean_features)
                    defence_raw.setdefault((name, variant, int(seed)), {})["auc"] = float(
                        summarize_scores(labels, scores)["auc"]
                    )

            print(
                f"[{cfg.data.name} seed={seed}] {name}: clean={base.get('clean'):.3f} "
                f"fit={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    return _summarise(
        cfg, models, seeds, layers_raw, ops_raw, kinds_raw, repairs_raw, base_raw, refs_raw, defence_raw, skipped, deep_variants
    )


def _agg(entries: list[dict], key: str) -> dict:
    values = [e[key] for e in entries if e.get(key) is not None and np.isfinite(e[key])]
    if not values:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def _summarise(
    cfg, models, seeds, layers_raw, ops_raw, kinds_raw, repairs_raw, base_raw, refs_raw, defence_raw, skipped, deep_variants
) -> dict:
    layers: dict[str, dict[str, dict[str, dict]]] = {}
    for (name, variant, seed), entries in layers_raw.items():
        for entry in entries:
            layers.setdefault(name, {}).setdefault(variant, {}).setdefault(f"L{entry['layer']}", {})
            bucket = layers[name][variant][f"L{entry['layer']}"]
            for key in ("drift", "probe_auc"):
                bucket.setdefault(key, []).append(entry[key])
    layer_summary = {
        model: {
            variant: {layer: {k: _agg([{"v": x} for x in vals], "v") for k, vals in per.items()} for layer, per in by_variant.items()}
            for variant, by_variant in per_model.items()
        }
        for model, per_model in layers.items()
    }

    operators: dict[str, dict[str, dict[str, dict]]] = {}
    for (name, variant, seed, mode), payload in ops_raw.items():
        operators.setdefault(name, {}).setdefault(mode, {}).setdefault(variant, []).append(payload["auc"])
    operator_summary = {
        model: {mode: {v: _agg([{"v": x} for x in vals], "v") for v, vals in by_variant.items()} for mode, by_variant in per.items()}
        for model, per in operators.items()
    }

    kinds: dict[str, dict[str, dict]] = {}
    for (name, variant, seed), payload in kinds_raw.items():
        kinds.setdefault(name, {}).setdefault(variant, []).append(payload)
    kind_summary = {
        model: {
            variant: {k: _agg(entries, k) for k in ("auc", "edges_before", "edges_after")} for variant, entries in per.items()
        }
        for model, per in kinds.items()
    }

    repairs: dict[str, dict[str, dict]] = {}
    for (name, key, seed), payload in repairs_raw.items():
        repairs.setdefault(name, {}).setdefault(key, []).append(payload["auc"])
    repair_summary = {
        model: {key: _agg([{"v": x} for x in vals], "v") for key, vals in per.items()} for model, per in repairs.items()
    }

    base: dict[str, dict[str, dict]] = {}
    for (name, variant, seed), payload in base_raw.items():
        base.setdefault(name, {}).setdefault(variant, []).append(payload["auc"])
    base_summary = {
        model: {variant: _agg([{"v": x} for x in vals], "v") for variant, vals in per.items()}
        for model, per in base.items()
    }

    refs: dict[str, dict[str, dict]] = {}
    for (key, variant, seed), payload in refs_raw.items():
        refs.setdefault(variant, {}).setdefault(key, []).append(payload["auc"])
    ref_summary = {
        variant: {key: _agg([{"v": x} for x in vals], "v") for key, vals in per.items()}
        for variant, per in refs.items()
    }

    defence: dict[str, dict[str, dict]] = {}
    for (name, variant, seed), payload in defence_raw.items():
        defence.setdefault(name, {}).setdefault(variant, []).append(payload["auc"])
    defence_summary = {
        model: {variant: _agg([{"v": x} for x in vals], "v") for variant, vals in per.items()}
        for model, per in defence.items()
    }

    return {
        "dataset": cfg.data.name,
        "models": models,
        "seeds": [int(s) for s in seeds],
        "variants": list(deep_variants),
        "base": base_summary,
        "layers": layer_summary,
        "operators": operator_summary,
        "edge_kinds": kind_summary,
        "repairs": repair_summary,
        "refs": ref_summary,
        "defence": defence_summary,
        "skipped": skipped,
        "protocol": {
            "setting": "evasion; one clean fit per model, all measurements on frozen weights",
            "layers": "linear probe trained on the clean layer activations, evaluated on every probe graph",
            "operators": "same weights, propagation operator swapped (gcn/sage/vgae)",
            "repairs": "inference-time only, test pairs excluded from the completion candidates",
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _f(entry, digits=3, signed=False):
    if not entry or not entry.get("n") or not np.isfinite(entry["mean"]):
        return "-"
    return f"{entry['mean']:+.{digits}f}" if signed else f"{entry['mean']:.{digits}f}"


def to_markdown(result: dict) -> str:
    lines = [f"# Deep probe: {result['dataset']}", ""]
    lines += [
        "* models: " + ", ".join(result["models"]),
        "* seeds: " + str(result["seeds"]),
        "* variants: " + ", ".join(VARIANT_LABEL.get(v, v) for v in result["variants"]),
        "",
        "## 1. Layer-wise damage (node-pair models)",
        "",
        "`drift` = 1 − mean cosine between the layer representation on the probe graph and",
        "on the clean graph; `probe_auc` = AUC of a linear read-out trained on the *clean*",
        "layer and applied to the probe layer (chance ≈ 0.5).",
        "",
    ]
    for model, per_variant in result["layers"].items():
        layers = sorted({layer for by_variant in per_variant.values() for layer in by_variant})
        if not layers:
            continue
        lines += [f"### {model}", ""]
        header = ["probe"] + [f"{layer} drift" for layer in layers] + [f"{layer} AUC" for layer in layers]
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for variant, by_layer in per_variant.items():
            row = [VARIANT_LABEL.get(variant, variant)]
            row += [_f(by_layer.get(layer, {}).get("drift")) for layer in layers]
            row += [_f(by_layer.get(layer, {}).get("probe_auc")) for layer in layers]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if result["operators"]:
        lines += ["## 2. Same weights, different propagation operator", ""]
        lines += ["| model | operator | " + " | ".join(sorted({v for m in result["operators"].values() for v in m["sym"]})) + " |"]
        variants = sorted({v for m in result["operators"].values() for v in m["sym"]})
        lines += ["|" + "|".join(["---"] * (len(variants) + 2)) + "|"]
        for model, per_mode in result["operators"].items():
            for mode, per_variant in per_mode.items():
                row = [model, mode] + [_f(per_variant.get(v)) for v in variants]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)", ""]
    lines += ["| model | clean | del_tri | del_bridge | tri edges | bridge edges |", "|---|---|---|---|---|---|"]
    for model, per_variant in result["edge_kinds"].items():
        clean = result["base"].get(model, {}).get("clean")
        tri = per_variant.get("del_tri@50%", {})
        bridge = per_variant.get("del_bridge@50%", {})
        lines.append(
            "| " + " | ".join(
                [
                    model,
                    _f(clean),
                    _f(tri.get("auc")),
                    _f(bridge.get("auc")),
                    _f(tri.get("edges_before"), 0),
                    _f(bridge.get("edges_before"), 0),
                ]
            ) + " |"
        )
    lines.append("")

    lines += ["## 4. Repairs on the inference graph (no retraining)", ""]
    ref_del = result.get("refs", {}).get("del@50%", {})
    lines += [
        f"Reference on the damaged graph itself: CN-only AUC {_f(ref_del.get('cn'))}, "
        f"AA-only {_f(ref_del.get('aa'))} — a repair cannot be expected to beat the prior it uses.",
        "",
        "| model | probe | base | repaired | gain |",
        "|---|---|---|---|---|",
    ]
    for model, per_key in result["repairs"].items():
        for key, entry in per_key.items():
            variant, repair = key.split("|")
            base = result["base"].get(model, {}).get(variant, {})
            gain = entry["mean"] - base.get("mean", float("nan"))
            lines.append(f"| {model} | {VARIANT_LABEL.get(variant, variant)} + {repair} | {_f(base)} | {_f(entry)} | {gain:+.3f} |")
    lines.append("")

    if result.get("defence"):
        lines += [
            "## 5. Model-side repair: give the sub-graph family more evidence (k = 2)",
            "",
            "SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).",
            "If the deletion damage is evidence starvation rather than a weight-space property,",
            "the k = 2 model should lose less under the deletion probes.",
            "",
            "| model | probe | k = 1 | k = 2 | gain |",
            "|---|---|---|---|---|",
        ]
        for model, per_variant in result["defence"].items():
            base = result["base"].get(model, {})
            for variant, entry in per_variant.items():
                gain = entry["mean"] - base.get(variant, {}).get("mean", float("nan"))
                lines.append(
                    f"| {model} | {VARIANT_LABEL.get(variant, variant)} | {_f(base.get(variant))} "
                    f"| {_f(entry)} | {gain:+.3f} |"
                )
        lines.append("")
    if result.get("skipped"):
        lines += ["## Skipped", ""]
        seen: dict[str, set[str]] = {}
        for skip in result["skipped"]:
            seen.setdefault(str(skip["variant"]), set()).add(str(skip["reason"]))
        for variant, why in seen.items():
            lines.append(f"* `{variant}`: {'; '.join(sorted(why))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=rp.DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="*", default=["del@50%", "del_hub@50%", "ins_hub@50%", "feature_shuffle"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--sgdiff-epochs", type=int, default=None)
    parser.add_argument("--subgraph-epochs", type=int, default=40)
    parser.add_argument("--node-pair-epochs", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"] + rp.PROFILES[args.profile] + args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)
    device = resolve_device(args.device)

    result = run(
        cfg,
        models=args.models,
        seeds=args.seeds,
        deep_variants=args.variants,
        device=device,
        max_pairs=args.max_pairs or 500,
        epochs={
            "sgdiff": args.sgdiff_epochs,
            "subgraph": args.subgraph_epochs,
            "node_pair": args.node_pair_epochs,
        },
    )
    result["markdown"] = to_markdown(result)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.output}/deep_{cfg.data.name}{suffix}"
    save_json(result, f"{stem}.json")
    Path(f"{stem}.md").write_text(result["markdown"] + "\n", encoding="utf-8")
    print()
    print(result["markdown"])
    print(f"\nwrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
