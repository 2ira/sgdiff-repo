#!/usr/bin/env python
"""Mechanism probes: *which* graph property does a model read, and what breaks it?

``scripts/run_robustness.py`` answers "how much does model M lose under attack A".
RF / DICE / EA mix deletions and insertions, so the answer cannot be attributed to
a mechanism.  This script replaces them with the targeted probes of
:mod:`sgdiff.data.perturb` (delete / delete-hub / insert / insert-hub /
insert-homophilous / insert-heterophilous / feature shuffle) and adds three
measurements that turn "it dropped" into "it dropped because of component X":

1. **property availability** -- cheap heuristics (degree product, common
   neighbours, Adamic-Adar, Jaccard, feature cosine) are scored on the *same*
   perturbed graph.  If a perturbation leaves Adamic-Adar at AUC 0.80 but kills
   the degree product, any model that loses accuracy there was reading degrees.
2. **signal alignment** -- Spearman rho between each model's clean scores and
   every heuristic on the same pairs: what the model actually ranks by.
3. **component ablation** -- the model is left frozen and its inputs are
   ablated: node features zeroed, DRNL labels permuted inside each sub-graph,
   and (for SGDiff) each fusion branch switched off via ``eta_1`` / ``eta_2``.

Protocol is the evasion protocol of the robustness harness: every model is
trained once on the clean graph, the perturbation is applied to the inference
graph only, the test positives are withheld from that graph, and every variant
scores the same test pairs.  Featureless graphs use the harness' fixed degree
encoding, except for the probes that are *about* features.

Examples
--------
    # Cora: attributed, labelled -> all seven probes
    python scripts/run_probe.py --dataset cora --profile cpu --seeds 0 1 2

    # USAir: no attributes, no labels -> only the structural probes run
    python scripts/run_probe.py --dataset usair --profile cpu --seeds 0

    # one model, the two deletion probes
    python scripts/run_probe.py --models gcn sgdiff --variants del@50% del_hub@50%
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.engine.baseline_trainer import _edge_index, node_feature_matrix  # noqa: E402
from sgdiff.engine.batch import PairSubgraphDataset, collate_subgraphs  # noqa: E402
from sgdiff.engine.benchmark import (  # noqa: E402
    BUDGET_PROFILES as PROFILES,
    METRICS,
    build_registry,
    fit_model,
)
from sgdiff.engine.factory import build_extractor, load_split, prepare_datasets  # noqa: E402
from sgdiff.engine.robustness import _cap_pairs  # noqa: E402
from sgdiff.utils.io import load_json, save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

DEFAULT_MODELS = ["sgdiff", "seal", "pencil", "gcn", "gat", "sage", "vgae"]
DEFAULT_VARIANTS = list(PERTURBATIONS)
SUBGRAPH_KINDS = ("subgraph", "sgdiff")
NODE_PAIR_KINDS = ("node_pair", "robust_node_pair")

#: Heuristic probes: name -> function(graph, pairs) -> scores (higher = more likely a link).
#: "deg" is the degree-product (preferential attachment), the only one-sided degree
#: signal; "cn"/"aa"/"jaccard" need shared neighbours, i.e. 2-hop connectivity.
HEURISTICS = ("deg", "cn", "aa", "jaccard", "featcos")


# --------------------------------------------------------------------------- #
# Property heuristics: how much signal of each kind survives the perturbation
# --------------------------------------------------------------------------- #
def _pair_arrays(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return p[:, 0], p[:, 1]


def heuristic_scores(graph, pairs: np.ndarray, features=None) -> dict[str, np.ndarray]:
    """Score ``pairs`` with every property heuristic available on ``graph``."""
    u, v = _pair_arrays(pairs)
    adj = graph.adjacency()
    deg = graph.degrees().astype(np.float64)
    out: dict[str, np.ndarray] = {"deg": deg[u] * deg[v]}

    a2 = (adj @ adj).tocsr()
    cn = np.asarray(a2[u, v]).ravel().astype(np.float64)
    out["cn"] = cn

    union = deg[u] + deg[v] - cn
    out["jaccard"] = np.divide(cn, np.maximum(union, 1.0))

    aa = np.zeros(len(u), dtype=np.float64)
    log_deg = np.zeros_like(deg)
    log_deg[deg > 1] = 1.0 / np.log(deg[deg > 1])
    for i in range(len(u)):
        inter = adj.getrow(int(u[i])).multiply(adj.getrow(int(v[i])))
        if inter.nnz:
            aa[i] = log_deg[inter.indices].sum()
    out["aa"] = aa

    if features is not None and graph.has_features:
        f = np.asarray(features, dtype=np.float64)
        norm = np.linalg.norm(f, axis=1, keepdims=True)
        unit = f / np.maximum(norm, 1e-12)
        out["featcos"] = (unit[u] * unit[v]).sum(axis=1)
    return out


def graph_stats(graph) -> dict:
    """Structural summary of a perturbed graph (what the probe did to the graph)."""
    deg = graph.degrees().astype(np.float64)
    e = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    stats = {
        "num_edges": int(graph.num_edges),
        "mean_degree": float(deg.mean()) if deg.size else float("nan"),
        "max_degree": int(deg.max()) if deg.size else 0,
        "density": float(2.0 * graph.num_edges / max(graph.num_nodes * (graph.num_nodes - 1), 1)),
    }
    if deg.size > 1 and deg.std() > 0 and len(e):
        du, dv = deg[e[:, 0]], deg[e[:, 1]]
        # degree assortativity: Pearson correlation of the endpoint degrees
        stats["assortativity"] = float(
            np.corrcoef(du, dv)[0, 1] if du.std() > 0 and dv.std() > 0 else 0.0
        )
    else:
        stats["assortativity"] = 0.0
    if graph.y is not None and len(e):
        y = np.asarray(graph.y).ravel()
        stats["homophily"] = float(np.mean(y[e[:, 0]] == y[e[:, 1]]))
    return stats


def _connected(adj: np.ndarray, i: int, j: int) -> bool:
    """Is ``j`` reachable from ``i`` inside the extracted sub-graph?"""
    seen, stack = {i}, [i]
    while stack:
        node = stack.pop()
        if node == j:
            return True
        for nb in np.flatnonzero(adj[node] > 0):
            nb = int(nb)
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return False


STAT_KEYS = ("num_edges", "mean_degree", "assortativity", "homophily", "density")


def variant_graphs(cfg, variants, seeds):
    """Yield ``(seed, graph, split, {variant: inference graph})`` per seed.

    Every cheap pass (graph statistics, property heuristics, sub-graph
    availability) consumes this, so they describe the same graphs the models see.
    """
    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        set_seed(int(seed))
        graph, split = load_split(cfg_seed)
        test_edges = split.test_pos.tolist()
        graphs = {"clean": graph.drop_edges(test_edges)}
        for variant in variants:
            try:
                graphs[variant] = PERTURBATIONS[variant](graph, int(seed)).drop_edges(test_edges)
            except ValueError:
                continue
        yield int(seed), graph, split, graphs


def graph_stats_summary(cfg, variants, seeds) -> dict:
    """Mean/std of the structural summary of every probe graph."""
    raw: dict[str, dict[int, dict]] = {}
    for seed, _graph, _split, graphs in variant_graphs(cfg, variants, seeds):
        for variant, eval_graph in graphs.items():
            raw.setdefault(variant, {})[seed] = graph_stats(eval_graph)
    out: dict[str, dict] = {}
    for variant, per_seed in raw.items():
        out[variant] = {}
        for key in STAT_KEYS:
            values = [entry[key] for entry in per_seed.values() if key in entry]
            if values:
                out[variant][key] = _mean_std(values)
    return out


def subgraph_diagnostics(cfg, variants, seeds, sample: int = 150) -> dict:
    """What the enclosing-sub-graph extractor can see per probe (no training).

    A sub-graph model can only lose accuracy when its *input* loses information:
    an empty, disconnected or hub-clogged sub-graph is why ``del`` and ``ins_hub``
    hurt SEAL/PENCIL/SGDiff while the same edge budget is absorbed by a full-graph
    encoder.
    """
    raw: dict[str, dict[str, list[float]]] = {}
    for seed, _graph, split, graphs in variant_graphs(cfg, variants, seeds):
        pos, neg = _cap_pairs(split, sample)
        pairs = np.vstack([pos, neg])
        for variant, eval_graph in graphs.items():
            extractor = build_extractor(eval_graph, cfg, seed=seed + 1)
            adj = eval_graph.adjacency()
            deg = eval_graph.degrees()
            hubs = set(np.argsort(-deg)[: max(1, eval_graph.num_nodes // 20)].tolist())
            bucket = raw.setdefault(
                variant,
                {"mean_nodes": [], "mean_edges": [], "connected_frac": [], "cn_frac": [], "hub_frac": []},
            )
            sizes, edges, conn, cn, hub = [], [], [], [], []
            for u, v in pairs:
                sub = extractor.build(int(u), int(v))
                n = sub.num_nodes
                sizes.append(n)
                edges.append(sub.num_edges)
                conn.append(1.0 if _connected(sub.adj, sub.u_idx, sub.v_idx) else 0.0)
                cn.append(
                    1.0
                    if adj.getrow(int(u)).multiply(adj.getrow(int(v))).nnz > 0
                    else 0.0
                )
                hub.append(1.0 if hubs & set(sub.nodes.tolist()) else 0.0)
            bucket["mean_nodes"].append(float(np.mean(sizes)))
            bucket["mean_edges"].append(float(np.mean(edges)))
            bucket["connected_frac"].append(float(np.mean(conn)))
            bucket["cn_frac"].append(float(np.mean(cn)))
            bucket["hub_frac"].append(float(np.mean(hub)))
    return {variant: {key: _mean_std(values) for key, values in per.items()} for variant, per in raw.items()}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
# Scoring: one frozen model, one graph, one view of its inputs
# --------------------------------------------------------------------------- #
def _apply_view(batch, view: str):
    """Ablate one input channel of a sub-graph batch."""
    if view == "features_zeroed":
        return replace(batch, x=torch.zeros_like(batch.x))
    if view == "drnl_permuted":
        labels = batch.labels.clone()
        for i in range(batch.batch_size):
            k = int(batch.mask[i].sum())
            if k > 1:  # permute the DRNL multiset: structure of the batch intact
                perm = torch.randperm(k, device=labels.device)
                labels[i, :k] = batch.labels[i, :k][perm]
        return replace(batch, labels=labels)
    return batch


@torch.no_grad()
def score_model(
    kind: str,
    model: torch.nn.Module,
    cfg,
    eval_graph,
    pairs: np.ndarray,
    device,
    seed: int,
    features=None,
    view: str = "plain",
) -> np.ndarray:
    """Ranking score for every pair (logit of the link class / dot product)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(0)

    if kind in NODE_PAIR_KINDS:
        if features is None:
            features = node_feature_matrix(eval_graph)
        x = np.zeros_like(features) if view == "features_zeroed" else features
        x_t = torch.as_tensor(x, device=device, dtype=torch.float32)
        edge_index = _edge_index(eval_graph, device)
        torch.manual_seed(int(seed))  # VGAE draws z from the posterior
        if hasattr(model, "decode"):
            z, _ = model(x_t, edge_index, eval_graph.num_nodes)
            scorer = model.decode
        else:
            z = model.encode(x_t, edge_index, eval_graph.num_nodes)
            scorer = model.score_pairs
        p = torch.as_tensor(pairs, device=z.device, dtype=torch.long)
        return scorer(z, p).float().cpu().numpy()

    extractor = build_extractor(eval_graph, cfg, seed=cfg.data.seed + 1)
    dataset = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
    )
    torch.manual_seed(int(seed))
    out: list[np.ndarray] = []
    for batch in loader:
        batch = _apply_view(batch.to(device), view)
        out.append(model.predict_logit(batch).float().cpu().numpy())
    return np.concatenate(out)


@contextmanager
def _eta_off(model, name: str):
    """Zero one fusion coefficient and restore it afterwards (frozen weights)."""
    tensor = getattr(model.fusion, name)
    saved = tensor.detach().clone()
    try:
        tensor.data.zero_()
        yield
    finally:
        tensor.data.copy_(saved)


# --------------------------------------------------------------------------- #
# Aggregation / rendering
# --------------------------------------------------------------------------- #
def _mean_std(values: list[float]) -> dict:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _aggregate(runs, models, variants, heuristics_raw, ablations, stats_raw) -> dict:
    model_summary: dict[str, dict] = {}
    for name in models:
        per_variant = {}
        for variant in variants:
            cells = {
                metric: _mean_std(
                    [r["metrics"][metric] for r in runs if r["model"] == name and r["variant"] == variant]
                )
                for metric in METRICS
            }
            drops = [
                r.get("drop", {}).get("auc", float("nan"))
                for r in runs
                if r["model"] == name and r["variant"] == variant
            ]
            cells["drop_auc"] = _mean_std(drops)
            per_variant[variant] = cells
        model_summary[name] = per_variant

    heuristic_summary = {
        variant: {h: _mean_std([s[h] for s in per_seed.values() if h in s]) for h in HEURISTICS}
        for variant, per_seed in heuristics_raw.items()
    }
    stats_summary: dict[str, dict] = {}
    for variant, per_seed in stats_raw.items():
        stats_summary[variant] = {}
        for key in STAT_KEYS:
            values = [entry[key] for entry in per_seed.values() if key in entry]
            if values:
                stats_summary[variant][key] = _mean_std(values)
    ablation_summary: dict[str, dict] = {}
    for name, per_seed in ablations.items():
        views: dict[str, dict] = {}
        for view in {v for entry in per_seed.values() for v in entry}:
            views[view] = {
                metric: _mean_std([e[view][metric] for e in per_seed.values() if view in e])
                for metric in METRICS
            }
        ablation_summary[name] = views
    return {
        "models": model_summary,
        "heuristics": heuristic_summary,
        "graph_stats": stats_summary,
        "ablations": ablation_summary,
    }


def _fmt(entry: dict | None, digits: int = 3, signed: bool = False) -> str:
    if not entry or entry.get("n", 0) == 0 or not np.isfinite(entry["mean"]):
        return "-"
    text = f"{entry['mean']:+.{digits}f}" if signed else f"{entry['mean']:.{digits}f}"
    if entry.get("n", 0) > 1 and entry.get("std", 0.0) > 0:
        text += f" ± {entry['std']:.{digits}f}"
    return text


def format_markdown(result: dict) -> str:
    variants = result["variants_used"]
    models = result["models"]
    summary = result["summary"]
    stats = summary["graph_stats"]
    lines: list[str] = []

    lines += ["## 1. What each probe did to the graph", ""]
    header = ["variant", "#edges", "mean deg", "assort.", "homophily"]
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for variant in variants:
        row = [variant]
        per_variant = stats.get(variant, {})
        for key, digits in (("num_edges", 0), ("mean_degree", 2), ("assortativity", 3), ("homophily", 3)):
            entry = per_variant.get(key)
            row.append("-" if entry is None else f"{entry['mean']:.{digits}f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 2. Property availability (heuristic AUC on the same graph)", ""]
    header = ["variant"] + [f"AUC {h}" for h in HEURISTICS if any(h in v for v in summary["heuristics"].values())]
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for variant in variants:
        cells = [variant]
        for h in HEURISTICS:
            entry = summary["heuristics"].get(variant, {}).get(h)
            cells.append(_fmt(entry) if entry and entry["n"] else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 3. Model AUC per probe", ""]
    header = ["model"] + variants
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name in models:
        cells = [name]
        for variant in variants:
            cells.append(_fmt(summary["models"].get(name, {}).get(variant, {}).get("auc")))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 4. Damage: clean AUC − probe AUC (positive = worse)", ""]
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name in models:
        cells = [name]
        for variant in variants:
            entry = summary["models"].get(name, {}).get(variant, {}).get("drop_auc")
            cells.append("-" if variant == "clean" else _fmt(entry, signed=True))
        lines.append("| " + " | ".join(cells) + " |")

    if summary["ablations"]:
        lines += ["", "## 5. Component ablations on the clean graph (AUC)", ""]
        views = sorted({v for entry in summary["ablations"].values() for v in entry})
        header = ["model"] + views
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for name in models:
            entry = summary["ablations"].get(name)
            if not entry:
                continue
            cells = [name]
            for view in views:
                cells.append(_fmt(entry.get(view, {}).get("auc")))
            lines.append("| " + " | ".join(cells) + " |")

    if result.get("subgraphs"):
        lines += ["", "## 6. What the sub-graph extractor could see (mean over the scored pairs)", ""]
        lines += [
            "`connected` = the two endpoints are in the same component of the extracted",
            "sub-graph; `has_cn` = they share at least one neighbour; `hub_in` = a top-5 %",
            "degree node is inside the (capped) sub-graph.",
            "",
        ]
        header = ["variant", "nodes", "edges", "connected", "has_cn", "hub_in"]
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for variant, per in result["subgraphs"].items():
            cells = [variant]
            for key, digits in (
                ("mean_nodes", 1),
                ("mean_edges", 1),
                ("connected_frac", 3),
                ("cn_frac", 3),
                ("hub_frac", 3),
            ):
                entry = per.get(key)
                cells.append("-" if not entry or not entry["n"] else f"{entry['mean']:.{digits}f}")
            lines.append("| " + " | ".join(cells) + " |")

    if result.get("alignment"):
        lines += ["", "## 7. What the clean model ranks by (Spearman rho with each heuristic)", ""]
        header = ["model"] + list(HEURISTICS)
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for name in models:
            per_h = result["alignment"].get(name, {})
            cells = [name] + [_fmt(per_h.get(h), signed=True) for h in HEURISTICS]
            lines.append("| " + " | ".join(cells) + " |")

    if result.get("skipped"):
        lines += ["", "## Skipped variants", ""]
        reasons: dict[str, set[str]] = {}
        for skip in result["skipped"]:
            reasons.setdefault(str(skip["variant"]), set()).add(str(skip["reason"]))
        for variant, why in reasons.items():
            lines.append(f"* `{variant}`: {'; '.join(sorted(why))}")
    return "\n".join(lines)


def _header(result: dict) -> str:
    return "\n".join(
        [
            f"# Mechanism probe: {result['dataset']}",
            "",
            "Generated by `scripts/run_probe.py`. One clean training run per model;",
            "the probes are applied to the *inference* graph only (evasion), the test",
            "positives are withheld, and every variant scores the same test pairs.",
            "",
            f"* models: {', '.join(result['models'])}",
            f"* probes: {', '.join(result['variants_used'])}",
            f"* seeds: {result['seeds']}",
            f"* scored pairs: {result['protocol']['num_pos']} pos / {result['protocol']['num_neg']} neg",
            f"* epochs: {result['protocol']['epochs']}",
            f"* device: {result['protocol']['device']}",
            "",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_probe(
    config,
    models=None,
    variants=None,
    seeds=(0,),
    sgdiff_epochs=None,
    subgraph_epochs=40,
    node_pair_epochs=300,
    device="auto",
    max_pairs=500,
    output_dir="results",
    run_root="runs/probe",
    tag="",
    write=True,
    verbose=True,
    quiet=False,
    diagnostics=True,
) -> dict:
    if quiet:
        import logging

        logging.getLogger("sgdiff").setLevel(logging.WARNING)
    device = resolve_device(device)
    set_seed(int(seeds[0]))

    registry = build_registry(config)
    models = list(models) if models else list(DEFAULT_MODELS)
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise KeyError(f"unknown models {unknown}; available: {sorted(registry)}")
    variants = list(variants) if variants else list(DEFAULT_VARIANTS)
    unknown_v = [v for v in variants if v not in PERTURBATIONS]
    if unknown_v:
        raise KeyError(f"unknown probes {unknown_v}; available: {sorted(PERTURBATIONS)}")

    runs: list[dict] = []
    heuristics_raw: dict[str, dict] = {}
    stats_raw: dict[str, dict] = {}
    ablations: dict[str, dict] = {}
    alignment_raw: dict[str, dict] = {}
    skipped: list[dict] = []
    variants_used: list[str] = ["clean"]
    used: set[str] = set()
    num_pos = num_neg = 0
    total = len(seeds) * len(models)

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.data.seed = int(seed)
        cfg.train.seed = int(seed)
        # the trainer resolves ``train.device`` on its own; pin it to the device the
        # scoring code uses, otherwise a frozen SGDiff lands on MPS and is scored on CPU
        cfg.train.device = str(device)
        set_seed(int(seed))

        graph, split = load_split(cfg)
        datasets = prepare_datasets(graph, split, cfg)
        pos, neg = _cap_pairs(split, max_pairs)
        num_pos, num_neg = int(len(pos)), int(len(neg))
        pairs_all = np.vstack([pos, neg])
        labels_all = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()
        clean_eval = graph.drop_edges(test_edges)
        clean_features = node_feature_matrix(clean_eval)

        # ---- build every perturbed inference graph once, for all models
        eval_graphs: dict[str, object] = {"clean": clean_eval}
        for variant in variants:
            try:
                perturbed = PERTURBATIONS[variant](graph, int(seed))
            except ValueError as exc:
                skipped.append({"seed": int(seed), "variant": variant, "reason": str(exc)})
                continue
            eval_graphs[variant] = perturbed.drop_edges(test_edges)
            used.add(variant)

        # ---- property availability on the very same graphs
        for variant, eval_graph in eval_graphs.items():
            stats_raw.setdefault(variant, {})[int(seed)] = graph_stats(eval_graph)
            feats = node_feature_matrix(eval_graph) if eval_graph.has_features else None
            per_seed = heuristics_raw.setdefault(variant, {})
            per_seed[int(seed)] = {
                h: float(summarize_scores(labels_all, s)["auc"])
                for h, s in heuristic_scores(eval_graph, pairs_all, feats).items()
            }

        # ---- one clean training run per model, then every probe on frozen weights
        for index, name in enumerate(models, start=1):
            spec = registry[name]
            started = time.perf_counter()
            set_seed(int(seed))  # identical weights regardless of the model list
            model, _train_metrics, num_params = fit_model(
                spec,
                cfg,
                graph,
                split,
                datasets,
                int(seed),
                sgdiff_epochs=sgdiff_epochs,
                subgraph_epochs=subgraph_epochs,
                node_pair_epochs=node_pair_epochs,
                device=device,
                run_root=f"{run_root}/{cfg.data.name}",
            )
            fit_seconds = time.perf_counter() - started
            clean_scores = None
            clean_metrics: dict[str, float] = {}

            for variant, eval_graph in eval_graphs.items():
                set_seed(int(seed))
                features = clean_features
                if variant == "feature_shuffle" and spec.kind in NODE_PAIR_KINDS:
                    # the attributes really are the input here: feed the shuffled rows
                    features = node_feature_matrix(eval_graph)
                scores = score_model(
                    spec.kind,
                    model,
                    cfg,
                    eval_graph,
                    pairs_all,
                    device,
                    int(seed),
                    features=features,
                )
                metrics = {k: float(v) for k, v in summarize_scores(labels_all, scores).items()}
                if variant == "clean":
                    clean_metrics = metrics
                    clean_scores = scores
                runs.append(
                    {
                        "model": name,
                        "seed": int(seed),
                        "variant": variant,
                        "metrics": metrics,
                        "num_params": int(num_params),
                        "fit_seconds": float(fit_seconds),
                    }
                )

            for run in runs:
                if run["model"] != name or run["seed"] != int(seed) or run["variant"] == "clean":
                    continue
                run["drop"] = {
                    k: clean_metrics[k] - v for k, v in run["metrics"].items() if k in clean_metrics
                }

            # ---- signal alignment: what does the clean model rank by?
            if clean_scores is not None:
                feats = node_feature_matrix(clean_eval) if clean_eval.has_features else None
                heur = heuristic_scores(clean_eval, pairs_all, feats)
                per_model = alignment_raw.setdefault(name, {})
                for h, values in heur.items():
                    per_model.setdefault(h, []).append(_spearman(clean_scores, values))

            # ---- component ablations on the clean graph
            entry: dict[str, dict] = {}
            zeros = score_model(
                spec.kind,
                model,
                cfg,
                clean_eval,
                pairs_all,
                device,
                int(seed),
                features=np.zeros_like(clean_features),
                view="features_zeroed",
            )
            entry["features_zeroed"] = {
                k: float(v) for k, v in summarize_scores(labels_all, zeros).items()
            }
            if spec.kind in SUBGRAPH_KINDS:
                set_seed(int(seed))
                perm = score_model(
                    spec.kind,
                    model,
                    cfg,
                    clean_eval,
                    pairs_all,
                    device,
                    int(seed),
                    view="drnl_permuted",
                )
                entry["drnl_permuted"] = {
                    k: float(v) for k, v in summarize_scores(labels_all, perm).items()
                }
            if name == "sgdiff":
                for view, eta in (("structure_only", "eta1"), ("feature_only", "eta2")):
                    if getattr(model.fusion, eta, None) is None:
                        continue
                    set_seed(int(seed))
                    with _eta_off(model, eta):
                        branch = score_model(
                            spec.kind, model, cfg, clean_eval, pairs_all, device, int(seed)
                        )
                    entry[view] = {
                        k: float(v) for k, v in summarize_scores(labels_all, branch).items()
                    }
            ablations.setdefault(name, {})[int(seed)] = entry

            if verbose:
                cells = " ".join(
                    f"{r['variant']}={r['metrics']['auc']:.3f}"
                    for r in runs
                    if r["model"] == name and r["seed"] == int(seed)
                )
                print(
                    f"[{index}/{total}] {cfg.data.name} seed={seed} {name}: {cells} "
                    f"({fit_seconds:.1f}s fit)",
                    flush=True,
                )

    variants_used += [v for v in variants if v in used]
    unknown_skipped = [v for v in variants if v not in used]
    if unknown_skipped:
        skipped.append({"variant": ",".join(unknown_skipped), "reason": "unavailable on this dataset"})

    alignment = {
        name: {h: _mean_std(vals) for h, vals in per_h.items()} for name, per_h in alignment_raw.items()
    }
    result = {
        "dataset": config.data.name,
        "models": models,
        "seeds": [int(s) for s in seeds],
        "variants": variants,
        "variants_used": variants_used,
        "protocol": {
            "setting": "evasion: clean weights, probes applied to the inference graph, test positives withheld",
            "scored_pairs": "test split (per-class cap: %s)" % ("full" if not max_pairs else max_pairs),
            "num_pos": num_pos,
            "num_neg": num_neg,
            "metrics": list(METRICS),
            "epochs": {
                "sgdiff": sgdiff_epochs if sgdiff_epochs is not None else config.train.epochs,
                "subgraph": subgraph_epochs,
                "node_pair": node_pair_epochs,
            },
            "device": str(device),
        },
        "runs": runs,
        "summary": _aggregate(runs, models, variants_used, heuristics_raw, ablations, stats_raw),
        "alignment": alignment,
        "subgraphs": (
            subgraph_diagnostics(config, [v for v in variants_used if v != "clean"], [int(s) for s in seeds])
            if diagnostics
            else {}
        ),
        "skipped": skipped,
    }
    result["markdown"] = format_markdown(result)

    if write:
        suffix = f"_{tag}" if tag else ""
        stem = f"{output_dir}/probe_{config.data.name}{suffix}"
        save_json(result, f"{stem}.json")
        with open(f"{stem}.md", "w", encoding="utf-8") as fh:
            fh.write(_header(result) + result["markdown"] + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None, help="defaults to configs/sgdiff_<dataset>.yaml")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--variants", nargs="*", default=None, help="default: every probe in sgdiff.data.perturb"
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="cpu")
    parser.add_argument("--sgdiff-epochs", type=int, default=None)
    parser.add_argument("--subgraph-epochs", type=int, default=40)
    parser.add_argument("--node-pair-epochs", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=500, help="per-class cap (0 = full test split)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true", help="skip the sub-graph availability pass")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="no training: (re)compute the sub-graph availability section of an existing artifact",
    )
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"]
    overrides += PROFILES[args.profile]
    overrides += args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)

    suffix = f"_{args.tag}" if args.tag else ""
    if args.diagnostics:
        stem = f"{args.output}/probe_{cfg.data.name}{suffix}"
        result = load_json(f"{stem}.json")
        probe_names = [v for v in result["variants_used"] if v != "clean"]
        seeds_used = [int(s) for s in result["seeds"]]
        result["subgraphs"] = subgraph_diagnostics(cfg, probe_names, seeds_used)
        result["summary"]["graph_stats"] = graph_stats_summary(cfg, probe_names, seeds_used)
        result["markdown"] = format_markdown(result)
        save_json(result, f"{stem}.json")
        with open(f"{stem}.md", "w", encoding="utf-8") as fh:
            fh.write(_header(result) + result["markdown"] + "\n")
        print(result["markdown"])
        print(f"\nrewrote {stem}.json and .md")
        return 0

    result = run_probe(
        cfg,
        models=args.models,
        variants=args.variants,
        seeds=args.seeds,
        sgdiff_epochs=args.sgdiff_epochs,
        subgraph_epochs=args.subgraph_epochs,
        node_pair_epochs=args.node_pair_epochs,
        device=args.device,
        max_pairs=args.max_pairs or None,
        output_dir=args.output,
        tag=args.tag,
        quiet=args.quiet,
        diagnostics=not args.no_diagnostics,
    )
    print()
    print(result["markdown"])
    print()
    print(f"wrote {args.output}/probe_{cfg.data.name}{suffix}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
