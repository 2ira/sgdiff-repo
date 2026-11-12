"""Targeted structural perturbations for mechanism attribution.

The RQ3 attacks (RF / DICE / EA) mix deletion and insertion, so they cannot say
*which* property a model relies on.  These probes separate the mechanisms:

``del``        delete a fraction of the true edges (connectivity damage)
``del_hub``    delete edges weighted by endpoint degree (degree damage)
``ins``        insert random non-edges (uniform noise)
``ins_hub``    insert non-edges attached to high-degree nodes (hub inflation)
``ins_homo``   insert non-edges whose endpoints share a label (invisible to
               homophily-based reasoning: the edge looks plausible)
``ins_hetero`` insert non-edges whose endpoints differ (maximally implausible)
``feature_shuffle``  permute node feature rows (destroys the feature signal
               without touching structure)

All functions return a new :class:`~sgdiff.data.graph.GraphData` and never mutate
the input.  Test edges are removed by the caller, exactly as in the robustness
harness, so the scored pairs are identical across variants.
"""

from __future__ import annotations

import numpy as np

from .graph import GraphData, unique_undirected_edges


def _keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.minimum(e[:, 0], e[:, 1]) * num_nodes + np.maximum(e[:, 0], e[:, 1])


def _degrees(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, e[:, 0], 1)
    np.add.at(deg, e[:, 1], 1)
    return deg


def delete_edges(graph: GraphData, ratio: float, mode: str = "random", seed: int = 0) -> GraphData:
    """Remove ``ratio * |E|`` true edges, uniformly or biased towards hubs."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(graph.edges, dtype=np.int64)
    count = int(round(ratio * len(edges)))
    if count <= 0:
        return graph
    if mode == "random":
        drop = rng.choice(len(edges), size=min(count, len(edges)), replace=False)
    elif mode == "hub":
        deg = _degrees(edges, graph.num_nodes).astype(np.float64)
        weight = deg[edges[:, 0]] + deg[edges[:, 1]]
        probability = weight / weight.sum()
        drop = rng.choice(len(edges), size=min(count, len(edges)), replace=False, p=probability)
    else:
        raise ValueError(f"unknown delete mode '{mode}'")
    keep = np.ones(len(edges), dtype=bool)
    keep[drop] = False
    return GraphData(
        name=f"{graph.name}-del-{mode}",
        num_nodes=graph.num_nodes,
        edges=edges[keep],
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "perturb": {"kind": f"del_{mode}", "ratio": ratio}},
    )


def insert_edges(
    graph: GraphData, ratio: float, mode: str = "random", seed: int = 0
) -> GraphData:
    """Insert ``ratio * |E|`` non-edges (random / hub-attached / homophilous / heterophilous)."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(graph.edges, dtype=np.int64)
    num_nodes = graph.num_nodes
    if mode not in ("random", "hub", "homo", "hetero"):
        raise ValueError(f"unknown insert mode '{mode}'")
    count = int(round(ratio * len(edges)))
    if count <= 0:
        return graph
    existing = set(_keys(edges, num_nodes).tolist())
    labels = graph.y
    if mode in ("homo", "hetero") and labels is None:
        raise ValueError(f"insert mode '{mode}' needs node labels")

    deg = _degrees(edges, num_nodes)
    hubs = np.argsort(-deg)[: max(1, num_nodes // 20)] if mode == "hub" else None

    added: list[tuple[int, int]] = []
    attempts = 0
    while len(added) < count and attempts < count * 50:
        attempts += 1
        if mode == "hub":
            u = int(hubs[rng.integers(0, len(hubs))])
            v = int(rng.integers(0, num_nodes))
        else:
            u = int(rng.integers(0, num_nodes))
            v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        key = min(u, v) * num_nodes + max(u, v)
        if key in existing:
            continue
        if mode == "homo" and labels[u] != labels[v]:
            continue
        if mode == "hetero" and labels[u] == labels[v]:
            continue
        existing.add(key)
        added.append((min(u, v), max(u, v)))

    new_edges = np.vstack([edges, np.asarray(added, dtype=np.int64)]) if added else edges
    return GraphData(
        name=f"{graph.name}-ins-{mode}",
        num_nodes=num_nodes,
        edges=unique_undirected_edges(new_edges, num_nodes),
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "perturb": {"kind": f"ins_{mode}", "ratio": ratio}},
    )


def shuffle_features(graph: GraphData, seed: int = 0) -> GraphData:
    """Permute node feature rows: structure intact, feature signal destroyed."""
    if not graph.has_features:
        raise ValueError("feature_shuffle needs node features")
    rng = np.random.default_rng(seed)
    order = rng.permutation(graph.num_nodes)
    return GraphData(
        name=f"{graph.name}-featshuffle",
        num_nodes=graph.num_nodes,
        edges=np.asarray(graph.edges),
        x=np.asarray(graph.x)[order],
        y=graph.y,
        meta={**graph.meta, "perturb": {"kind": "feature_shuffle"}},
    )


#: Perturbation plan used by ``scripts/run_probe.py``: name -> callable(graph, ratio).
PERTURBATIONS = {
    "del@50%": lambda g, s: delete_edges(g, 0.5, "random", s),
    "del_hub@50%": lambda g, s: delete_edges(g, 0.5, "hub", s),
    "ins@50%": lambda g, s: insert_edges(g, 0.5, "random", s),
    "ins_hub@50%": lambda g, s: insert_edges(g, 0.5, "hub", s),
    "ins_homo@50%": lambda g, s: insert_edges(g, 0.5, "homo", s),
    "ins_hetero@50%": lambda g, s: insert_edges(g, 0.5, "hetero", s),
    "feature_shuffle": lambda g, s: shuffle_features(g, s),
}
