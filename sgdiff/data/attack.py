"""Structural adversarial attacks used by the robustness experiment (Sec. 4.4).

The paper attacks the graph structure with DeepRobust.  We ship two
self-contained, dependency-free attacks that share the same interface so that a
DeepRobust wrapper can be dropped in:

* :func:`random_perturbation` -- random edge insertion / deletion;
* :func:`dice_attack` -- DICE (delete internally, connect externally), which
  rewires links between different communities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .graph import GraphData


def _budget(graph: GraphData, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    return int(round(ratio * graph.num_edges))


def random_perturbation(
    graph: GraphData,
    ratio: float = 0.1,
    mode: str = "flip",
    seed: int = 0,
) -> GraphData:
    """Randomly delete and/or insert edges.

    ``mode`` is one of ``"delete"``, ``"insert"``, ``"flip"`` (half/half).
    """
    rng = np.random.default_rng(seed)
    budget = _budget(graph, ratio)
    if mode == "flip":
        n_del, n_ins = budget // 2, budget - budget // 2
    elif mode == "delete":
        n_del, n_ins = budget, 0
    elif mode == "insert":
        n_del, n_ins = 0, budget
    else:
        raise ValueError(f"unknown mode '{mode}'")

    out = graph
    if n_del:
        drop = graph.edges[
            rng.choice(len(graph.edges), size=min(n_del, len(graph.edges)), replace=False)
        ]
        out = out.drop_edges(drop)
    if n_ins:
        existing = {tuple(sorted(map(int, e))) for e in out.edges}
        new: list[tuple[int, int]] = []
        tries = 0
        while len(new) < n_ins and tries < 100 * n_ins:
            tries += 1
            u = int(rng.integers(0, out.num_nodes))
            v = int(rng.integers(0, out.num_nodes))
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in existing:
                continue
            existing.add(key)
            new.append(key)
        if new:
            out = out.add_edges(np.asarray(new, dtype=np.int64))
    return GraphData(
        name=f"{graph.name}-rand{mode}{ratio}",
        num_nodes=out.num_nodes,
        edges=out.edges,
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "attack": {"method": "random", "mode": mode, "ratio": ratio}},
    )


def dice_attack(
    graph: GraphData,
    ratio: float = 0.1,
    community: Optional[np.ndarray] = None,
    seed: int = 0,
) -> GraphData:
    """DICE attack: drop intra-community links, add inter-community links.

    ``community`` defaults to connected-component labels computed with BFS,
    which is a cheap proxy when no ground-truth partition is available.
    """
    rng = np.random.default_rng(seed)
    budget = _budget(graph, ratio)
    labels = community if community is not None else _component_labels(graph)

    same = labels[graph.edges[:, 0]] == labels[graph.edges[:, 1]]
    intra = graph.edges[same]
    n_del = min(budget // 2, len(intra))
    drop = intra[rng.choice(len(intra), size=n_del, replace=False)] if n_del else np.zeros((0, 2), int)
    out = graph.drop_edges(drop) if n_del else graph

    existing = {tuple(sorted(map(int, e))) for e in out.edges}
    new: list[tuple[int, int]] = []
    n_ins = budget - n_del
    tries = 0
    while len(new) < n_ins and tries < 100 * max(n_ins, 1):
        tries += 1
        u = int(rng.integers(0, out.num_nodes))
        v = int(rng.integers(0, out.num_nodes))
        if u == v or labels[u] == labels[v]:
            continue
        key = (u, v) if u < v else (v, u)
        if key in existing:
            continue
        existing.add(key)
        new.append(key)
    if new:
        out = out.add_edges(np.asarray(new, dtype=np.int64))

    return GraphData(
        name=f"{graph.name}-dice{ratio}",
        num_nodes=out.num_nodes,
        edges=out.edges,
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "attack": {"method": "dice", "ratio": ratio}},
    )


def _component_labels(graph: GraphData) -> np.ndarray:
    labels = -np.ones(graph.num_nodes, dtype=np.int64)
    neighbors = graph.neighbors()
    current = 0
    for start in range(graph.num_nodes):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current
        while stack:
            node = stack.pop()
            for nxt in neighbors[node]:
                if labels[nxt] == -1:
                    labels[nxt] = current
                    stack.append(int(nxt))
        current += 1
    return labels
