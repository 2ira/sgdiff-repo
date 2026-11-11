"""Structural adversarial attacks used by the robustness experiment (Sec. 4.4).

The paper attacks the graph structure with DeepRobust.  We ship self-contained,
dependency-free attacks that share the same interface so that a DeepRobust
wrapper can be dropped in:

* :func:`random_perturbation` -- random edge insertion / deletion;
* :func:`dice_attack` -- DICE (delete internally, connect externally), which
  rewires links between different communities;
* :func:`embedding_attack` -- "EA": a meta-gradient poisoning attack on a node
  *embedding* model (see the docstring for how it differs from DeepRobust's
  exact implementation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .graph import GraphData

#: The dense ``dL/dA`` of :func:`embedding_attack` needs ``num_nodes**2`` floats.
EA_MAX_NODES = 4000

_METHOD_ALIASES: dict[str, Optional[str]] = {
    "none": None,
    "clean": None,
    "": None,
    "rf": "random",
    "rand": "random",
    "random": "random",
    "random_perturbation": "random",
    "dice": "dice",
    "ea": "ea",
    "embedding": "ea",
    "embedding_attack": "ea",
}


def _budget(graph: GraphData, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    return int(round(ratio * graph.num_edges))


def normalize_attack(method: Optional[str]) -> Optional[str]:
    """Map CLI/driver aliases onto ``None``, ``"random"``, ``"dice"`` or ``"ea"``."""
    key = "" if method is None else str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"unknown attack '{method}', expected one of {sorted(_METHOD_ALIASES)}"
        )
    return _METHOD_ALIASES[key]


def supports_attack(
    method: Optional[str], num_nodes: int, max_nodes: int = EA_MAX_NODES
) -> bool:
    """``False`` only when the attack cannot run at this graph size (dense EA)."""
    canonical = normalize_attack(method)
    return canonical != "ea" or int(num_nodes) <= int(max_nodes)


def apply_attack(
    graph: GraphData,
    method: Optional[str] = "rf",
    ratio: float = 0.1,
    seed: int = 0,
    mode: str = "flip",
    device: str = "cpu",
    max_nodes: int = EA_MAX_NODES,
) -> GraphData:
    """Apply one structural attack by name (the single entry point used by the drivers).

    ``method`` accepts ``None``/``"clean"`` (no-op), ``"random"``/``"rf"``,
    ``"dice"`` and ``"ea"``/``"embedding"``.  ``mode`` only affects the random
    attack.  Raises :class:`ValueError` when EA is asked for a graph larger than
    ``max_nodes`` — callers that iterate over graphs should probe
    :func:`supports_attack` first.
    """
    canonical = normalize_attack(method)
    if canonical is None:
        return graph
    if canonical == "random":
        return random_perturbation(graph, ratio=ratio, mode=mode, seed=seed)
    if canonical == "dice":
        return dice_attack(graph, ratio=ratio, seed=seed)
    return embedding_attack(graph, ratio=ratio, device=device, max_nodes=max_nodes, seed=seed)


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
    dropped = out
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
        meta={
            **graph.meta,
            "attack": {
                "method": "random",
                "mode": mode,
                "ratio": ratio,
                "deleted": int(len(graph.edges) - len(dropped.edges)),
                "inserted": int(len(out.edges) - len(dropped.edges)),
            },
        },
    )


def dice_attack(
    graph: GraphData,
    ratio: float = 0.1,
    community: Optional[np.ndarray] = None,
    seed: int = 0,
) -> GraphData:
    """DICE attack: drop intra-community links, add inter-community links.

    ``community`` defaults to :func:`default_community`, which prefers the node
    labels ``graph.y`` (what DeepRobust's DICE uses), falls back to connected
    components and finally to label propagation.  Without that fallback a
    connected graph collapses to a single community and DICE could delete edges
    but never insert one.
    """
    rng = np.random.default_rng(seed)
    budget = _budget(graph, ratio)
    if community is None:
        labels, partition = default_community(graph, seed=seed)
    else:
        labels, partition = np.asarray(community).ravel(), "provided"

    same = labels[graph.edges[:, 0]] == labels[graph.edges[:, 1]]
    intra = graph.edges[same]
    n_del = min(budget // 2, len(intra))
    drop = intra[rng.choice(len(intra), size=n_del, replace=False)] if n_del else np.zeros((0, 2), int)
    out = graph.drop_edges(drop) if n_del else graph
    dropped = out

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
        meta={
            **graph.meta,
            "attack": {
                "method": "dice",
                "ratio": ratio,
                "partition": partition,
                "communities": int(np.unique(labels).size),
                "deleted": int(len(graph.edges) - len(dropped.edges)),
                "inserted": int(len(out.edges) - len(dropped.edges)),
            },
        },
    )


def embedding_attack(
    graph: GraphData,
    ratio: float = 0.1,
    dim: int = 32,
    surrogate_steps: int = 60,
    pairs_per_step: int = 4096,
    max_nodes: int = 4000,
    device: str = "cpu",
    seed: int = 0,
) -> GraphData:
    """Embedding attack (EA): meta-gradient poisoning of a node-embedding model.

    Sec. 4.4 of the paper uses DeepRobust's embedding attack, which poisons the
    structure so that a *learned node embedding* degrades.  The pipeline is:

    1. train a differentiable surrogate embedding ``Z = A_norm W`` (``W`` is
       ``num_nodes x dim``) with a dot-product link objective;
    2. differentiate that objective **through the adjacency** and keep
       ``dL/dA`` — the meta-gradient;
    3. flip the ``ratio * |E|`` entries whose flip increases the loss the most
       (delete edges with ``dL/dA < 0``, insert non-edges with ``dL/dA > 0``).

    Deviations from DeepRobust's implementation, which needs the package and a
    spectral/NetMF surrogate: we use a one-layer linear encoder as the surrogate
    and a single attack step (no PGD over the perturbation set).  The objective
    (degrade an embedding via structure) and the budget semantics (fraction of
    the edge count) are the same, so the numbers are comparable in trend, not in
    value, to the paper's EA column.

    Dense ``dL/dA`` needs ``num_nodes^2`` floats, so the attack refuses graphs
    bigger than ``max_nodes`` (Cora 2 708 fits; Pubmed 19 717 does not).  The
    paper's RQ3 runs on Cora.
    """
    if graph.num_nodes > max_nodes:
        raise ValueError(
            f"embedding_attack is dense: {graph.num_nodes} nodes > max_nodes={max_nodes}. "
            "Run it on a smaller graph (the paper's RQ3 uses Cora)."
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    n = graph.num_nodes
    budget = _budget(graph, ratio)
    if budget == 0:
        return graph

    edges = torch.as_tensor(np.asarray(graph.edges, dtype=np.int64), device=dev)
    adj = torch.zeros(n, n, device=dev)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj[edges[:, 1], edges[:, 0]] = 1.0
    degree = adj.sum(dim=1).clamp(min=1.0)
    adj_norm = adj / degree.unsqueeze(1)

    weight = torch.nn.Parameter(torch.randn(n, dim, device=dev) * 0.1)
    opt = torch.optim.Adam([weight], lr=0.01)
    n_pos = len(edges)

    half = pairs_per_step // 2

    def sample_pairs() -> tuple[torch.Tensor, torch.Tensor]:
        # tiny graphs have fewer edges than `half`, so sample with replacement
        pos = edges[rng.choice(n_pos, size=half, replace=n_pos < half)]
        neg = torch.as_tensor(
            np.stack(
                [rng.choice(n, size=half, replace=True), rng.choice(n, size=half, replace=True)],
                axis=1,
            ),
            device=dev,
        )
        pairs = torch.cat([pos, neg], dim=0)
        labels = torch.cat(
            [torch.ones(len(pos), device=dev), torch.zeros(len(neg), device=dev)]
        )
        return pairs, labels

    # 1. fit the surrogate embedding on the clean graph
    for _ in range(surrogate_steps):
        pairs, labels = sample_pairs()
        z = adj_norm @ weight
        logits = (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 2. meta-gradient of the *trained* objective w.r.t. the adjacency
    a = adj.clone().requires_grad_(True)
    pairs, labels = sample_pairs()
    z = (a / a.sum(dim=1, keepdim=True).clamp(min=1.0)) @ weight.detach()
    logits = (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    grad = torch.autograd.grad(loss, a)[0]
    grad = 0.5 * (grad + grad.t())
    # only the upper triangle, otherwise (i, j) and (j, i) would both be picked
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=1)

    # 3. rank the flips: deletions need grad < 0, insertions need grad > 0
    n_del = min(budget // 2, len(edges))
    n_ins = budget - n_del
    del_scores = torch.where((adj > 0) & upper, grad, torch.full_like(grad, torch.inf))
    ins_scores = torch.where(upper & (adj == 0), grad, torch.full_like(grad, -torch.inf))

    drop_idx = torch.topk(-del_scores.flatten(), k=n_del).indices
    ins_idx = torch.topk(ins_scores.flatten(), k=n_ins).indices
    drop = np.stack([(drop_idx // n).cpu().numpy(), (drop_idx % n).cpu().numpy()], axis=1)
    insert = np.stack([(ins_idx // n).cpu().numpy(), (ins_idx % n).cpu().numpy()], axis=1)

    dropped = graph.drop_edges(drop) if n_del else graph
    out = dropped
    if n_ins:
        existing = {tuple(sorted(map(int, e))) for e in out.edges}
        fresh = [tuple(map(int, p)) for p in insert if tuple(sorted(map(int, p))) not in existing]
        if fresh:
            out = out.add_edges(np.asarray(fresh, dtype=np.int64))

    return GraphData(
        name=f"{graph.name}-ea{ratio}",
        num_nodes=out.num_nodes,
        edges=out.edges,
        x=graph.x,
        y=graph.y,
        meta={
            **graph.meta,
            "attack": {
                "method": "ea",
                "ratio": ratio,
                "dim": dim,
                "surrogate_steps": surrogate_steps,
                "deleted": int(len(graph.edges) - len(dropped.edges)),
                "inserted": int(len(out.edges) - len(dropped.edges)),
            },
        },
    )


def default_community(graph: GraphData, seed: int = 0) -> tuple[np.ndarray, str]:
    """The partition DICE rewires across, plus the name of its source.

    Priority: node labels ``graph.y`` (DeepRobust's DICE uses them) → connected
    components → label propagation.  A partition is rejected when it is trivial
    (one community, or one community holding ≥ 95 % of the nodes): every edge is
    then "intra-community", so DICE could delete links but never insert any.
    """
    labels = _target_labels(graph)
    if labels is not None and _is_nontrivial(labels, graph.num_nodes):
        return labels, "targets"
    labels = _component_labels(graph)
    if _is_nontrivial(labels, graph.num_nodes):
        return labels, "components"
    return _label_propagation_labels(graph, seed=seed), "lpa"


def _is_nontrivial(labels: np.ndarray, num_nodes: int, max_share: float = 0.95) -> bool:
    if num_nodes <= 1:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    return unique.size >= 2 and counts.max() <= max_share * num_nodes


def _target_labels(graph: GraphData) -> Optional[np.ndarray]:
    """``graph.y`` re-indexed to ``0..k-1`` when it is a usable class vector."""
    if graph.y is None:
        return None
    y = np.asarray(graph.y).ravel()
    if y.size != graph.num_nodes or not np.issubdtype(y.dtype, np.number):
        return None
    if not np.all(np.isfinite(y.astype(np.float64))):
        return None
    unique = np.unique(y)
    if not 2 <= unique.size <= max(2, min(100, graph.num_nodes // 2)):
        return None
    if not np.allclose(unique, np.round(unique)):  # continuous targets are not classes
        return None
    return np.unique(y, return_inverse=True)[1].astype(np.int64)


def _label_propagation_labels(graph: GraphData, seed: int = 0, iterations: int = 10) -> np.ndarray:
    """Deterministic (seeded) label propagation, used when components collapse."""
    rng = np.random.default_rng(seed)
    labels = np.arange(graph.num_nodes, dtype=np.int64)
    neighbors = graph.neighbors()
    for _ in range(iterations):
        changed = 0
        for node in rng.permutation(graph.num_nodes):
            node = int(node)
            adjacent = neighbors[node]
            if len(adjacent) == 0:
                continue
            values, counts = np.unique(labels[adjacent], return_counts=True)
            best = values[counts == counts.max()]
            chosen = int(best[rng.integers(0, len(best))]) if len(best) > 1 else int(best[0])
            if chosen != labels[node]:
                labels[node] = chosen
                changed += 1
        if changed == 0:
            break
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


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
