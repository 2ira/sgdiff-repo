"""Graph purifiers: 2025-2026 defences that clean the graph *before* prediction.

The robustness section used to import RGCN (KDD 2019) because none of the
2025-2026 entries in ``docs/related_work.md`` is an attack-robust link
predictor.  That was the wrong call for the *defence* half of the problem: the
recent literature moved from "robust classifier" to "purify the input, keep the
classifier untouched".  Two representatives are reproduced here:

``gpr``  - **Self-supervised Adversarial Purification for GNNs** (Lee & Park,
           ICML 2025, arXiv:2605.23239).  A graph auto-encoder whose encoder is a
           Generalized-PageRank filter bank (learnable mixture over 1..K hops),
           trained self-supervised to reconstruct the clean adjacency, then
           applied *multi-step* as a plug-and-play purifier.
``gddm`` - **Graph Defense Diffusion Model** (He et al., KDD 2026,
           arXiv:2501.11568).  "Noise" is edges: a denoiser is trained on
           corrupted graphs (random edge drop + random edge insert, noise level
           conditioned) and the reverse chain iteratively *removes* implausible
           edges and *adds* plausible ones.  Two of its components are kept:
           the **structure-driven refiner** (the predicted probability decides
           keep/insert, and the edge budget is pinned to the clean graph so the
           purified graph stays in scope) and the **feature-constrained
           regularizer** (drop edges whose endpoint features disagree).

Both purifiers are trained **self-supervised on the clean training graph** - no
labels, no attack knowledge, no retraining of the downstream predictor - which is
what makes them comparable across every model of the zoo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graph import GraphData
from ..utils.seed import resolve_device
from .layers import sparse_normalize, spmm


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _to_edge_index(edges: np.ndarray, num_nodes: int, device) -> torch.Tensor:
    """Undirected ``(E, 2)`` array -> symmetric ``(2, 2E)`` index tensor."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=device)


def _edge_keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """Canonical ``u * num_nodes + v`` keys with ``u < v``."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.minimum(e[:, 0], e[:, 1]) * num_nodes + np.maximum(e[:, 0], e[:, 1])


def _sample_non_edges(
    num_nodes: int, count: int, rng: np.random.Generator, forbidden: np.ndarray
) -> np.ndarray:
    """Uniform node pairs that are not in ``forbidden`` (canonical keys)."""
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    out: list[tuple[int, int]] = []
    blocked = set(forbidden.tolist())
    tries = 0
    while len(out) < count and tries < count * 20:
        tries += 1
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        key = min(u, v) * num_nodes + max(u, v)
        if key in blocked:
            continue
        blocked.add(key)
        out.append((min(u, v), max(u, v)))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def _feature_threshold(graph: GraphData, quantile: float = 0.1) -> Optional[float]:
    """Cosine-similarity cut calibrated on clean edges (``None`` without features)."""
    if not graph.has_features or graph.num_edges == 0:
        return None
    x = np.asarray(graph.x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    x = x / norm
    e = graph.edges
    sim = (x[e[:, 0]] * x[e[:, 1]]).sum(1)
    return float(np.quantile(sim, quantile))


# --------------------------------------------------------------------------- #
# GPR-GAE  (ICML 2025, arXiv:2605.23239)
# --------------------------------------------------------------------------- #
class GPRGAE(nn.Module):
    """Generalized-PageRank filter bank + self-supervised dot-product decoder."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, hops: int = 3, dropout: float = 0.5):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.gpr = nn.Parameter(torch.zeros(hops + 1))  # one weight per propagation depth
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col, weight = sparse_normalize(edge_index, num_nodes)
        features = [x]
        for _ in range(self.gpr.numel() - 1):
            features.append(spmm(row, col, weight, features[-1], num_nodes))
        coefficients = torch.softmax(self.gpr, dim=0)
        mixed = sum(c * f for c, f in zip(coefficients, features))
        # No ReLU here: with a dot-product decoder and a 1:1 positive/negative
        # ratio, a non-negative embedding starting at zero sits on a saddle point
        # (every logit is 0, so every gradient is 0 and the loss stays at ln 2).
        return F.dropout(self.lin(mixed), p=self.dropout, training=self.training)

    def decode(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)


# --------------------------------------------------------------------------- #
# GDDM edge denoiser  (KDD 2026, arXiv:2501.11568)
# --------------------------------------------------------------------------- #
class EdgeDenoiser(nn.Module):
    """GCN edge denoiser conditioned on a noise level.

    The noise level is a scalar ``t/T in [0, 1)`` appended to the node features,
    so one network serves the whole reverse chain.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lins = nn.ModuleList()
        dims = [in_dim + 1] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, level: float) -> torch.Tensor:
        row, col, weight = sparse_normalize(edge_index, num_nodes)
        h = torch.cat([x, torch.full((x.shape[0], 1), float(level), device=x.device)], dim=-1)
        for index, lin in enumerate(self.lins):
            h = lin(spmm(row, col, weight, h, num_nodes))
            if index < len(self.lins) - 1:
                h = F.relu(F.dropout(h, p=self.dropout, training=self.training))
        return h

    def edge_logits(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(dim=-1)


# --------------------------------------------------------------------------- #
# training (self-supervised, clean graph only)
# --------------------------------------------------------------------------- #
def train_purifier(
    graph: GraphData,
    kind: str = "gddm",
    epochs: int = 200,
    hidden_dim: int = 128,
    lr: float = 0.01,
    max_rate: float = 0.5,
    neg_ratio: float = 1.0,
    threshold_quantile: float = 0.05,
    device: str = "auto",
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Fit a purifier on the clean graph; returns ``{"model", "kind", "feature_threshold"}``.

    ``gddm`` samples a noise level per epoch (random edge drop + insert) and is
    trained to score "is this edge clean?"; ``gpr`` is trained to reconstruct the
    observed adjacency from fresh negative samples each epoch.
    """
    device = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_np = (
        np.asarray(graph.x, dtype=np.float32)
        if graph.has_features
        else np.eye(min(graph.num_nodes, 32), dtype=np.float32)[
            np.arange(graph.num_nodes) % min(graph.num_nodes, 32)
        ]
    )
    x = torch.as_tensor(x_np, device=device)
    clean_edges = np.asarray(graph.edges, dtype=np.int64)
    clean_keys = set(_edge_keys(clean_edges, graph.num_nodes).tolist())
    num_nodes = graph.num_nodes
    edge_budget = max(len(clean_edges), 1)

    if kind == "gpr":
        model = GPRGAE(x.shape[1], hidden_dim=hidden_dim).to(device)
    elif kind == "gddm":
        model = EdgeDenoiser(x.shape[1], hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"unknown purifier kind '{kind}'")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if kind == "gddm":
            level = float(rng.random())
            rate = max_rate * level
            keep = rng.random(len(clean_edges)) >= rate
            dropped = clean_edges[keep]
            inserted = _sample_non_edges(
                num_nodes,
                int(round(rate * edge_budget)),
                rng,
                np.asarray(list(clean_keys), dtype=np.int64),
            )
            noisy = np.vstack([dropped, inserted]) if len(inserted) else dropped
            candidates = noisy
            labels = np.isin(_edge_keys(candidates, num_nodes), list(clean_keys)).astype(np.float32)
            h = model(x, _to_edge_index(noisy, num_nodes, device), num_nodes, level)
            logits = model.edge_logits(h, torch.as_tensor(candidates, dtype=torch.long, device=device))
        else:
            negatives = _sample_non_edges(
                num_nodes, int(neg_ratio * edge_budget), rng, np.asarray(list(clean_keys), dtype=np.int64)
            )
            candidates = np.vstack([clean_edges, negatives]) if len(negatives) else clean_edges
            labels = np.concatenate(
                [np.ones(len(clean_edges), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32)]
            )
            z = model(x, _to_edge_index(clean_edges, num_nodes, device), num_nodes)
            logits = model.decode(z, torch.as_tensor(candidates, dtype=torch.long, device=device))
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(labels, device=device)
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  [{kind}] epoch {epoch + 1} loss {history[-1]:.4f}")

    state = {
        "model": model,
        "kind": kind,
        "feature_threshold": _feature_threshold(graph),
        "train_loss": history,
        "edge_budget": int(edge_budget),
    }
    # Calibrate the keep-threshold on *clean* edges: keep the fraction that
    # preserves 90% of them, so an attacked graph loses the low-scoring noise and
    # the clean graph is essentially untouched (no clean-accuracy tax).
    clean_scores = _edge_probabilities(state, graph, clean_edges, 0.0, device)
    state["edge_threshold"] = float(np.quantile(clean_scores, 0.1)) if len(clean_scores) else 0.5
    state["clean_edge_auc"] = _clean_edge_separation(state, graph, clean_edges, seed, device)
    return state


# --------------------------------------------------------------------------- #
# purification
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _edge_probabilities(state: dict, graph: GraphData, edges: np.ndarray, level: float, device) -> np.ndarray:
    model = state["model"]
    model.eval()
    x_np = (
        np.asarray(graph.x, dtype=np.float32)
        if graph.has_features
        else np.eye(min(graph.num_nodes, 32), dtype=np.float32)[
            np.arange(graph.num_nodes) % min(graph.num_nodes, 32)
        ]
    )
    x = torch.as_tensor(x_np, device=device)
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float32)
    labels = torch.as_tensor(edges, dtype=torch.long, device=device)
    if state["kind"] == "gddm":
        h = model(x, _to_edge_index(edges, graph.num_nodes, device), graph.num_nodes, level)
        logits = model.edge_logits(h, labels)
    else:
        z = model(x, _to_edge_index(edges, graph.num_nodes, device), graph.num_nodes)
        logits = model.decode(z, labels)
    return torch.sigmoid(logits).float().cpu().numpy()


def _clean_edge_separation(state: dict, graph: GraphData, clean_edges: np.ndarray, seed: int, device) -> float:
    """AUC of the purifier at telling clean edges from random node pairs.

    This is the number that decides whether a purifier can work at all: a score
    near 0.5 means it has learned nothing about structure and any "defence" it
    provides is noise.
    """
    from ..utils.metrics import summarize_scores

    keys = _edge_keys(clean_edges, graph.num_nodes)
    negatives = _sample_non_edges(graph.num_nodes, len(keys), np.random.default_rng(seed), keys)
    if len(negatives) == 0:
        return float("nan")
    pos_scores = _edge_probabilities(state, graph, clean_edges, 0.0, device)
    neg_scores = _edge_probabilities(state, graph, negatives, 0.0, device)
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return float(summarize_scores(labels, np.concatenate([pos_scores, neg_scores]))["auc"])


def _two_hop_candidates(edges: np.ndarray, num_nodes: int, existing: set[int], limit: int) -> np.ndarray:
    """A bounded sample of 2-hop non-edges (the only plausible insertions)."""
    if limit <= 0 or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    neighbours: list[set] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))
    rng = np.random.default_rng(0)
    out: list[tuple[int, int]] = []
    nodes = rng.permutation(num_nodes)
    for u in nodes:
        if len(out) >= limit:
            break
        for mid in neighbours[int(u)]:
            for v in neighbours[mid]:
                if v == u:
                    continue
                key = min(int(u), v) * num_nodes + max(int(u), v)
                if key in existing:
                    continue
                existing.add(key)
                out.append((min(int(u), v), max(int(u), v)))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


@torch.no_grad()
def purify_graph(
    graph: GraphData,
    state: dict,
    steps: int = 3,
    keep_threshold: Optional[float] = None,
    rank_budget: bool = True,
    insert_threshold: float = 0.9,
    insert_fraction: float = 0.05,
    feature_filter: bool = True,
    device="auto",
) -> GraphData:
    """Return a purified copy of ``graph`` (edge set only; nodes/features untouched).

    Reverse chain (GDDM) or multi-step refinement (GPR): at each step every
    current edge is scored by the purifier and kept when ``p >= keep_threshold``;
    a bounded number of 2-hop candidates is inserted when ``p >= insert_threshold``;
    the total edge count is capped at the clean graph's budget so the purified
    graph stays in scope.  The final step applies the feature-constrained filter.
    """
    device = resolve_device(device)
    edges = np.asarray(graph.edges, dtype=np.int64)
    budget = state.get("edge_budget", len(edges))
    existing = set(_edge_keys(edges, graph.num_nodes).tolist())

    for step in range(max(steps, 1)):
        level = 1.0 - (step + 1) / max(steps, 1)
        if len(edges):
            probabilities = _edge_probabilities(state, graph, edges, level, device)
            threshold = state.get("edge_threshold") if keep_threshold is None else keep_threshold
            if threshold is None:
                threshold = 0.5
            keep = probabilities >= threshold
            edges = edges[keep]
            if rank_budget and len(edges) > budget:
                # "fidelity / scope" component: never exceed the clean graph's
                # edge count, so an attack that only inserts edges cannot inflate
                # the purified graph.
                probabilities = probabilities[keep]
                order = np.argsort(-probabilities)[:budget]
                edges = edges[np.sort(order)]
            existing = set(_edge_keys(edges, graph.num_nodes).tolist())
        if insert_fraction > 0 and len(edges) < budget:
            wanted = min(int(insert_fraction * budget), budget - len(edges))
            candidates = _two_hop_candidates(edges, graph.num_nodes, existing, limit=wanted * 4)
            if len(candidates):
                probabilities = _edge_probabilities(state, graph, candidates, level, device)
                accepted = candidates[probabilities >= insert_threshold][:wanted]
                if len(accepted):
                    edges = np.vstack([edges, accepted]) if len(edges) else accepted
                    existing = set(_edge_keys(edges, graph.num_nodes).tolist())

    if feature_filter and state.get("feature_threshold") is not None and graph.has_features and len(edges):
        x = np.asarray(graph.x, dtype=np.float32)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        unit = x / norm
        similarity = (unit[edges[:, 0]] * unit[edges[:, 1]]).sum(1)
        edges = edges[similarity >= state["feature_threshold"]]

    return GraphData(
        name=f"{graph.name}-purified",
        num_nodes=graph.num_nodes,
        edges=edges,
        x=graph.x,
        y=graph.y,
        meta={**graph.meta, "purified": {"kind": state["kind"], "steps": steps, "edges": int(len(edges))}},
    )
