"""Enclosing sub-graph extraction (SGNN step of SGDiff).

For a node pair ``(u, v)`` a :math:`k`-hop enclosing sub-graph is the union of
the :math:`k`-hop neighbourhoods of ``u`` and ``v``.  Nodes are annotated with
the Double-Radius Node Labeling (DRNL) of Zhang & Chen (SEAL), which resolves
the automorphic-node problem mentioned in Sec. 2.1 of the paper.

The edge ``(u, v)`` itself is removed from the extracted adjacency so that the
model cannot read the label off the input (leakage-free evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .graph import GraphData


def drnl_label(du: int, dv: int, max_label: int) -> int:
    """Double-radius node label for distances ``du``/``dv`` to ``u``/``v``."""
    d = du + dv
    half = d // 2
    label = 1 + min(du, dv) + half * (half + (d % 2) - 1)
    return int(min(label, max_label))


def _bfs(
    neighbors: list[np.ndarray],
    root: int,
    hops: int,
    cap: Optional[int],
    rng: np.random.Generator,
    blocked: Optional[int] = None,
) -> dict:
    """BFS up to ``hops`` hops from ``root``.

    ``blocked`` removes the edge ``(root, blocked)`` from the traversal, which is
    how the target link is hidden from the enclosing sub-graph.
    """
    dist = {root: 0}
    frontier = [root]
    for hop in range(1, hops + 1):
        nxt: list[int] = []
        seen = set()
        for node in frontier:
            for nb in neighbors[node]:
                nb = int(nb)
                if blocked is not None and node == root and nb == blocked:
                    continue
                if nb in dist or nb in seen:
                    continue
                seen.add(nb)
                nxt.append(nb)
        if cap is not None and len(nxt) > cap:
            keep = rng.choice(len(nxt), size=cap, replace=False)
            nxt = [nxt[i] for i in np.sort(keep)]
        for nb in nxt:
            dist[nb] = hop
        frontier = nxt
        if not frontier:
            break
    return dist


@dataclass
class Subgraph:
    """A single enclosing sub-graph with node labels and optional features."""

    src: int
    dst: int
    nodes: np.ndarray
    adj: np.ndarray
    labels: np.ndarray
    x: Optional[np.ndarray] = None
    _pos: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        self.adj = np.asarray(self.adj, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self._pos = {int(node): i for i, node in enumerate(self.nodes)}
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=np.float32)

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.size)

    @property
    def num_edges(self) -> int:
        return int(self.adj.sum() // 2)

    @property
    def u_idx(self) -> int:
        return self._pos[int(self.src)]

    @property
    def v_idx(self) -> int:
        return self._pos[int(self.dst)]

    @property
    def candidates(self) -> tuple[int, int]:
        """Local ids of the target pair (the entries the classifier scores)."""
        return self.u_idx, self.v_idx


def extract_enclosing_subgraph(
    neighbors: list[np.ndarray],
    src: int,
    dst: int,
    hops: int = 2,
    max_nodes_per_hop: Optional[int] = 50,
    max_label: int = 16,
    features: Optional[np.ndarray] = None,
    remove_target: bool = True,
    seed: int = 0,
) -> Subgraph:
    """Extract the ``hops``-hop enclosing sub-graph of ``(src, dst)``."""
    if src == dst:
        raise ValueError("self-loop pairs are not valid link-prediction candidates")
    rng = np.random.default_rng((seed * 1_000_003 + int(src) * 9176 + int(dst)) % (2**32))

    block_src = int(dst) if remove_target else None
    block_dst = int(src) if remove_target else None
    du = _bfs(neighbors, int(src), hops, max_nodes_per_hop, rng, blocked=block_src)
    dv = _bfs(neighbors, int(dst), hops, max_nodes_per_hop, rng, blocked=block_dst)

    node_set = set(du) | set(dv)
    nodes = np.asarray(sorted(node_set), dtype=np.int64)
    pos = {int(node): i for i, node in enumerate(nodes)}

    adj = np.zeros((nodes.size, nodes.size), dtype=np.float32)
    for i, node in enumerate(nodes):
        for nb in neighbors[int(node)]:
            j = pos.get(int(nb))
            if j is not None:
                adj[i, j] = 1.0

    fallback = hops + 1
    labels = np.asarray(
        [
            drnl_label(du.get(int(n), fallback), dv.get(int(n), fallback), max_label)
            for n in nodes
        ],
        dtype=np.int64,
    )

    if remove_target and pos[int(src)] is not None:
        i, j = pos[int(src)], pos[int(dst)]
        adj[i, j] = 0.0
        adj[j, i] = 0.0

    x = None if features is None else features[nodes]
    return Subgraph(src=int(src), dst=int(dst), nodes=nodes, adj=adj, labels=labels, x=x)


class SubgraphExtractor:
    """Builds :class:`Subgraph` objects for many pairs, with a small LRU-style cache."""

    def __init__(
        self,
        graph: GraphData,
        hops: int = 2,
        max_nodes_per_hop: Optional[int] = 50,
        max_label: int = 16,
        feature_mode: str = "auto",
        seed: int = 0,
        remove_target: bool = True,
        cache_size: int = 200_000,
        target_feature_dim: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.hops = int(hops)
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_label = int(max_label)
        self.seed = int(seed)
        self.remove_target = bool(remove_target)
        self.cache_size = int(cache_size)
        self._cache: dict[tuple[int, int], Subgraph] = {}

        self.feature_mode = self._resolve_mode(feature_mode)
        self.target_feature_dim = target_feature_dim
        self._neighbors = graph.neighbors()
        if self.feature_mode == "attribute":
            self._features = self._standardize(graph.x)
            self.feature_dim = int(self._features.shape[1])
        else:
            self._features = None
            self.feature_dim = self.max_label + 1

        # Cross-dataset transfer: project onto the dimension the checkpoint was
        # trained with so that node attributes never need to be compatible.
        if target_feature_dim is not None and int(target_feature_dim) != self.feature_dim:
            self._features = self._project_to(self._features, int(target_feature_dim))
            self.feature_dim = int(target_feature_dim)

    # ------------------------------------------------------------------ setup
    def _resolve_mode(self, mode: str) -> str:
        if mode not in {"auto", "attribute", "structural"}:
            raise ValueError(f"unknown feature_mode '{mode}'")
        if mode == "auto":
            return "attribute" if self.graph.has_features else "structural"
        if mode == "attribute" and not self.graph.has_features:
            raise ValueError(f"{self.graph.name} has no node attributes; use feature_mode='structural'")
        return mode

    def _project_to(self, features: Optional[np.ndarray], out_dim: int) -> Optional[np.ndarray]:
        """Random (Johnson-Lindenstrauss) projection to ``out_dim`` dimensions.

        Featureless datasets are represented by their DRNL one-hot encoding,
        zero-padded to ``out_dim``.
        """
        if features is None:
            n = self.graph.num_nodes
            padded = np.zeros((n, out_dim), dtype=np.float32)
            width = min(self.max_label + 1, out_dim)
            deg = self.graph.degrees()
            bins = np.minimum(deg, width - 1)
            padded[np.arange(n), bins] = 1.0
            return padded
        in_dim = features.shape[1]
        if in_dim == out_dim:
            return features
        rng = np.random.default_rng(self.seed)
        proj = rng.normal(0.0, 1.0 / np.sqrt(max(out_dim, 1)), size=(in_dim, out_dim)).astype(np.float32)
        return features @ proj

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (x - mean) / std

    # ------------------------------------------------------------------ build
    def build(self, src: int, dst: int) -> Subgraph:
        key = (int(src), int(dst)) if int(src) < int(dst) else (int(dst), int(src))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sub = extract_enclosing_subgraph(
            self._neighbors,
            key[0],
            key[1],
            hops=self.hops,
            max_nodes_per_hop=self.max_nodes_per_hop,
            max_label=self.max_label,
            features=self._features,
            remove_target=self.remove_target,
            seed=self.seed,
        )
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = sub
        return sub

    def build_many(self, pairs: np.ndarray) -> list[Subgraph]:
        return [self.build(int(u), int(v)) for u, v in np.asarray(pairs).reshape(-1, 2)]
