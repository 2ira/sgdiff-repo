"""Graph container used by every loader in :mod:`sgdiff.data`.

The whole implementation is numpy/scipy based: sub-graph extraction only needs
neighbour lists, and the diffusion models operate on dense padded sub-graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp


def unique_undirected_edges(
    edges: np.ndarray, num_nodes: int, remove_self_loops: bool = True
) -> np.ndarray:
    """Return sorted, de-duplicated, undirected edges with ``u < v``."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if remove_self_loops:
        e = e[e[:, 0] != e[:, 1]]
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    keys = lo * max(num_nodes, 1) + hi
    _, idx = np.unique(keys, return_index=True)
    return np.stack([lo[idx], hi[idx]], axis=1)


@dataclass
class GraphData:
    """An undirected, unweighted graph with optional node attributes."""

    name: str
    num_nodes: int
    edges: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    _neighbors: Optional[list] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.num_nodes = int(self.num_nodes)
        self.edges = unique_undirected_edges(self.edges, self.num_nodes)
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=np.float32)
            if self.x.shape[0] != self.num_nodes:
                raise ValueError(
                    f"feature matrix has {self.x.shape[0]} rows but the graph "
                    f"has {self.num_nodes} nodes"
                )
        if self.y is not None:
            self.y = np.asarray(self.y).ravel()

    # ------------------------------------------------------------------ stats
    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def has_features(self) -> bool:
        return self.x is not None and self.x.ndim == 2

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1]) if self.has_features else 0

    @property
    def average_degree(self) -> float:
        return 2.0 * self.num_edges / max(self.num_nodes, 1)

    # ------------------------------------------------------------- structures
    def adjacency(self, symmetric: bool = True) -> sp.csr_matrix:
        """Binary CSR adjacency (both directions by default)."""
        e = self.edges
        rows = np.concatenate([e[:, 0], e[:, 1]]) if symmetric else e[:, 0]
        cols = np.concatenate([e[:, 1], e[:, 0]]) if symmetric else e[:, 1]
        data = np.ones(rows.size, dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))
        return adj.tocsr()

    def neighbors(self) -> list[np.ndarray]:
        """List of neighbour arrays indexed by node id (cached)."""
        if self._neighbors is None:
            adj = self.adjacency()
            self._neighbors = [
                adj.indices[adj.indptr[i] : adj.indptr[i + 1]] for i in range(self.num_nodes)
            ]
        return self._neighbors

    def degrees(self) -> np.ndarray:
        return np.asarray([len(nb) for nb in self.neighbors()], dtype=np.int64)

    def edge_set(self) -> set:
        return {(int(u), int(v)) for u, v in self.edges}

    # ------------------------------------------------------------------ edits
    def drop_edges(self, drop: Iterable) -> "GraphData":
        """Return a copy without the given edges (direction agnostic)."""
        drop_set = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in drop}
        keep = np.asarray([(int(u), int(v)) not in drop_set for u, v in self.edges], dtype=bool)
        return GraphData(
            name=self.name,
            num_nodes=self.num_nodes,
            edges=self.edges[keep],
            x=self.x,
            y=self.y,
            meta=dict(self.meta),
        )

    def add_edges(self, new_edges: np.ndarray) -> "GraphData":
        """Return a copy with additional edges (self-loops dropped)."""
        merged = np.vstack([self.edges, np.asarray(new_edges, dtype=np.int64).reshape(-1, 2)])
        return GraphData(
            name=self.name,
            num_nodes=self.num_nodes,
            edges=merged,
            x=self.x,
            y=self.y,
            meta=dict(self.meta),
        )
