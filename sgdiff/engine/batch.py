"""Mini-batch construction for sub-graph models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.subgraph import Subgraph, SubgraphExtractor


@dataclass
class SubgraphBatch:
    """A padded batch of enclosing sub-graphs."""

    adj: torch.Tensor  # (B, n, n) clean induced adjacency without the target edge
    x: torch.Tensor  # (B, n, F) node features (attributes or DRNL one-hot)
    labels: torch.Tensor  # (B, n) DRNL labels
    mask: torch.Tensor  # (B, n) bool
    y: torch.Tensor  # (B,) link labels
    src: torch.Tensor  # (B,) global node ids
    dst: torch.Tensor  # (B,) global node ids
    nodes: torch.Tensor  # (B, n) global ids of the sub-graph nodes (0 = padding)
    u_idx: torch.Tensor  # (B,) local index of the source endpoint
    v_idx: torch.Tensor  # (B,) local index of the destination endpoint

    @property
    def batch_size(self) -> int:
        return int(self.adj.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.adj.shape[1])

    def to(self, device: torch.device | str) -> "SubgraphBatch":
        return SubgraphBatch(
            adj=self.adj.to(device),
            x=self.x.to(device),
            labels=self.labels.to(device),
            mask=self.mask.to(device),
            y=self.y.to(device),
            src=self.src.to(device),
            dst=self.dst.to(device),
            nodes=self.nodes.to(device),
            u_idx=self.u_idx.to(device),
            v_idx=self.v_idx.to(device),
        )


def collate_subgraphs(
    items: Sequence[tuple[Subgraph, int]],
    feature_dim: Optional[int] = None,
) -> SubgraphBatch:
    """Pad a list of ``(subgraph, label)`` into a :class:`SubgraphBatch`."""
    subs = [item[0] for item in items]
    ys = [int(item[1]) for item in items]
    b = len(subs)
    n = max(s.num_nodes for s in subs)

    if feature_dim is None:
        feature_dim = subs[0].x.shape[1] if subs[0].x is not None else int(subs[0].labels.max()) + 1

    adj = np.zeros((b, n, n), dtype=np.float32)
    x = np.zeros((b, n, feature_dim), dtype=np.float32)
    labels = np.zeros((b, n), dtype=np.int64)
    mask = np.zeros((b, n), dtype=bool)
    nodes = np.zeros((b, n), dtype=np.int64)
    u_idx = np.zeros(b, dtype=np.int64)
    v_idx = np.zeros(b, dtype=np.int64)

    for i, sub in enumerate(subs):
        k = sub.num_nodes
        adj[i, :k, :k] = sub.adj
        labels[i, :k] = sub.labels
        mask[i, :k] = True
        nodes[i, :k] = sub.nodes
        u_idx[i] = sub.u_idx
        v_idx[i] = sub.v_idx
        if sub.x is not None:
            x[i, :k] = sub.x
        else:  # structural features: one-hot DRNL label
            x[i, np.arange(k), sub.labels] = 1.0

    return SubgraphBatch(
        adj=torch.from_numpy(adj),
        x=torch.from_numpy(x),
        labels=torch.from_numpy(labels),
        mask=torch.from_numpy(mask),
        y=torch.as_tensor(ys, dtype=torch.long),
        src=torch.as_tensor([s.src for s in subs], dtype=torch.long),
        dst=torch.as_tensor([s.dst for s in subs], dtype=torch.long),
        nodes=torch.from_numpy(nodes),
        u_idx=torch.as_tensor(u_idx, dtype=torch.long),
        v_idx=torch.as_tensor(v_idx, dtype=torch.long),
    )


class PairSubgraphDataset(Dataset):
    """Lazily extracts the enclosing sub-graph of every ``(u, v)`` pair."""

    def __init__(
        self,
        pairs: np.ndarray,
        labels: np.ndarray,
        extractor: SubgraphExtractor,
    ) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.labels = np.asarray(labels, dtype=np.int64).ravel()
        if self.pairs.shape[0] != self.labels.shape[0]:
            raise ValueError("pairs and labels have different lengths")
        self.extractor = extractor

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int) -> tuple[Subgraph, int]:
        u, v = self.pairs[index]
        sub = self.extractor.build(int(u), int(v))
        if sub.x is None:
            x = np.zeros((sub.num_nodes, self.extractor.feature_dim), dtype=np.float32)
            x[np.arange(sub.num_nodes), sub.labels] = 1.0
            sub.x = x
        return sub, int(self.labels[index])

    @property
    def feature_dim(self) -> int:
        return int(self.extractor.feature_dim)
