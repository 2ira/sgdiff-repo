"""Train/val/test link splits and negative sampling.

Protocol (paper Sec. 4.1 / Appendix A.2): existing links are split into
80% / 5% / 15% train/valid/test; an equal number of unconnected node pairs is
sampled as negatives for each split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .graph import GraphData


@dataclass
class LinkSplit:
    """Positive/negative pairs for the three splits."""

    train_pos: np.ndarray
    val_pos: np.ndarray
    test_pos: np.ndarray
    train_neg: np.ndarray
    val_neg: np.ndarray
    test_neg: np.ndarray
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def positives(self, split: str) -> np.ndarray:
        return getattr(self, f"{split}_pos")

    def negatives(self, split: str) -> np.ndarray:
        return getattr(self, f"{split}_neg")

    def pairs(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pairs, labels)`` with positives first."""
        pos = self.positives(split)
        neg = self.negatives(split)
        pairs = np.vstack([pos, neg]) if len(neg) else pos
        labels = np.concatenate(
            [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
        )
        return pairs, labels

    def summary(self) -> dict:
        return {
            split: {
                "pos": int(len(self.positives(split))),
                "neg": int(len(self.negatives(split))),
            }
            for split in ("train", "val", "test")
        }


def make_link_split(
    graph: GraphData,
    val_ratio: float = 0.05,
    test_ratio: float = 0.15,
    neg_ratio: float = 1.0,
    seed: int = 0,
    max_negatives: Optional[int] = None,
) -> LinkSplit:
    """Split links and sample mutually disjoint negatives without rejection loops."""
    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("validation/test ratios must be nonnegative and sum to less than one")
    if not np.isfinite(neg_ratio) or neg_ratio < 0:
        raise ValueError("neg_ratio must be finite and nonnegative")
    if max_negatives is not None and (int(max_negatives) != max_negatives or max_negatives < 0):
        raise ValueError("max_negatives must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    edges = graph.edges.copy()
    rng.shuffle(edges)

    n_val = int(round(val_ratio * len(edges)))
    n_test = int(round(test_ratio * len(edges)))
    n_train = len(edges) - n_val - n_test
    if n_train <= 0:
        raise ValueError("not enough edges for a 80/5/15 split")

    train_pos = edges[:n_train]
    val_pos = edges[n_train : n_train + n_val]
    test_pos = edges[n_train + n_val :]

    counts = [int(neg_ratio * len(pos)) for pos in (train_pos, val_pos, test_pos)]
    if max_negatives is not None:
        counts = [min(count, int(max_negatives)) for count in counts]
    total = sum(counts)
    n = graph.num_nodes
    capacity = n * (n - 1) // 2 - len(edges)
    if total > capacity:
        raise ValueError(f"requested {total} disjoint negatives but only {capacity} nonedges exist")

    # Rank nonedges in the strict upper triangle; invert by a bounded binary
    # search over the forbidden edge ranks rather than rejecting dense edges.
    rows = np.arange(n, dtype=np.int64)
    row_start = rows * (2 * n - rows - 1) // 2
    forbidden = np.sort(row_start[edges[:, 0]] + edges[:, 1] - edges[:, 0] - 1)
    ranks = rng.choice(capacity, size=total, replace=False).astype(np.int64)
    low, high = ranks.copy(), ranks + len(forbidden)
    while np.any(low < high):
        mid = low + (high - low) // 2
        available = mid + 1 - np.searchsorted(forbidden, mid, side="right")
        move_right = available < ranks + 1
        low = np.where(move_right, mid + 1, low)
        high = np.where(move_right, high, mid)
    u = np.searchsorted(row_start, low, side="right") - 1
    v = low - row_start[u] + u + 1
    negatives = np.column_stack([u, v]).astype(np.int64)
    train_end, val_end = counts[0], counts[0] + counts[1]

    return LinkSplit(
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        train_neg=negatives[:train_end],
        val_neg=negatives[train_end:val_end],
        test_neg=negatives[val_end:],
        seed=seed,
        meta={
            "val_ratio": val_ratio, "test_ratio": test_ratio, "neg_ratio": neg_ratio,
            "negative_splits_disjoint": True, "nonedge_capacity": int(capacity),
        },
    )


def subsample_pairs(pairs: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """Randomly keep ``ratio`` of ``pairs`` (used by the 1% train-size study)."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    n = max(1, int(round(ratio * len(pairs))))
    if n >= len(pairs):
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=n, replace=False)
    return pairs[np.sort(idx)]
