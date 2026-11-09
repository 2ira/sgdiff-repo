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
    """Randomly split the links and sample an equal number of negatives."""
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

    taken = {tuple(sorted(map(int, e))) for e in edges}

    def sample(count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 2), dtype=np.int64)
        if max_negatives is not None:
            count = min(count, int(max_negatives))
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        max_tries = max(50 * count, 10_000)
        tries = 0
        while len(out) < count and tries < max_tries:
            tries += 1
            u = int(rng.integers(0, graph.num_nodes))
            v = int(rng.integers(0, graph.num_nodes))
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in taken or key in seen:
                continue
            seen.add(key)
            out.append(key)
        if len(out) < count:
            raise RuntimeError(
                f"could only sample {len(out)}/{count} negatives for a graph with "
                f"{graph.num_nodes} nodes and {graph.num_edges} edges"
            )
        return np.asarray(out, dtype=np.int64)

    return LinkSplit(
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        train_neg=sample(int(neg_ratio * len(train_pos))),
        val_neg=sample(int(neg_ratio * len(val_pos))),
        test_neg=sample(int(neg_ratio * len(test_pos))),
        seed=seed,
        meta={"val_ratio": val_ratio, "test_ratio": test_ratio, "neg_ratio": neg_ratio},
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
