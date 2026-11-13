"""Link-prediction metrics: ROC-AUC, Average Precision and Hit@k.

The implementations are dependency-free (numpy only) so that they can be unit
tested and reused outside of the training loop.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must be binary values 0 or 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return labels.astype(np.int64), scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve computed through the Mann-Whitney U statistic.

    Ties are handled by average ranks (equivalent to ``sklearn.metrics.roc_auc_score``).
    """
    labels, scores = _validate_inputs(labels, scores)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = _average_ranks(sorted_scores)
    pos_rank_sum = ranks[labels[order] == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-based average precision, including tied-score groups (sklearn rule)."""
    labels, scores = _validate_inputs(labels, scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    threshold_end = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), scores.size - 1]
    tp = np.cumsum(sorted_labels)[threshold_end]
    precision = tp / (threshold_end + 1)
    recall_increments = np.diff(np.r_[0, tp]) / n_pos
    return float(np.sum(recall_increments * precision))


def hit_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    """Fraction of the top-``k`` ranked candidates that are true links.

    This is precision@k over the pooled candidates, retained under the legacy
    ``hit@k`` result key. It is not recall@k or OGB's per-positive Hits@k;
    no equivalence to the paper's Hit@100 protocol is asserted. Ties retain
    input order. With fewer than ``k`` items, use the available pool.
    """
    labels, scores = _validate_inputs(labels, scores)
    if int(k) != k or k <= 0:
        raise ValueError("k must be a positive integer")
    if labels.size == 0:
        return float("nan")
    k = int(min(k, labels.size))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[order].sum() / max(k, 1))


def summarize_scores(labels: np.ndarray, scores: np.ndarray, k: int = 100) -> dict[str, float]:
    """Return ROC-AUC, AP, and pooled precision@k under the legacy ``hit@k`` key."""
    return {
        "auc": roc_auc(labels, scores),
        "ap": average_precision(labels, scores),
        f"hit@{k}": hit_at_k(labels, scores, k),
    }


def _average_ranks(sorted_scores: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank."""
    n = sorted_scores.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks
