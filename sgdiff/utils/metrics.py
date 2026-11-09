"""Link-prediction metrics: ROC-AUC, Average Precision and Hit@k.

The implementations are dependency-free (numpy only) so that they can be unit
tested and reused outside of the training loop.
"""

from __future__ import annotations

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve computed through the Mann-Whitney U statistic.

    Ties are handled by average ranks (equivalent to ``sklearn.metrics.roc_auc_score``).
    """
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
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
    """Average precision (area under the precision-recall curve, step rule)."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, sorted_labels.size + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def hit_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    """Fraction of the top-``k`` ranked candidates that are true links.

    Following the paper's protocol the candidate pool contains one positive per
    sampled negative, so ``Hit@100`` is the recall of the top-100 ranked items.
    When fewer than ``k`` items are available the metric is computed on the
    available pool.
    """
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels.size == 0:
        return float("nan")
    k = int(min(k, labels.size))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[order].sum() / max(k, 1))


def summarize_scores(labels: np.ndarray, scores: np.ndarray, k: int = 100) -> dict[str, float]:
    """Return all three paper metrics in one dictionary."""
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
