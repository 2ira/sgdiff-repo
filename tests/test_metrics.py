"""Unit tests for the link-prediction metrics."""

import numpy as np

from sgdiff.utils.metrics import average_precision, hit_at_k, roc_auc, summarize_scores


def test_roc_auc_matches_pairwise_definition():
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.6, 0.4, 0.8, 0.1])
    # 9 of 9 positive/negative pairs are correctly ordered
    assert roc_auc(labels, scores) == 1.0

    scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.85])
    # positives beat 0, 2 and 2 negatives respectively
    expected = (0 + 2 + 2) / 9.0
    assert abs(roc_auc(labels, scores) - expected) < 1e-9


def test_roc_auc_handles_ties():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.9, 0.1])
    assert 0.0 <= roc_auc(labels, scores) <= 1.0


def test_average_precision_and_hit():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert average_precision(labels, scores) == 1.0
    assert hit_at_k(labels, scores, k=2) == 1.0
    assert hit_at_k(labels, scores, k=4) == 0.5


def test_summarize_scores_keys():
    metrics = summarize_scores(np.array([1, 0]), np.array([0.7, 0.2]))
    assert set(metrics) == {"auc", "ap", "hit@100"}
    assert metrics["auc"] == 1.0
