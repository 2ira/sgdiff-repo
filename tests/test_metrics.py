"""Unit tests for the link-prediction metrics."""

import numpy as np
import pytest

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
    assert hit_at_k(np.array([1, 0, 0, 0]), scores, k=2) == 0.5  # precision, not recall


def test_summarize_scores_keys():
    metrics = summarize_scores(np.array([1, 0]), np.array([0.7, 0.2]))
    assert set(metrics) == {"auc", "ap", "hit@100"}
    assert metrics["auc"] == 1.0


def test_average_precision_groups_ties_independently_of_input_order():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert average_precision(labels, scores) == 0.5
    assert average_precision(labels[::-1], scores) == 0.5
    assert average_precision(np.array([1, 0, 1, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.5


def test_average_precision_matches_sklearn_with_ties():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.8, 0.5, 0.5, 0.2, 0.1])
    assert average_precision(labels, scores) == pytest.approx(
        sklearn_metrics.average_precision_score(labels, scores)
    )


@pytest.mark.parametrize("metric", [roc_auc, average_precision, hit_at_k])
@pytest.mark.parametrize("scores", [[np.nan, 0.5], [np.inf, 0.5], [-np.inf, 0.5]])
def test_metrics_reject_nonfinite_scores(metric, scores):
    with pytest.raises(ValueError, match="finite"):
        metric([0, 1], scores)


@pytest.mark.parametrize("metric", [roc_auc, average_precision, hit_at_k])
@pytest.mark.parametrize("labels", [[0.5, 1], [-1, 1], [0, 2], [np.nan, 1]])
def test_metrics_reject_invalid_binary_labels(metric, labels):
    with pytest.raises(ValueError, match="binary"):
        metric(labels, [0.2, 0.8])


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_hit_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        hit_at_k([1, 0], [0.8, 0.1], k=k)
