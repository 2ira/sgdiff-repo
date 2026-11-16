"""Course spectral and random-walk baselines on the shared link protocol."""

import numpy as np
import pytest

from sgdiff.config import load_config
from sgdiff.data.graph import GraphData
from sgdiff.engine.benchmark import BUDGET_PROFILES, build_registry, run_benchmark
from sgdiff.engine.course_baselines import build_course_model, score_course_model
from sgdiff.models.baselines.spectral_walk import (
    katz_scores,
    random_walk_scores,
    spectral_scores,
)


def test_katz_counts_walks_without_crossing_components():
    graph = GraphData("path", 4, np.asarray([[0, 1], [1, 2]], dtype=np.int64))
    scores = katz_scores(graph, np.asarray([[0, 2], [0, 3]], dtype=np.int64))
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_spectral_scores_are_finite_symmetric_and_deterministic():
    graph = GraphData(
        "components", 6,
        np.asarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int64),
    )
    pairs = np.asarray([[0, 1], [1, 0], [0, 4]], dtype=np.int64)
    first = spectral_scores(graph, pairs, dimensions=3)
    second = spectral_scores(graph, pairs, dimensions=3)
    assert np.all(np.isfinite(first))
    assert first[0] == pytest.approx(first[1])
    np.testing.assert_allclose(first, second, atol=1e-10)


@pytest.mark.parametrize("method", ["deepwalk", "node2vec"])
def test_random_walk_scores_are_reproducible(method):
    pytest.importorskip("gensim")
    graph = GraphData(
        "toy", 6,
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]], dtype=np.int64),
    )
    pairs = np.asarray([[0, 1], [0, 2], [1, 4]], dtype=np.int64)
    first = random_walk_scores(graph, pairs, method, seed=17)
    second = random_walk_scores(graph, pairs, method, seed=17)
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, atol=1e-10)


def test_registry_and_shared_benchmark_include_new_baselines(tmp_path):
    pytest.importorskip("gensim")
    cfg = load_config(None, BUDGET_PROFILES["smoke"])
    cfg.data.root = str(tmp_path / "data")
    names = ["katz", "spectral", "deepwalk", "node2vec"]
    registry = build_registry(cfg)
    assert all(registry[name].kind == "course" for name in names)
    result = run_benchmark(
        cfg, models=names, seeds=[0], node_pair_epochs=1,
        device="cpu", write=False, verbose=False,
    )
    assert set(result["summary"]) == set(names)
    assert len({row["split_fingerprint"] for row in result["runs"]}) == 1
    assert all(np.isfinite(row["metrics"]["auc"]) for row in result["runs"])


def test_course_wrapper_scores_every_new_baseline():
    pytest.importorskip("gensim")
    graph = GraphData(
        "toy", 6,
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]], dtype=np.int64),
    )
    pairs = np.asarray([[0, 2], [1, 4]], dtype=np.int64)
    labels = np.asarray([1, 0], dtype=np.int64)
    for name in ("katz", "spectral", "deepwalk", "node2vec"):
        model = build_course_model(name, 0).fit(graph, pairs, labels, seed=3)
        scores = score_course_model(model, graph, pairs)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
