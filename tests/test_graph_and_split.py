"""Tests for the graph container, the link split and the attack helpers."""

import numpy as np
import pytest

from sgdiff.data.attack import dice_attack, random_perturbation
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split, subsample_pairs


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=160, num_features=12, seed=0)


def test_graph_edges_are_unique_and_undirected(graph):
    edges = graph.edges
    assert edges.shape[0] == graph.num_edges
    assert np.all(edges[:, 0] < edges[:, 1])
    keys = {(int(u), int(v)) for u, v in edges}
    assert len(keys) == len(edges)


def test_adjacency_is_symmetric(graph):
    adj = graph.adjacency()
    assert (adj != adj.T).nnz == 0
    assert graph.neighbors()[0].size > 0


def test_split_is_disjoint_and_negatives_are_non_edges(graph):
    split = make_link_split(graph, seed=1)
    positives = np.vstack([split.train_pos, split.val_pos, split.test_pos])
    assert positives.shape[0] == graph.num_edges

    edges = {tuple(sorted(map(int, e))) for e in graph.edges}
    sampled = set()
    for name in ("train", "val", "test"):
        negatives = split.negatives(name)
        keys = {tuple(map(int, e)) for e in negatives}
        assert negatives.shape[0] == split.positives(name).shape[0]
        assert len(keys) == len(negatives)
        assert not keys & (edges | sampled)
        sampled.update(keys)


def test_subsample_pairs():
    pairs = np.arange(200).reshape(-1, 2)
    kept = subsample_pairs(pairs, 0.1, seed=0)
    assert kept.shape[0] == 10  # 10% of 100 pairs


def test_random_perturbation_keeps_schema(graph):
    attacked = random_perturbation(graph, ratio=0.05, seed=0)
    assert attacked.num_nodes == graph.num_nodes
    assert attacked.num_edges != graph.num_edges
    assert attacked.meta["attack"]["method"] == "random"


def test_dice_attack_runs(graph):
    attacked = dice_attack(graph, ratio=0.05, seed=0)
    assert attacked.num_nodes == graph.num_nodes
    assert attacked.meta["attack"]["method"] == "dice"


def test_dense_negative_capacity_is_checked_before_sampling():
    from sgdiff.data.graph import GraphData

    edges = np.column_stack(np.triu_indices(5, k=1))
    graph = GraphData("complete", 5, edges)
    with pytest.raises(ValueError, match="only 0 nonedges"):
        make_link_split(graph)


def test_dense_negative_sampling_exhausts_capacity_without_rejection():
    from sgdiff.data.graph import GraphData

    edges = np.column_stack(np.triu_indices(8, k=1))
    missing = {(0, 1), (2, 5), (6, 7)}
    edges = np.array([edge for edge in edges if tuple(edge) not in missing])
    graph = GraphData("dense", 8, edges)
    split = make_link_split(graph, val_ratio=0.2, test_ratio=0.2, max_negatives=1, seed=4)
    negatives = np.vstack([split.train_neg, split.val_neg, split.test_neg])
    assert {tuple(edge) for edge in negatives} == missing
    assert len(negatives) == 3
    again = make_link_split(graph, val_ratio=0.2, test_ratio=0.2, max_negatives=1, seed=4)
    assert np.array_equal(split.train_neg, again.train_neg)


def test_negative_capacity_is_global_across_splits():
    from sgdiff.data.graph import GraphData

    edges = np.column_stack(np.triu_indices(8, k=1))
    graph = GraphData("dense", 8, edges[2:])
    with pytest.raises(ValueError, match="requested 3 disjoint negatives but only 2"):
        make_link_split(graph, val_ratio=0.2, test_ratio=0.2, max_negatives=1)
