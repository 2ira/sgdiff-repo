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
    for name in ("train", "val", "test"):
        negatives = split.negatives(name)
        assert negatives.shape[0] == split.positives(name).shape[0]
        assert all(tuple(sorted(map(int, e))) not in edges for e in negatives)


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
