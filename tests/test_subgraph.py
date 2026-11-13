"""Tests for enclosing sub-graph extraction (SEAL-style DRNL)."""

import numpy as np
import pytest

from sgdiff.data.datasets import make_synthetic
from sgdiff.data.subgraph import SubgraphExtractor, drnl_label, extract_enclosing_subgraph


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=120, num_features=8, seed=3)


def test_drnl_labels_are_positive_and_monotone():
    assert drnl_label(0, 0, 16) == 1
    assert drnl_label(0, 1, 16) == 2 or drnl_label(0, 1, 16) == 1
    assert drnl_label(0, 8, 16) <= 16


def test_target_edge_is_removed(graph):
    u, v = (int(x) for x in graph.edges[5])
    sub = extract_enclosing_subgraph(graph.neighbors(), u, v, hops=2, max_nodes_per_hop=20)
    assert sub.adj[sub.u_idx, sub.v_idx] == 0.0
    assert sub.adj[sub.v_idx, sub.u_idx] == 0.0
    assert u in sub.nodes and v in sub.nodes


def test_keep_target_edge_option(graph):
    u, v = (int(x) for x in graph.edges[5])
    sub = extract_enclosing_subgraph(
        graph.neighbors(), u, v, hops=2, max_nodes_per_hop=20, remove_target=False
    )
    assert sub.adj[sub.u_idx, sub.v_idx] == 1.0


def test_nodes_respect_hop_limit(graph):
    """Every extracted node must be within ``hops`` hops of u or v (edge removed)."""
    u, v = (int(x) for x in graph.edges[7])
    hops = 2
    sub = extract_enclosing_subgraph(graph.neighbors(), u, v, hops=hops, max_nodes_per_hop=30)

    neighbors = [set(nb.tolist()) for nb in graph.neighbors()]
    neighbors[u].discard(v)
    neighbors[v].discard(u)

    for node in sub.nodes.tolist():
        node = int(node)
        seen = {node: 0}
        frontier = [node]
        for hop in range(1, hops + 1):
            nxt = []
            for cur in frontier:
                for nb in neighbors[cur]:
                    if nb not in seen:
                        seen[nb] = hop
                        nxt.append(nb)
            frontier = nxt
        assert u in seen or v in seen, f"node {node} is farther than {hops} hops"


def test_budget_is_enforced(graph):
    u, v = (int(x) for x in graph.edges[3])
    cap, hops = 3, 2
    sub = extract_enclosing_subgraph(graph.neighbors(), u, v, hops=hops, max_nodes_per_hop=cap)
    # two roots, at most ``cap`` new nodes per hop each
    assert sub.num_nodes <= 2 + 2 * hops * cap


def test_extractor_is_deterministic(graph):
    extractor = SubgraphExtractor(graph, hops=2, max_nodes_per_hop=15, seed=0)
    u, v = (int(x) for x in graph.edges[11])
    a = extractor.build(u, v)
    b = extractor.build(v, u)  # order independent (cached, sorted pair)
    assert np.array_equal(a.nodes, b.nodes)
    assert np.array_equal(a.adj, b.adj)


def test_feature_modes(graph):
    attr = SubgraphExtractor(graph, hops=2, feature_mode="attribute")
    assert attr.feature_dim == graph.num_features
    struct = SubgraphExtractor(graph, hops=2, feature_mode="structural")
    assert struct.feature_dim == struct.max_label + 1


def test_drnl_removes_opposite_endpoint_and_labels_unreachable_zero():
    from sgdiff.data.graph import GraphData

    graph = GraphData("path", 5, np.array([[0, 2], [2, 3], [3, 1], [1, 4]]))
    sub = extract_enclosing_subgraph(graph.neighbors(), 0, 1, hops=1, max_nodes_per_hop=None)
    labels = dict(zip(sub.nodes.tolist(), sub.labels.tolist()))
    assert labels == {0: 1, 1: 1, 2: 3, 3: 3, 4: 0}
    assert drnl_label(np.inf, 1, 16) == 0


def test_total_cap_fills_budget_without_dropping_endpoints():
    from sgdiff.data.graph import GraphData

    graph = GraphData("star", 6, np.array([[0, 2], [0, 3], [1, 4], [1, 5]]))
    sub = extract_enclosing_subgraph(
        graph.neighbors(), 0, 1, hops=1, max_nodes_per_hop=None, max_nodes_total=4
    )
    assert sub.num_nodes == 4
    assert {0, 1} <= set(sub.nodes.tolist())
    assert np.all(sub.labels[2:] == 0)


def test_drnl_uses_final_capped_subgraph_only():
    from sgdiff.data.graph import GraphData

    graph = GraphData("path", 4, np.array([[0, 2], [2, 3], [3, 1]]))
    sub = extract_enclosing_subgraph(
        graph.neighbors(), 0, 1, hops=2, max_nodes_per_hop=None, max_nodes_total=3
    )
    assert sub.num_nodes == 3
    assert sub.labels[sub.u_idx] == sub.labels[sub.v_idx] == 1
    assert sub.labels[(sub.nodes != 0) & (sub.nodes != 1)].tolist() == [0]


@pytest.mark.parametrize("normalization", ["none", "standardize", "row"])
def test_attribute_normalization_is_explicit(normalization):
    from sgdiff.data.graph import GraphData

    features = np.array([[1.0, 3.0], [0.0, 0.0], [-2.0, 2.0], [5.0, 5.0]], dtype=np.float32)
    graph = GraphData("features", 4, np.array([[0, 2], [2, 3], [3, 1]]), x=features)
    extractor = SubgraphExtractor(graph, feature_normalization=normalization)
    actual = extractor.build(0, 1).x
    assert extractor.feature_normalization == normalization
    if normalization == "none":
        assert np.array_equal(actual, features)
    elif normalization == "standardize":
        assert np.allclose(actual.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(actual.std(axis=0), 1.0, atol=1e-6)
    else:
        assert np.allclose(np.abs(actual).sum(axis=1), [1, 0, 1, 1])
        assert np.array_equal(actual[1], [0, 0])
    assert np.array_equal(graph.x, features)


def test_attribute_normalization_defaults_to_none(graph):
    extractor = SubgraphExtractor(graph)
    sub = extractor.build(*graph.edges[0])
    assert extractor.feature_normalization == "none"
    assert np.array_equal(sub.x, graph.x[sub.nodes])


def test_unknown_attribute_normalization_is_rejected(graph):
    with pytest.raises(ValueError, match="feature_normalization"):
        SubgraphExtractor(graph, feature_normalization="implicit")
