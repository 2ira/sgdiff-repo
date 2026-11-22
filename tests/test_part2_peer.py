import numpy as np
import torch

from sgdiff.data.graph import GraphData
from sgdiff.engine.baseline_trainer import _edge_index
from sgdiff.engine.part2_peer import build_gcn


def _graph(edges, features):
    return GraphData("tiny", len(features), np.asarray(edges, dtype=np.int64),
                     x=np.asarray(features, dtype=np.float32))


def test_gcn_input_ablation_invariances():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    graph_a = _graph([[0, 1], [1, 2]], x.numpy())
    graph_b = _graph([[0, 2], [2, 3]], x.numpy()[::-1].copy())

    no_neighbors = build_gcn(2, "gcn_no_neighbors").eval()
    first = no_neighbors.encode(x, _edge_index(graph_a, "cpu"), 4)
    second = no_neighbors.encode(x, _edge_index(graph_b, "cpu"), 4)
    torch.testing.assert_close(first, second)

    no_attributes = build_gcn(2, "gcn_no_attributes").eval()
    first = no_attributes.encode(x, _edge_index(graph_a, "cpu"), 4)
    second = no_attributes.encode(x.flip(0), _edge_index(graph_a, "cpu"), 4)
    torch.testing.assert_close(first, second)


def test_gcn_ablation_capacity_matches_full_hidden_widths():
    full = build_gcn(8, "gcn")
    no_neighbors = build_gcn(8, "gcn_no_neighbors")
    assert full.encoder.dims == no_neighbors.encoder.dims == [8, 64, 64]
