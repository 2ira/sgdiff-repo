import numpy as np
import pytest
import scipy.sparse as sp
import torch

from experiment import (Graph, LinkModel, cached_input, load_graph, make_split,
                        metrics, model_input, negative_edges, sparse_tensor)


def keys(edges, n):
    return set((edges[:, 0] * n + edges[:, 1]).tolist())


def test_disjoint_splits_and_non_edges():
    n = 20
    edges = np.array([(u, v) for u in range(n) for v in range(u + 1, n) if (u + v) % 3 == 0])
    s = make_split(n, edges, 3)
    positive = keys(edges, n)
    assert keys(s['train'], n) | keys(s['valid'], n) | keys(s['test'], n) == positive
    all_sets = [keys(s[k], n) for k in s]
    for i, a in enumerate(all_sets):
        for b in all_sets[i + 1:]:
            assert not a & b
    for k, v in s.items():
        assert len(v) == len(keys(v, n))
        assert (v[:, 0] < v[:, 1]).all()
        assert np.array_equal(v, make_split(n, edges, 3)[k])
    assert len(s['valid']) == int(len(edges) * .05)
    assert len(s['test']) == int(len(edges) * .15)
    for k in ['train', 'valid', 'test']:
        assert len(s[k]) == len(s[k + '_neg'])


def test_dense_graph_negative_sampling_rejects_impossible_request():
    e = np.array([[0, 1], [0, 2], [1, 2]])
    with pytest.raises(ValueError):
        negative_edges(3, e, 1, np.random.default_rng(0))


def test_real_graph_held_out_edges_are_absent_and_degree_uses_train_only(tmp_path):
    torch.set_num_threads(1)
    g = load_graph('Cora', 0, tmp_path)
    a = g.adjacency.to_dense().numpy()
    assert np.allclose(a, a.T)
    for part in ['valid', 'test']:
        e = g.split[part]
        assert (a[e[:, 0], e[:, 1]] == 0).all()
        assert (a[e[:, 1], e[:, 0]] == 0).all()
    e = g.split['train']
    degree = np.bincount(e.ravel(), minlength=g.n)
    assert np.array_equal(g.structural.argmax(1).numpy(), np.minimum(degree, 31))


def test_cached_gcn_matches_direct_formula_and_gradients():
    torch.manual_seed(5)
    a = sparse_tensor(sp.csr_matrix([[.5, .5, 0], [.5, .5, 0], [0, 0, 1]]))
    x = sparse_tensor(sp.csr_matrix([[1., 0], [0, 1], [1, 1]]))
    m = LinkModel(2, 4, 2, 0., True)
    cached = torch.sparse.mm(a, x).coalesce()
    result = m(cached, a)
    direct = a.to_dense() @ (F_relu(a.to_dense() @ x.to_dense() @ m.weight + m.bias) @ m.output.weight.T) + m.output.bias
    assert torch.allclose(result, direct, atol=1e-6)
    grad1 = torch.autograd.grad(result.square().sum(), m.weight, retain_graph=True)[0]
    grad2 = torch.autograd.grad(direct.square().sum(), m.weight)[0]
    assert torch.allclose(grad1, grad2, atol=1e-6)


def F_relu(x):
    return torch.relu(x)


def test_feature_only_has_no_dependency_on_adjacency():
    m = LinkModel(3, 4, 2, 0, False).eval()
    x = torch.randn(5, 3)
    assert torch.equal(m(x, torch.eye(5)), m(x, torch.ones(5, 5)))


def test_transfer_ablation_mask_and_projection_reproducible(tmp_path):
    g = load_graph('Cora', 7, tmp_path)
    h = load_graph('Cora', 7, tmp_path)
    assert torch.equal(g.projected, h.projected)
    structure = model_input(g, 'structure', 'transfer')
    feature = model_input(g, 'feature', 'transfer')
    combined = model_input(g, 'combined', 'transfer')
    assert structure.shape == feature.shape == combined.shape == (g.n, 96)
    assert torch.equal(structure[:, :32], combined[:, :32])
    assert torch.equal(feature[:, 32:], combined[:, 32:])
    assert not torch.count_nonzero(structure[:, 32:])
    assert not torch.count_nonzero(feature[:, :32])


def test_hits_ties_and_small_pool():
    assert metrics([2., 3.], np.zeros(100))['hits100'] == 1.
    assert metrics([0., 0.], np.zeros(100))['hits100'] == 0.
    assert metrics([-100.], np.zeros(99))['hits100'] == 1.
    assert metrics([2., 3.], [0., 1.])['auc'] == 1.
