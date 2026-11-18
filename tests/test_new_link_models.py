"""Protocol invariants and numerical bridges for the optional author models."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('sklearn', reason='install experiments/new_link_models/requirements.txt')

from experiments.new_link_models.protocol import edge_keys, metrics, perturb, split_data
from sgdiff.data.graph import GraphData


def cache_worker(path):
    import time
    from experiments.new_link_models.protocol import cache_lock, save_npz
    path = Path(path)
    with cache_lock(path):
        if not path.exists():
            time.sleep(.1)
            save_npz(path, creator=np.array(os.getpid()))
            with path.with_suffix('.created').open('a') as f:
                f.write('created\n')


def test_concurrent_cache_is_created_once(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    path = tmp_path / 'shared.npz'
    workers = [ctx.Process(target=cache_worker, args=(str(path),)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0
    assert path.with_suffix('.created').read_text() == 'created\n'
    with np.load(path) as f:
        assert int(f['creator']) in {p.pid for p in workers}


def graph():
    edges = [(i, j) for i in range(24) for j in range(i + 1, 24) if (i + j) % 5 == 0]
    return GraphData('fixture', 24, edges, x=np.eye(24, dtype=np.float32), y=np.arange(24) % 3)


@pytest.mark.parametrize('attack', ['rf', 'dice', 'delete', 'delete_hub', 'insert', 'ea'])
def test_targets_protected_and_budget_realized(tmp_path, attack):
    g = graph()
    split, _ = split_data(g, 4, tmp_path)
    base = GraphData(g.name, g.num_nodes, np.vstack([split.train_pos, split.val_pos]), x=g.x, y=g.y)
    result, stats = perturb(base, split, attack, .5, 4)
    assert len(base.edge_set() ^ result.edge_set()) == round(.5 * base.num_edges)
    assert stats['effective_edits'] == stats['deleted'] + stats['inserted']
    blocked = set(edge_keys(np.vstack([split.test_pos, split.test_neg]), g.num_nodes))
    assert not blocked & set(edge_keys(result.edges, g.num_nodes))
    if attack == 'dice':
        assert all(g.y[u] == g.y[v] for u, v in base.edge_set() - result.edge_set())
        assert all(g.y[u] != g.y[v] for u, v in result.edge_set() - base.edge_set())
    again, _ = perturb(base, split, attack, .5, 4)
    np.testing.assert_array_equal(result.edges, again.edges)


def test_splits_disjoint_and_replayed(tmp_path):
    g = graph()
    split, sha = split_data(g, 7, tmp_path)
    assert sha == split_data(g, 7, tmp_path)[1]
    arrays = [getattr(split, part) for part in ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']]
    sets = [set(edge_keys(a, g.num_nodes)) for a in arrays]
    assert sum(map(len, sets)) == len(set.union(*sets))


def test_metrics_use_strict_hits_and_unordered_cutoff_ties():
    scores = np.ones(120)
    values = metrics(scores, scores)
    assert values['auc'] == values['ap'] == values['precision100'] == .5
    assert values['hits100'] == 0
    assert not values['hits100_trivial']
    assert metrics([1.], [0.])['hits100_trivial']


@pytest.mark.parametrize('in_dim,out_dim,weight', [(7, 3, True), (3, 7, True), (4, 4, False)])
def test_graphconv_bridge_matches_dgl_outputs_and_gradients(in_dim, out_dim, weight):
    dgl = pytest.importorskip('dgl')
    from dgl.nn import GraphConv
    from experiments.new_link_models.adapters import ExactGraphConv, RefinedGAE
    torch.manual_seed(3)
    original = GraphConv(in_dim, out_dim, weight=weight, bias=True).double()
    bridge = ExactGraphConv(copy.deepcopy(original))
    g = graph()
    e = torch.tensor(g.edges)
    dg = dgl.add_self_loop(dgl.graph((torch.cat([e[:, 0], e[:, 1]]), torch.cat([e[:, 1], e[:, 0]])), num_nodes=g.num_nodes))
    stub = type('Context', (), {'device': torch.device('cpu')})()
    ctx = RefinedGAE.context(stub, g)
    # Float64 normalization avoids float32 rounding obscuring gradient checks.
    adj = ctx['adjacency'].double()
    ix = adj.indices()
    degree = torch.bincount(ix[0], minlength=g.num_nodes).double()
    adj = torch.sparse_coo_tensor(ix, degree[ix[0]].rsqrt() * degree[ix[1]].rsqrt(), adj.shape)
    x1 = torch.randn(g.num_nodes, in_dim, dtype=torch.double, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = original(dg, x1), bridge(adj, x2)
    torch.testing.assert_close(y1, y2, atol=1e-12, rtol=1e-12)
    y1.square().sum().backward(); y2.square().sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-11, rtol=1e-11)
    for a, b in zip(original.parameters(), bridge.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-11)


def author_root():
    root = os.environ.get('NEW_LINK_SOURCE_ROOT')
    if root is None:
        pytest.skip('Set NEW_LINK_SOURCE_ROOT to verified optional author checkouts')
    pytest.importorskip('torch_sparse')
    return Path(root)


def test_ppr_cache_changes_with_graph(tmp_path):
    from experiments.new_link_models.adapters import ppr_matrix
    from experiments.new_link_models.sources import verify
    source = verify(author_root(), 'lpformer')
    g = graph()
    a = ppr_matrix(g, source, tmp_path, eps=1e-5)
    b = ppr_matrix(g, source, tmp_path, eps=1e-5)
    torch.testing.assert_close(a.to_dense(), b.to_dense())
    altered = g.drop_edges(g.edges[:4])
    c = ppr_matrix(altered, source, tmp_path, eps=1e-5)
    assert not torch.allclose(a.to_dense(), c.to_dense())
    assert len(list(tmp_path.glob('*.npz'))) == 2


@pytest.mark.parametrize('eps', [1e-5, 1e-7])
def test_ppr_array_backend_equals_author_reference(eps):
    import ast
    root = author_root()
    from experiments.new_link_models.ppr_backend import calc_ppr
    from experiments.new_link_models.sources import verify
    source = verify(root, 'lpformer')
    file = source / 'src/util/calc_ppr_scores.py'
    fn = next(n for n in ast.parse(file.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'get_calc_ppr')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), 'exec'), namespace)
    reference = namespace['get_calc_ppr']()
    rng = np.random.default_rng(21)
    e = np.array([(u, v) for u in range(45) for v in range(u + 1, 45) if rng.random() < .12])
    # Include three isolated vertices, as in Citeseer.
    g = GraphData('random', 48, e)
    a = g.adjacency()
    args = (a.indptr.astype(np.int64), a.indices.astype(np.int64), np.diff(a.indptr).astype(np.int64), .15, eps)
    expected, actual = reference(*args), calc_ppr(*args)
    for ref, tested in zip(expected, actual):
        for x, y in zip(ref, tested):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('model', ['ncnc', 'lpformer', 'refined_gae'])
def test_author_components_forward_and_backward(tmp_path, model):
    from experiments.new_link_models.adapters import ADAPTERS, VARIANTS
    root = author_root()
    g = graph()
    pairs = torch.tensor([[0, 2], [1, 4], [5, 6]])
    for variant in VARIANTS[model]:
        module = ADAPTERS[model](root, 'cora', g.x.shape[1], variant, 'cpu', tmp_path)
        module.train()
        output = module.score(module.context(g), pairs)
        assert output.shape == (3,) and torch.isfinite(output).all()
        output.square().mean().backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(p).all() for p in grads)
        if model == 'ncnc' and variant == 'no_completion':
            assert module.predictor.depth == 0
        if model == 'lpformer' and variant == 'uniform_attention':
            module.eval()
            att = module.core.att_layers[0].att
            edge_x, node_x = torch.randn(2, 256), torch.randn(24, 128)
            ei = torch.tensor([[0, 0, 1, 1, 1], [1, 2, 3, 4, 5]])
            _, weights = att(edge_x, node_x, ei, torch.randn(5, 128), return_att_weights=True)
            torch.testing.assert_close(weights, torch.tensor([[.5, .5, 1/3, 1/3, 1/3]]))
