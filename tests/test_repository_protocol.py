"""Protocol parity and PubMed author-model forward/backward checks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('torch_sparse')
pytest.importorskip('sklearn')

from experiments.new_link_models import repository_protocol as rp
from experiments.new_link_models.adapters import VARIANTS
from experiments.new_link_models.repository_pubmed import PUBMED_ADAPTERS, pubmed_settings
from sgdiff.data.attack import apply_attack
from sgdiff.data.graph import GraphData
from sgdiff.data.splits import make_link_split


def tiny():
    edges = [(u, v) for u in range(24) for v in range(u + 1, 24) if (u + v) % 3 == 0]
    return GraphData('cora', 24, edges, x=np.eye(24, dtype=np.float32), y=np.arange(24) % 3)


def test_pubmed_uses_the_original_ea_size_limit():
    assert len(rp.plan(3327)) == 10
    assert len(rp.plan(19717)) == 7
    assert all(attack != 'ea' for attack, _ in rp.plan(19717))


def test_generated_cache_matches_main_dispatcher_before_and_after_test_removal(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    graph = tiny()
    monkeypatch.setattr(rp, 'load_graph', lambda *a, **kw: graph)
    # This unit checks dispatcher parity, not Git history availability (CI can
    # use a shallow checkout). Production prepare still verifies every pin.
    monkeypatch.setattr(rp, 'pinned_sources', lambda: {'fixture': 'main dispatcher in process'})
    args = argparse.Namespace(output=tmp_path, data_root=tmp_path, datasets=['cora'], seeds=[2], device='cpu')
    rp.prepare(args)
    split = make_link_split(graph, seed=2)
    for attack, ratio in rp.plan(graph.num_nodes):
        expected = graph if attack == 'clean' else apply_attack(graph, attack, ratio=ratio, seed=2, device='cpu')
        path = tmp_path / 'attacks' / f'cora_s2_{rp.tag(attack, ratio)}.npz'
        with np.load(path, allow_pickle=False) as f:
            assert np.array_equal(f['raw_edges'], expected.edges)
            assert np.array_equal(f['edges'], expected.drop_edges(split.test_pos.tolist()).edges)
            stats = json.loads(str(f['statistics']))
            assert stats['requested_budget'] == round(ratio * graph.num_edges)
            assert stats['protected_test_positives']
    _, _, pos, neg = rp.load_inputs(tmp_path, 'cora', 2)
    assert np.array_equal(pos, split.test_pos[:500])
    assert np.array_equal(neg, split.test_neg[:500])


@pytest.mark.parametrize('model,variant', [(m, v) for m in VARIANTS for v in VARIANTS[m]])
def test_pubmed_author_variants_have_finite_forward_and_gradients(tmp_path, model, variant):
    root = os.environ.get('NEW_LINK_SOURCE_ROOT')
    if not root:
        pytest.skip('set NEW_LINK_SOURCE_ROOT to pinned author repositories')
    pytest.importorskip('numba')
    if model == 'refined_gae':
        pytest.importorskip('dgl')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    graph = tiny()
    adapter = PUBMED_ADAPTERS[model](Path(root), 'pubmed', 24, variant, 'cpu', tmp_path)
    adapter.train()
    context = adapter.context(graph)
    pairs = torch.tensor([[0, 3], [1, 2], [0, 1], [1, 3]])
    scores = adapter.score(context, pairs)
    assert scores.shape == (4,) and torch.isfinite(scores).all()
    scores.sum().backward()
    grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if model == 'refined_gae':
        assert adapter.encoder.prop_step == 2
        assert len(adapter.encoder.alphas) == 2
    if model == 'lpformer':
        assert pubmed_settings(model)['ppr_eps'] == 1e-5
