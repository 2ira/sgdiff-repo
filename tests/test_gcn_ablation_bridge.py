"""Numerical contract between the archived GAE baseline and shared GCN."""
import json

import pytest
import torch

pytest.importorskip('sklearn', reason='install experiments/gcn_ablation/requirements.txt')

from experiments.gcn_ablation.shared_backend import (
    ReferenceLinkModel, SharedLinkModel, configure,
)
from sgdiff.models.baselines.node_pair import GCNEncoder
from sgdiff.models.layers import sparse_normalize


def adjacency():
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    row, col, values = sparse_normalize(edges, 4)
    return edges, torch.sparse_coo_tensor(torch.stack([row, col]), values, (4, 4)).coalesce()


@pytest.mark.parametrize('propagate,sparse', [(True, False), (True, True), (False, False), (False, True)])
def test_shared_matches_reference_outputs_gradients_and_dropout_rng(propagate, sparse):
    _, a = adjacency()
    x = torch.tensor([[1., 0., 2.], [0., 1., 0.], [2., 0., 1.], [1., 1., 0.]])
    if propagate:
        x = a @ x
    if sparse:
        x = x.to_sparse()
    torch.manual_seed(71)
    reference = ReferenceLinkModel(3, 5, 2, .3, propagate)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    shared = SharedLinkModel(3, 5, 2, .3, propagate)
    assert torch.equal(expected_rng, torch.get_rng_state())
    pairs = torch.tensor([[0, 1], [0, 3], [1, 2]])
    torch.manual_seed(32)
    expected = reference.score(reference(x, a), pairs)
    expected.square().sum().backward()
    torch.manual_seed(32)
    actual = shared.score(shared(x, a), pairs)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    first, second = shared.predictor.encoder.lins
    for got, want in [(first.weight.grad.T, reference.weight.grad),
                      (first.bias.grad, reference.bias.grad),
                      (second.weight.grad, reference.output.weight.grad),
                      (second.bias.grad, reference.output.bias.grad)]:
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)


def test_preaggregation_matches_existing_forward_with_nonzero_biases():
    edges, a = adjacency()
    torch.manual_seed(4)
    encoder = GCNEncoder(3, 5, 2, num_layers=3, dropout=.3).eval()
    x = torch.randn(4, 3)
    torch.testing.assert_close(encoder(x, edges, 4), encoder.forward_preaggregated(a @ x, a))


def test_identity_input_stays_sparse_and_feature_only_ignores_graph():
    _, a = adjacency()
    model = SharedLinkModel(4, 5, 2, 0., True).eval()
    first, second = model.predictor.encoder.lins
    explicit = a @ (torch.relu(a @ first.weight.T + first.bias) @ second.weight.T) + second.bias
    torch.testing.assert_close(model(a, a), explicit)
    model.propagate = False
    torch.testing.assert_close(model(a, a), model(a, torch.eye(4)))


def test_provenance_separates_archived_and_shared_outputs(tmp_path):
    out = tmp_path / 'shared'
    configure(['--output', str(out)])
    configure(['--output', str(out)], replay=True)
    marker = out / 'backend.json'
    record = json.loads(marker.read_text())
    record['backend'] = 'another-model'
    marker.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='Backend changed'):
        configure(['--output', str(out)])
    archived = tmp_path / 'archived'
    archived.mkdir()
    (archived / 'config.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Missing shared backend'):
        configure(['--output', str(archived)])
