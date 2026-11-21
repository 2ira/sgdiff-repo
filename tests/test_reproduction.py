import copy

import torch

from sgdiff.engine.team_study import PRIMARY, study_plan
from sgdiff.engine.reproduction import COMPONENTS, OWNED, conditions, expanded
from sgdiff.models.transformer_denoiser import TransformerStructureDenoiser


def test_owned_components_and_target_list():
    assert set(OWNED) == {"katz", "sp4lp"}
    assert list(expanded(["gae"])) == ["gae", "gae_no_neighbors", "gae_no_attributes"]
    assert set(OWNED) <= set(COMPONENTS)
    assert len({c.name for c in conditions()}) == 21
    assert len(PRIMARY) == 8 and len(study_plan()) == 15
    assert list(expanded(["cn", "katz"])) == ["cn", "katz", "katz_h2", "katz_h3"]


def test_bounded_transformer_preserves_dense_valid_outputs_and_gradients():
    torch.manual_seed(1)
    dense = TransformerStructureDenoiser(8, node_attn=16, edge_attn=8, global_attn=8,
        heads=2, layers=2, node_mlp=16, edge_mlp=8, global_mlp=8, dropout=0).double()
    packed = copy.deepcopy(dense)
    packed.max_padded_pairs = 20
    mask = torch.arange(8)[None, :] < torch.tensor([3, 8, 4, 2, 3])[:, None]
    x = torch.randint(2, (5, 8, 8))
    label = torch.randint(8, (5, 8))
    cond = torch.randn(5, 8, dtype=torch.double)
    expected = dense(x, label, mask, cond)
    actual = packed(x, label, mask, cond)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    for outputs in (actual, expected):
        sum(t.square().sum() for t in outputs).backward()
    for a, b in zip(packed.parameters(), dense.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9)
