"""Equation-level regression tests for SGDiff's conditional feature DDPM."""

import torch
import torch.nn.functional as F

from sgdiff.models.feature_diffusion import FeatureDiffusionModel


def batch(dim=7):
    torch.manual_seed(0)
    x = torch.randn(2, 4, dim)
    adj = torch.zeros(2, 4, 4)
    adj[:, 0, 1] = adj[:, 1, 0] = 1
    labels = torch.tensor([[1, 1, 2, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    y = torch.tensor([0, 1])
    t = torch.tensor([1, 3])
    return x, adj, labels, mask, y, t


def test_feature_forward_matches_equation_11():
    x, adj, labels, mask, y, t = batch()
    model = FeatureDiffusionModel(7, hidden_dim=5, num_steps=4)
    actual = model(x, adj, labels, mask, y, t)
    a = adj + torch.diag_embed(mask.to(adj.dtype))
    inv = a.sum(-1).clamp(min=1).rsqrt()
    a = a * inv.unsqueeze(-1) * inv.unsqueeze(-2)
    condition = F.one_hot(y, 2).float()[:, None, :].expand(-1, 4, -1)
    inputs = torch.cat([x * mask[..., None], condition], -1) * mask[..., None]
    hidden = F.relu(a @ (inputs @ model.layers[0].weight.T))
    expected = (a @ (hidden @ model.layers[1].weight.T)) * mask[..., None]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_feature_has_no_drnl_dependency_and_masks_padding():
    x, adj, labels, mask, y, t = batch()
    model = FeatureDiffusionModel(7, hidden_dim=5, num_steps=4)
    expected = model(x, adj, labels, mask, y, t)
    x[~mask] = 1e8
    adj[:, 3, :] = adj[:, :, 3] = 1
    actual = model(x, adj, torch.full_like(labels, 999), mask, y, t)
    assert torch.allclose(expected, actual)
    assert torch.count_nonzero(actual[~mask]) == 0


def test_feature_noising_matches_eq10():
    x, adj, labels, mask, y, t = batch()
    model = FeatureDiffusionModel(7, num_steps=4)
    noise = torch.randn_like(x)
    noisy, sampled = model.q_sample(x, t, noise)
    alpha = model.schedule.alphas_bar[t - 1, None, None]
    assert torch.equal(sampled, noise)
    assert torch.allclose(noisy, alpha.sqrt() * x + (1 - alpha).sqrt() * noise)


def test_feature_classes_share_noise_exactly():
    x, adj, labels, mask, y, t = batch()
    model = FeatureDiffusionModel(7, num_steps=4)
    with torch.no_grad():
        model.layers[0].weight[:, 7:9] = 0
    scores = model.class_scores(x, adj, labels, mask, num_samples=4)
    assert torch.equal(scores[:, 0], scores[:, 1])


def test_feature_is_permutation_equivariant():
    x, adj, labels, mask, y, t = batch()
    model = FeatureDiffusionModel(7, hidden_dim=5, num_steps=4)
    p = torch.tensor([2, 0, 3, 1])
    expected = model(x, adj, labels, mask, y, t)[:, p]
    actual = model(x[:, p], adj[:, p][:, :, p], labels[:, p], mask[:, p], y, t)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_feature_high_dimensional_noise_is_trainable_without_identity_skip():
    torch.manual_seed(3)
    model = FeatureDiffusionModel(64, hidden_dim=16, num_steps=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.zeros(8, 3, 64)
    adj = torch.zeros(8, 3, 3)
    labels = torch.ones(8, 3, dtype=torch.long)
    mask = torch.ones(8, 3, dtype=torch.bool)
    y = torch.zeros(8, dtype=torch.long)
    t = torch.full((8,), 8, dtype=torch.long)
    noise = torch.randn_like(x)
    initial = model.compute_mse(x, adj, labels, mask, y, t, noise).mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = model.compute_mse(x, adj, labels, mask, y, t, noise).mean()
        loss.backward()
        optimizer.step()
    final = model.compute_mse(x, adj, labels, mask, y, t, noise).mean().item()
    assert final < initial * 0.75
