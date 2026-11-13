"""Tests for the diffusion components (schedules, posteriors, ELBO)."""

import torch

from sgdiff.models.feature_diffusion import FeatureDiffusionModel
from sgdiff.models.fusion import LikelihoodFusion
from sgdiff.models.schedules import GaussianSchedule, UniformDiscreteSchedule
from sgdiff.models.structure_diffusion import StructureDiffusionModel


def _binary_adj(b=2, n=6, p=0.4, seed=0):
    torch.manual_seed(seed)
    upper = torch.triu((torch.rand(b, n, n) < p).float(), diagonal=1)
    return upper + upper.transpose(1, 2)


def test_gaussian_schedule_shapes():
    schedule = GaussianSchedule(num_steps=10)
    assert schedule.alphas_bar.shape == (10,)
    assert torch.all(schedule.alphas_bar[1:] < schedule.alphas_bar[:-1])
    assert torch.allclose(schedule.alphas_bar_prev[1:], schedule.alphas_bar[:-1])


def test_uniform_schedule_first_step():
    schedule = UniformDiscreteSchedule(num_steps=5, beta_start=0.1, beta_end=0.1)
    assert abs(float(schedule.alphas_bar[0]) - 0.9) < 1e-6
    assert abs(float(schedule.alphas_bar[-1]) - 0.9**5) < 1e-6


def test_discrete_q_sample_flip_rate():
    schedule = UniformDiscreteSchedule(num_steps=4, beta_start=0.2, beta_end=0.2)
    model = StructureDiffusionModel(num_labels=8, hidden_dim=8, num_layers=1, num_steps=4,
                                    beta_start=0.2, beta_end=0.2)
    x0 = torch.ones(4000, 4, 4) - torch.eye(4).unsqueeze(0)
    t = torch.full((4000,), 3, dtype=torch.long)
    x_t = model.q_sample(x0, t)
    candidates = model.candidate_mask(torch.ones(4000, 4, dtype=torch.bool)).squeeze(-1)
    flip_rate = float((x_t != x0)[candidates].float().mean())
    expected = (1.0 - float(schedule.alphas_bar[2])) / 2
    assert abs(flip_rate - expected) < 0.03


def test_posterior_is_a_distribution():
    model = StructureDiffusionModel(num_labels=8, hidden_dim=8, num_layers=1, num_steps=10)
    x0 = _binary_adj()
    t = torch.tensor([1, 10])
    x_t = model.q_sample(x0, t)
    pq = model.posterior_state1(x0, x_t, t)
    assert torch.all(pq >= 0) and torch.all(pq <= 1)
    # at t=1 the posterior must be (almost) the clean sample
    assert torch.allclose(pq[0], x0[0], atol=1e-2)


def test_structure_loss_is_finite_and_differentiable():
    model = StructureDiffusionModel(num_labels=8, hidden_dim=16, num_layers=2, num_steps=5)
    adj = _binary_adj()
    labels = torch.randint(1, 6, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    y = torch.tensor([1, 0])
    loss = model.loss(adj, labels, mask, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)

    with torch.no_grad():
        ll = model.log_likelihood(adj, labels, mask, y, num_samples=3)
    assert ll.shape == (2,)
    assert torch.all(ll <= 0)


def test_feature_diffusion_forward_and_loss():
    model = FeatureDiffusionModel(feature_dim=5, num_labels=8, hidden_dim=16, num_layers=2, num_steps=5)
    x = torch.randn(2, 6, 5)
    adj = _binary_adj()
    labels = torch.randint(1, 6, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    y = torch.tensor([1, 0])

    loss = model.loss(x, adj, labels, mask, y)
    assert torch.isfinite(loss)
    loss.backward()

    with torch.no_grad():
        ll = model.log_likelihood(x, adj, labels, mask, y, num_samples=2)
    assert ll.shape == (2,)
    assert torch.all(ll <= 0)


def test_fusion_matches_closed_form():
    fusion = LikelihoodFusion(init_eta=2.0)
    feat = torch.tensor([[0.1, 0.5], [0.2, 0.3]])
    struct = torch.tensor([[0.4, 0.1], [0.0, 0.1]])
    logits = fusion(feat, struct)
    expected = 2.0 * feat + 2.0 * struct + fusion.log_prior
    assert torch.allclose(logits, expected, atol=1e-6)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
