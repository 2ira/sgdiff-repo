"""Matrix-oracle and invariance regressions for categorical structure diffusion."""

import math
import types

import pytest
import torch
import torch.nn.functional as F

from sgdiff.models.schedules import UniformDiscreteSchedule
from sgdiff.models.structure_diffusion import StructureDiffusionModel
from sgdiff.models.transformer_denoiser import _masked_statistics


def _model(arch="edge_gnn", **kwargs):
    return StructureDiffusionModel(
        num_labels=4, hidden_dim=16, edge_dim=8, num_layers=2, num_steps=4,
        beta_start=0.15, beta_end=0.6, arch=arch, node_attn=16, edge_attn=8,
        global_attn=8, node_mlp=16, edge_mlp=16, global_mlp=16, heads=4,
        dropout=0, **kwargs,
    )


def _inputs():
    adj = torch.tensor([[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]])
    labels = torch.tensor([[1, 0, 3]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    return adj, labels, mask


@pytest.mark.parametrize("noise_schedule", ["linear", "cosine"])
@pytest.mark.parametrize("marginal", [[0.5, 0.5], [0.8, 0.2], [0.1, 0.2, 0.3, 0.4]])
def test_categorical_posterior_matches_matrix_oracle(marginal, noise_schedule):
    model = _model(noise_schedule=noise_schedule)
    marginal = torch.tensor(marginal)
    k = marginal.numel()
    eye = torch.eye(k)
    prior_matrix = marginal.unsqueeze(0).expand(k, -1)
    for step in range(1, model.num_steps + 1):
        q_prev = eye.clone()
        for beta in model.schedule.betas[:step - 1]:
            q_prev = q_prev @ ((1 - beta) * eye + beta * prior_matrix)
        beta = model.schedule.betas[step - 1]
        q_step = (1 - beta) * eye + beta * prior_matrix
        for clean in range(k):
            for observed in range(k):
                numerator = q_prev[clean] * q_step[:, observed]
                oracle = numerator / numerator.sum()
                actual = model.posterior_distribution(
                    eye[clean].reshape(1, 1, k), torch.tensor([[observed]]),
                    torch.tensor([step]), marginal,
                )[0, 0]
                assert torch.allclose(actual, oracle, atol=1e-6)
                if step == 1:
                    assert torch.equal(actual, eye[clean])


@pytest.mark.parametrize("noise_schedule", ["linear", "cosine"])
def test_schedule_cumulative_matches_product_with_empirical_prior(noise_schedule):
    schedule = UniformDiscreteSchedule(num_steps=5, beta_start=0.1, beta_end=0.7,
                                       noise_schedule=noise_schedule)
    marginal = torch.tensor([0.6, 0.3, 0.1])
    product = torch.eye(3)
    assert torch.equal(schedule.cumulative(torch.tensor([0]), marginal)[0], product)
    for step in range(1, 6):
        transition = schedule.transition(torch.tensor([step]), marginal)[0]
        product = product @ transition
        assert torch.allclose(schedule.cumulative(torch.tensor([step]), marginal)[0], product, atol=1e-6)
        assert torch.allclose(transition.sum(-1), torch.ones(3))


def test_binary_posterior_state1_matches_exact_matrix():
    model = _model(edge_marginal=torch.tensor([0.7, 0.3]))
    clean = torch.tensor([[[0., 0.], [1., 1.]]])
    observed = torch.tensor([[[0., 1.], [0., 1.]]])
    for step in [1, 2, 4]:
        t = torch.tensor([step])
        prev = model.schedule.cumulative(t - 1, model.edge_marginal)[0]
        q = model.schedule.transition(t, model.edge_marginal)[0]
        oracle = torch.empty_like(clean)
        for i in range(2):
            for j in range(2):
                numerator = prev[int(clean[0, i, j])] * q[:, int(observed[0, i, j])]
                oracle[0, i, j] = (numerator / numerator.sum())[1]
        assert torch.allclose(model.posterior_state1(clean, observed, t), oracle, atol=1e-6)


def test_q_sample_symmetric_zero_diagonal_and_padding():
    model = _model()
    adj = torch.ones(2, 5, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    sampled = model.q_sample(adj, torch.tensor([1, 4]), mask, torch.rand_like(adj))
    assert torch.equal(sampled, sampled.transpose(1, 2))
    assert torch.count_nonzero(sampled.diagonal(dim1=1, dim2=2)) == 0
    pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    assert torch.count_nonzero(sampled[~pair]) == 0
    assert torch.all((sampled == 0) | (sampled == 1))


def test_node_diffusion_empirical_prior_and_valid_zero():
    model = _model()
    model.set_marginals(node_marginal=[0, 0, 0, 5], edge_marginal=[9, 1])
    model.schedule = UniformDiscreteSchedule(num_steps=4, beta_start=1, beta_end=1)
    labels = torch.tensor([[0, 1, 2, 3, -99]])
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    sampled = model.q_sample_nodes(labels, torch.tensor([1]), mask, torch.full((1, 5), 0.2))
    assert torch.equal(sampled, torch.tensor([[3, 3, 3, 3, 0]]))
    assert torch.allclose(model.edge_marginal, torch.tensor([0.9, 0.1]))
    restored = _model()
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.node_marginal, model.node_marginal)
    with pytest.raises(ValueError):
        model.set_marginals(edge_marginal=[0, 0])
    with pytest.raises(ValueError):
        model.set_marginals(node_marginal=[1, -1, 1, 1])


def test_reverse_uses_predicted_clean_distribution_not_logits_as_reverse():
    model = _model()
    t = torch.tensor([3])
    node_logits = torch.tensor([[[0., 2., -1., 1.]]])
    edge_logits = torch.tensor([[[1.5]]])
    labels_t = torch.tensor([[2]])
    x_t = torch.tensor([[[0.]]])
    node, edge = model.reverse_probabilities(node_logits, edge_logits, labels_t, x_t, t)
    for prediction, observation, marginal, actual in [
        (node_logits.softmax(-1)[0, 0], 2, model.node_marginal, node[0, 0]),
        (torch.tensor([0., 1.5]).softmax(-1), 0, model.edge_marginal, edge[0, 0, 0]),
    ]:
        prev = model.schedule.cumulative(t - 1, marginal)[0]
        one_step = model.schedule.transition(t, marginal)[0]
        oracle = (prediction @ prev) * one_step[:, observation]
        oracle /= oracle.sum()
        assert torch.allclose(actual, oracle, atol=1e-6)
        assert torch.allclose(actual.sum(), torch.tensor(1.))
        assert not torch.allclose(actual, prediction)


def test_oracle_clean_denoiser_has_zero_node_and_edge_kl():
    model = _model()
    adj, labels, mask = _inputs()

    def clean_forward(self, x_t, labels_t, mask, y, t):
        return F.one_hot(labels, 4).float() * 160 - 80, adj * 160 - 80

    model.forward = types.MethodType(clean_forward, model)
    for step in [1, 2, 4]:
        kl = model.compute_kl(adj, labels, mask, torch.tensor([0]), torch.tensor([step]))
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_score_includes_node_kl_even_when_edges_are_perfect():
    model = _model(normalize_score=False)
    adj, labels, mask = _inputs()
    t = torch.tensor([1])

    def imperfect_nodes(self, *args):
        return torch.zeros(1, 3, 4), adj * 160 - 80

    model.forward = types.MethodType(imperfect_nodes, model)
    actual = model.compute_kl(adj, labels, mask, torch.tensor([0]), t, x_t=adj, labels_t=labels)
    p_node, _ = model.reverse_probabilities(torch.zeros(1, 3, 4), adj * 160 - 80, labels, adj, t)
    oracle = -p_node.gather(-1, labels.unsqueeze(-1)).log().sum((1, 2))
    assert actual.item() > 0
    assert torch.allclose(actual, oracle, atol=1e-6)
    model.normalize_score = True
    normalized = model.compute_kl(adj, labels, mask, torch.tensor([0]), t, x_t=adj, labels_t=labels)
    assert torch.allclose(normalized, oracle / 6, atol=1e-6)


def test_ce_trains_label_zero_and_masks_only_padding():
    model = _model(aux_ce_weight=0)
    adj = torch.zeros(1, 3, 3)
    labels = torch.tensor([[0, 1, -99]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    node_logits = torch.zeros(1, 3, 4, requires_grad=True)
    edge_logits = torch.zeros(1, 3, 3, requires_grad=True)

    def constant_forward(self, *args):
        return node_logits, edge_logits

    model.forward = types.MethodType(constant_forward, model)
    loss = model.loss(adj, labels, mask, torch.tensor([0]), t=torch.tensor([2]))
    assert torch.allclose(loss, torch.tensor(math.log(4) + math.log(2)))
    loss.backward()
    assert node_logits.grad[0, 0, 0] < 0
    assert node_logits.grad[0, 1, 1] < 0
    assert torch.count_nonzero(node_logits.grad[0, 2]) == 0
    assert edge_logits.grad[0, 0, 1] > 0
    assert torch.count_nonzero(edge_logits.grad[0, 1:]) == 0


@pytest.mark.parametrize("arch", ["edge_gnn", "transformer"])
def test_denoiser_permutation_equivariance_and_padding(arch):
    model = _model(arch).eval()
    adj, labels, mask = _inputs()
    y, t = torch.tensor([1]), torch.tensor([2])
    original_node, original_edge = model(adj, labels, mask, y, t)
    permutation = torch.tensor([2, 0, 1])
    perm_node, perm_edge = model(adj[:, permutation][:, :, permutation], labels[:, permutation],
                                 mask[:, permutation], y, t)
    assert torch.allclose(perm_node, original_node[:, permutation], atol=2e-6)
    assert torch.allclose(perm_edge, original_edge[:, permutation][:, :, permutation], atol=2e-6)
    padded_adj = F.pad(adj, (0, 2, 0, 2))
    padded_labels = F.pad(labels, (0, 2), value=-99)
    padded_mask = F.pad(mask, (0, 2), value=False)
    padded_node, padded_edge = model(padded_adj, padded_labels, padded_mask, y, t)
    assert torch.allclose(padded_node[:, :3], original_node, atol=2e-6)
    assert torch.allclose(padded_edge[:, :3, :3], original_edge, atol=2e-6)
    assert torch.count_nonzero(padded_node[:, 3:]) == 0
    assert torch.count_nonzero(padded_edge[:, 3:]) == 0
    assert torch.equal(original_edge, original_edge.transpose(1, 2))


@pytest.mark.parametrize("arch", ["edge_gnn", "transformer"])
def test_shared_noise_class_scores_and_gradients(arch):
    model = _model(arch).eval()
    adj, labels, mask = _inputs()
    t = torch.tensor([[1], [4]])
    edge_noise = torch.rand(2, 1, 3, 3)
    edge_noise = 0.5 * (edge_noise + edge_noise.transpose(-1, -2))
    noise = (torch.rand(2, 1, 3), edge_noise)
    equal = model.class_scores(adj, labels, mask, t=t, noise=noise, conditions=torch.tensor([1, 1]))
    assert equal.shape == (1, 2)
    assert torch.equal(equal[:, 0], equal[:, 1])
    assert torch.all(equal <= 0)
    (-equal.sum()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())
    scores = model.class_scores(adj, labels, mask, t=t, noise=noise)
    permutation = torch.tensor([2, 0, 1])
    perm_noise = (noise[0][:, :, permutation], noise[1][:, :, permutation][:, :, :, permutation])
    permuted = model.class_scores(adj[:, permutation][:, :, permutation], labels[:, permutation],
                                  mask[:, permutation], t=t, noise=perm_noise)
    assert torch.allclose(scores, permuted, atol=2e-6)
    for cls in range(2):
        manual = torch.stack([
            -model.compute_kl(adj, labels, mask, torch.tensor([cls]), t[index],
                              noise=(noise[0][index], noise[1][index]))
            for index in range(2)
        ]).mean(0)
        assert torch.allclose(scores[:, cls], manual, atol=1e-6)


def test_transformer_global_updates_and_finite_empty_singleton_graphs():
    model = _model("transformer")
    adj = torch.zeros(2, 2, 2)
    labels = torch.zeros(2, 2, dtype=torch.long)
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    loss = model.loss(adj, labels, mask, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    for block in model.denoiser.blocks[:-1]:
        assert block.global_in.weight.grad is not None
        assert torch.isfinite(block.global_in.weight.grad).all()
    assert model.denoiser.blocks[0].node_global.weight.grad.abs().sum() > 0
    assert model.denoiser.blocks[-1].global_in.weight.grad is None
    stats = _masked_statistics(torch.ones(2, 2, 3), mask)
    assert torch.isfinite(stats).all()
    assert torch.equal(stats[1], torch.zeros(12))


def test_class_count_and_random_timestep_compatibility(monkeypatch):
    model = _model().eval()
    adj, labels, mask = _inputs()
    seen_timesteps = []
    original = model.q_sample_nodes

    def tracked_nodes(labels, t, mask=None, noise=None):
        seen_timesteps.append(t.clone())
        return original(labels, t, mask, noise)

    monkeypatch.setattr(model, "q_sample_nodes", tracked_nodes)
    scores = model.class_scores(adj, labels, mask, num_classes=2, num_samples=3, random_timesteps=True)
    assert scores.shape == (1, 2)
    assert len(seen_timesteps) == 3
    assert all(torch.all((t >= 1) & (t <= 4)) for t in seen_timesteps)
    assert torch.isfinite(scores).all()
    with pytest.raises(ValueError):
        model.class_scores(adj, labels, mask, num_classes=3)


def test_cosine_shape_terminal_prior_and_ce_hybrid_modes():
    model = _model(noise_schedule="cosine", loss_mode="ce", aux_ce_weight=7).eval()
    grid = torch.arange(5, dtype=torch.float64) / 4
    expected = torch.cos((grid + 0.008) / 1.008 * torch.pi / 2).square()
    expected = (expected / expected[0])[1:].float()
    expected[-1] = 0
    assert torch.allclose(model.schedule.alphas_bar, expected, atol=1e-6)
    assert model.schedule.betas[-1] == 1
    terminal = model.schedule.cumulative(torch.tensor([4]), model.node_marginal)[0]
    assert torch.allclose(terminal, model.node_marginal.expand(4, -1))
    adj, labels, mask = _inputs()
    noise = (torch.rand_like(labels.float()), torch.rand_like(adj))
    y, t = torch.tensor([0]), torch.tensor([2])
    ce = model.loss(adj, labels, mask, y, t=t, noise=noise)
    model.aux_ce_weight = 2
    assert torch.equal(model.loss(adj, labels, mask, y, t=t, noise=noise), ce)
    kl = model.compute_kl(adj, labels, mask, y, t=t, noise=noise).mean()
    model.loss_mode = "hybrid"
    hybrid = model.loss(adj, labels, mask, y, t=t, noise=noise)
    assert torch.allclose(hybrid, kl + 2 * ce, atol=1e-6)
    assert torch.isfinite(hybrid)
    hybrid.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("bad_t", [torch.tensor([0]), torch.tensor([5])])
def test_invalid_timesteps_rejected(bad_t):
    model = _model()
    adj, labels, mask = _inputs()
    with pytest.raises(ValueError):
        model.compute_kl(adj, labels, mask, torch.tensor([0]), bad_t)
