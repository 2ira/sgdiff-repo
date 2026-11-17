import torch

from sgdiff.models.baselines import VGAE


def test_vgae_sampling_and_kl_describe_the_same_normal():
    model = VGAE(3)
    x = torch.ones(4, 3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    mean, log_std = model.encode(x, edges, 4)
    posterior = torch.distributions.Normal(mean, log_std.exp())
    prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    torch.manual_seed(7)
    expected = posterior.rsample()
    torch.manual_seed(7)
    sampled, kl = model(x, edges, 4)
    torch.testing.assert_close(sampled, expected)
    torch.testing.assert_close(kl, torch.distributions.kl_divergence(posterior, prior).sum(1).mean())
    model.eval()
    first, _ = model(x, edges, 4)
    second, _ = model(x, edges, 4)
    torch.testing.assert_close(first, mean)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
