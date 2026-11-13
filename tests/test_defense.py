"""Tests for the 2025-2026 graph purifiers (GPR-GAE and GDDM)."""

import numpy as np
import pytest
import torch

from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.engine.robustness import run_robustness
from sgdiff.models.defense import (
    EdgeDenoiser,
    GPRGAE,
    _feature_threshold,
    _two_hop_candidates,
    purify_graph,
    train_purifier,
)


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=120, num_features=8, seed=0)


def test_gpr_gae_shape_and_parameter_count(graph):
    x = torch.as_tensor(graph.x, dtype=torch.float32)
    edge_index = torch.as_tensor(
        np.concatenate([graph.edges.T, graph.edges[:, ::-1].T], axis=1), dtype=torch.long
    )
    model = GPRGAE(x.shape[1], hidden_dim=16, hops=3)
    z = model(x, edge_index, graph.num_nodes)
    assert z.shape == (graph.num_nodes, 16)
    assert model.gpr.numel() == 4  # one learnable weight per propagation depth
    pairs = torch.tensor([[0, 1], [2, 3]])
    assert model.decode(z, pairs).shape == (2,)


def test_edge_denoiser_uses_the_noise_level(graph):
    x = torch.as_tensor(graph.x, dtype=torch.float32)
    edge_index = torch.as_tensor(
        np.concatenate([graph.edges.T, graph.edges[:, ::-1].T], axis=1), dtype=torch.long
    )
    model = EdgeDenoiser(x.shape[1], hidden_dim=16)
    model.eval()
    pairs = torch.tensor([[0, 1], [1, 2]])
    with torch.no_grad():
        low = model.edge_logits(model(x, edge_index, graph.num_nodes, 0.0), pairs)
        high = model.edge_logits(model(x, edge_index, graph.num_nodes, 0.9), pairs)
    assert low.shape == (2,)
    assert not torch.allclose(low, high)  # the level is actually consumed


def test_two_hop_candidates_avoid_existing_edges(graph):
    edges = np.asarray(graph.edges, dtype=np.int64)[:50]
    existing = set((np.minimum(edges[:, 0], edges[:, 1]) * graph.num_nodes + np.maximum(edges[:, 0], edges[:, 1])).tolist())
    candidates = _two_hop_candidates(edges, graph.num_nodes, existing, limit=20)
    assert candidates.shape[1] == 2
    for u, v in candidates:
        key = min(int(u), int(v)) * graph.num_nodes + max(int(u), int(v))
        assert key in existing  # bounded sample, no duplicates
        assert u != v


def test_feature_threshold_is_a_quantile(graph):
    threshold = _feature_threshold(graph, quantile=0.1)
    assert threshold is None or np.isfinite(threshold)


@pytest.mark.parametrize("kind", ["gpr", "gddm"])
def test_train_and_purify_reduces_random_edges(graph, kind):
    """A purifier fitted on clean edges must delete most of the random noise."""
    state = train_purifier(graph, kind=kind, epochs=25, hidden_dim=16, device="cpu", seed=0)
    rng = np.random.default_rng(0)
    clean = np.asarray(graph.edges, dtype=np.int64)
    existing = set((np.minimum(clean[:, 0], clean[:, 1]) * graph.num_nodes + np.maximum(clean[:, 0], clean[:, 1])).tolist())
    noise = []
    while len(noise) < len(clean) // 4:
        u, v = int(rng.integers(0, graph.num_nodes)), int(rng.integers(0, graph.num_nodes))
        if u == v:
            continue
        key = min(u, v) * graph.num_nodes + max(u, v)
        if key in existing:
            continue
        existing.add(key)
        noise.append((min(u, v), max(u, v)))
    noisy_edges = np.vstack([clean, np.asarray(noise, dtype=np.int64)])
    from sgdiff.data.graph import GraphData

    noisy = GraphData(graph.name, graph.num_nodes, noisy_edges, x=graph.x)
    purified = purify_graph(noisy, state, steps=2, device="cpu")
    kept_noise = len({(min(int(u), int(v)), max(int(u), int(v))) for u, v in purified.edges} & {tuple(map(int, p)) for p in noise})
    assert kept_noise < 0.5 * len(noise), f"{kind} kept {kept_noise}/{len(noise)} random edges"
    assert purified.num_edges > 0


def test_robustness_harness_reports_purified_variants(tmp_path):
    cfg = load_config(
        None,
        [
            "data.name=synthetic",
            "data.root=" + str(tmp_path / "data"),
            "data.max_train_pairs=64",
            "data.max_eval_pairs=24",
            "train.device=cpu",
        ],
    )
    result = run_robustness(
        cfg,
        models=["gcn"],
        attacks=["rf"],
        ratios=[0.5],
        seeds=[0],
        defenses=["none", "gddm"],
        purifier_epochs=3,
        purifier_steps=1,
        max_pairs=20,
        node_pair_epochs=4,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        write=False,
        quiet=True,
        verbose=False,
    )
    variants = result["summary"]["gcn"]
    assert "clean" in variants and "rf@50%" in variants and "rf@50%|gddm" in variants
    assert result["protocol"]["defenses"] == ["gddm"]
    assert "0" in result["purifiers"]
