"""Tests for the robust GNN baselines (RGCN) and their robustness wiring."""

import numpy as np
import pytest
import torch

from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.benchmark import build_registry
from sgdiff.engine.baseline_trainer import node_feature_matrix, train_rgcn, _edge_index
from sgdiff.engine.robustness import run_robustness
from sgdiff.models.baselines import RGCN


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=120, num_features=8, seed=0)


def test_rgcn_is_deterministic_at_inference(graph):
    x = torch.as_tensor(node_feature_matrix(graph))
    edge_index = _edge_index(graph, "cpu")
    model = RGCN(x.shape[1], hidden_dim=16, out_dim=16, num_layers=2)
    model.eval()
    with torch.no_grad():
        first = model.encode(x, edge_index, graph.num_nodes)
        second = model.encode(x, edge_index, graph.num_nodes)
    assert torch.allclose(first, second)


def test_rgcn_samples_while_training(graph):
    x = torch.as_tensor(node_feature_matrix(graph))
    edge_index = _edge_index(graph, "cpu")
    model = RGCN(x.shape[1], hidden_dim=16, out_dim=16, num_layers=2)
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        first = model.encode(x, edge_index, graph.num_nodes)
        second = model.encode(x, edge_index, graph.num_nodes)
    assert not torch.allclose(first, second)


def test_rgcn_score_pairs_and_sigma(graph):
    x = torch.as_tensor(node_feature_matrix(graph))
    edge_index = _edge_index(graph, "cpu")
    model = RGCN(x.shape[1], hidden_dim=16, out_dim=16, num_layers=2)
    model.eval()
    z = model.encode(x, edge_index, graph.num_nodes)
    pairs = torch.tensor([[0, 1], [2, 3]])
    scores = model.score_pairs(z, pairs)
    assert scores.shape == (2,)
    assert model.last_sigma is not None and torch.all(model.last_sigma > 0)
    assert model.last_sigma.shape == z.shape


def test_train_rgcn_returns_metrics(graph):
    from sgdiff.data.splits import make_link_split

    split = make_link_split(graph, seed=0)
    x_dim = node_feature_matrix(graph).shape[1]
    model = RGCN(x_dim, hidden_dim=16, out_dim=16, num_layers=1, var_weight=0.1)
    metrics = train_rgcn(model, graph, split, device="cpu", epochs=4, seed=0)
    assert 0.0 <= metrics["auc"] <= 1.0
    assert "epochs_run" in metrics


def test_registry_exposes_robust_baselines():
    cfg = load_config(None, [])
    registry = build_registry(cfg)
    assert registry["rgcn"].kind == "robust_node_pair"
    assert registry["rgcn_novar"].kind == "robust_node_pair"
    # the ablation must differ from the full model only in the penalty
    x_dim = 8
    with_var = registry["rgcn"].builder(x_dim, cfg)
    without = registry["rgcn_novar"].builder(x_dim, cfg)
    assert with_var.var_weight > 0.0
    assert without.var_weight == 0.0


def test_variance_penalty_keeps_sigma_smaller(graph):
    """The penalty must actually act on sigma; otherwise the ablation is empty."""
    split = make_link_split(graph, seed=0)
    x_dim = node_feature_matrix(graph).shape[1]
    sigmas = {}
    for name, weight in (("with", 0.5), ("without", 0.0)):
        model = RGCN(x_dim, hidden_dim=16, out_dim=16, num_layers=1, var_weight=weight)
        torch.manual_seed(0)
        train_rgcn(model, graph, split, device="cpu", epochs=30, seed=0)
        model.eval()
        with torch.no_grad():
            model.encode(
                torch.as_tensor(node_feature_matrix(graph)), _edge_index(graph, "cpu"), graph.num_nodes
            )
        sigmas[name] = float(model.last_sigma.mean())
    assert sigmas["with"] < sigmas["without"]


def test_robustness_harness_runs_the_new_kind(tmp_path):
    cfg = load_config(
        None,
        [
            "data.name=synthetic",
            "data.root=" + str(tmp_path / "data"),
            "data.max_train_pairs=64",
            "data.max_eval_pairs=24",
            "subgraph.max_nodes_per_hop=6",
            "train.device=cpu",
        ],
    )
    result = run_robustness(
        cfg,
        models=["rgcn", "gcn"],
        attacks=["rf"],
        ratios=[0.5],
        seeds=[0],
        max_pairs=20,
        node_pair_epochs=4,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        write=False,
        quiet=True,
        verbose=False,
    )
    assert set(result["summary"]) == {"rgcn", "gcn"}
    for name in ("rgcn", "gcn"):
        variants = result["summary"][name]
        assert "clean" in variants and "rf@50%" in variants
        clean = variants["clean"]["auc"]["mean"]
        attacked = variants["rf@50%"]["auc"]["mean"]
        assert np.isfinite(clean) and np.isfinite(attacked)
        assert "drop_mean" in variants["rf@50%"]["auc"]
