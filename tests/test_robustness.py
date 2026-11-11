"""Tests for the robustness (RQ3) harness: RF / DICE / EA against every model.

The contract under test:

* one dispatch point (:func:`sgdiff.data.attack.apply_attack`) with aliases, and
  a budget that both random flip and DICE actually respect;
* EA runs on small graphs and is refused (not silently degraded) on large ones;
* the harness scores the *clean* column and every attacked column with the same
  code, for all three model kinds (node-pair, sub-graph, diffusion).
"""

import numpy as np
import pytest

from sgdiff.config import load_config
from sgdiff.data.attack import (
    EA_MAX_NODES,
    apply_attack,
    normalize_attack,
    supports_attack,
)
from sgdiff.data.datasets import make_synthetic
from sgdiff.engine.robustness import run_robustness, variant_name


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory):
    return load_config(
        None,
        [
            "data.name=synthetic",
            "data.root=" + str(tmp_path_factory.mktemp("data")),
            "data.max_train_pairs=64",
            "data.max_eval_pairs=24",
            "subgraph.max_nodes_per_hop=6",
            "structure.num_steps=3",
            "feature.num_steps=3",
            "structure.hidden_dim=16",
            "feature.hidden_dim=16",
            "structure.num_layers=1",
            "feature.num_layers=1",
            "structure.num_mc_samples=2",
            "feature.num_mc_samples=2",
            "train.epochs=1",
            "train.pretrain_epochs=1",
            "train.batch_size=16",
            "train.eval_batch_size=16",
            "train.patience=5",
            "train.device=cpu",
        ],
    )


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=150, num_features=10, seed=0)


def test_normalize_attack_aliases():
    assert normalize_attack(None) is None
    assert normalize_attack("clean") is None
    assert normalize_attack("rf") == "random"
    assert normalize_attack("random") == "random"
    assert normalize_attack("DICE") == "dice"
    assert normalize_attack("ea") == "ea"
    assert normalize_attack("embedding") == "ea"
    with pytest.raises(ValueError):
        normalize_attack("pgd")


def test_apply_attack_clean_returns_the_same_graph(graph):
    assert apply_attack(graph, "none") is graph
    assert apply_attack(graph, "clean") is graph


def test_apply_attack_respects_the_budget(graph):
    ratio = 0.1
    budget = int(round(ratio * graph.num_edges))
    for method in ("rf", "dice", "ea"):
        attacked = apply_attack(graph, method, ratio=ratio, seed=0)
        meta = attacked.meta["attack"]
        assert meta["deleted"] + meta["inserted"] <= budget
        assert meta["deleted"] > 0 and meta["inserted"] > 0
        # an edge never appears twice and self-loops never appear
        assert len(attacked.edge_set()) == attacked.num_edges
        assert np.all(attacked.edges[:, 0] != attacked.edges[:, 1])


def test_ea_is_skipped_not_degraded_on_large_graphs(graph):
    assert supports_attack("ea", graph.num_nodes, max_nodes=EA_MAX_NODES)
    assert not supports_attack("ea", EA_MAX_NODES + 1, max_nodes=EA_MAX_NODES)
    assert supports_attack("dice", EA_MAX_NODES + 1, max_nodes=EA_MAX_NODES)
    with pytest.raises(ValueError):
        apply_attack(graph, "ea", ratio=0.1, max_nodes=graph.num_nodes - 1)


def test_variant_name():
    assert variant_name("clean", 0.1) == "clean"
    assert variant_name("rf", 0.25) == "rf@25%"
    assert variant_name("ea", 0.1) == "ea@10%"


def test_robustness_scores_three_attacks_for_three_model_kinds(tiny_config, graph, tmp_path):
    result = run_robustness(
        tiny_config,
        models=["gcn", "seal", "sgdiff"],
        attacks=["rf", "dice", "ea"],
        ratios=[0.05],
        seeds=[0],
        sgdiff_epochs=1,
        subgraph_epochs=1,
        node_pair_epochs=4,
        device="cpu",
        max_pairs=24,
        run_root=str(tmp_path / "runs"),
        write=False,
        quiet=True,
        verbose=False,
    )
    assert result["variants"] == ["clean", "rf@5%", "dice@5%", "ea@5%"]
    assert result["skipped"] == []
    assert set(result["attack_stats"]) == {"rf@5%", "dice@5%", "ea@5%"}
    for stats in result["attack_stats"].values():
        assert stats["deleted"] > 0 and stats["inserted"] > 0
    assert len(result["runs"]) == 3 * 4  # 3 models x (clean + 3 attacks)
    for run in result["runs"]:
        for metric in ("auc", "ap", "hit@100"):
            value = run["metrics"][metric]
            assert np.isnan(value) or 0.0 <= value <= 1.0
        if run["variant"] != "clean":
            assert set(run["drop"]) == {"auc", "ap", "hit@100"}
    assert set(result["summary"]) == {"gcn", "seal", "sgdiff"}
    assert "AUC ea@5%" in result["markdown"]


def test_robustness_records_the_ea_skip(tiny_config, graph, tmp_path):
    result = run_robustness(
        tiny_config,
        models=["gcn"],
        attacks=["rf", "ea"],
        ratios=[0.05],
        seeds=[0],
        node_pair_epochs=2,
        device="cpu",
        max_pairs=16,
        run_root=str(tmp_path / "runs"),
        ea_max_nodes=graph.num_nodes - 1,
        write=False,
        quiet=True,
        verbose=False,
    )
    assert result["skipped"] and result["skipped"][0]["variant"] == "ea@5%"
    assert "ea@5%" in result["variants"]
    assert not any(run["variant"] == "ea@5%" for run in result["runs"])
    assert result["summary"]["gcn"].get("ea@5%") is None
