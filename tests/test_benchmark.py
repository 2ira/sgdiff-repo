"""Tests for the data-identical benchmark harness and the PENCIL baseline.

The point of these tests is the *comparability contract*: every model in a
benchmark run must see the same split and the same enclosing sub-graphs, and the
new Transformer baseline must plug into the shared batch / trainer interfaces.
"""

import numpy as np
import pytest
import torch

from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.batch import collate_subgraphs
from sgdiff.engine.benchmark import (
    build_registry,
    run_benchmark,
    split_fingerprint,
    subgraph_fingerprint,
)
from sgdiff.engine.factory import prepare_datasets
from sgdiff.models.baselines import PENCILBaseline


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


def test_split_fingerprint_is_stable_and_seed_dependent(graph):
    first = make_link_split(graph, seed=0)
    again = make_link_split(graph, seed=0)
    other = make_link_split(graph, seed=1)
    assert split_fingerprint(first) == split_fingerprint(again)
    assert split_fingerprint(first) != split_fingerprint(other)


def test_split_fingerprint_ignores_pair_order(graph):
    split = make_link_split(graph, seed=0)
    expected = split_fingerprint(split)
    split.test_pos = np.asarray(split.test_pos)[:, ::-1].copy()
    split.train_neg = np.asarray(split.train_neg)[:, ::-1].copy()
    assert split_fingerprint(split) == expected


def test_subgraph_fingerprint_is_reproducible(tiny_config, graph):
    split = make_link_split(graph, seed=0)
    first = prepare_datasets(graph, split, tiny_config)
    second = prepare_datasets(graph, split, tiny_config)
    assert subgraph_fingerprint(first) == subgraph_fingerprint(second)


def test_subgraph_fingerprint_changes_with_extraction(tiny_config, graph):
    split = make_link_split(graph, seed=0)
    base = prepare_datasets(graph, split, tiny_config)
    tiny_config.subgraph.hops = 2
    try:
        deeper = prepare_datasets(graph, split, tiny_config)
    finally:
        tiny_config.subgraph.hops = 1
    assert subgraph_fingerprint(base) != subgraph_fingerprint(deeper)


def test_pencil_shapes_and_scores(tiny_config, graph):
    split = make_link_split(graph, seed=0)
    datasets = prepare_datasets(graph, split, tiny_config)
    dataset = datasets["train"]
    items = [dataset[i] for i in range(4)]
    batch = collate_subgraphs(items, feature_dim=dataset.feature_dim)

    model = PENCILBaseline(
        dataset.feature_dim,
        num_labels=tiny_config.subgraph.max_label + 1,
        hidden_dim=16,
        num_layers=1,
        heads=2,
    )
    model.eval()  # dropout must be off for the two forward passes to agree
    logits = model(batch)
    assert logits.shape == (4, 2)
    logit = model.predict_logit(batch)
    assert logit.shape == (4,)
    assert torch.allclose(logit, logits[:, 1] - logits[:, 0])
    proba = model.predict_proba(batch).detach()
    assert proba.shape == (4,)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_pencil_handles_endpoint_indices(tiny_config, graph):
    split = make_link_split(graph, seed=0)
    datasets = prepare_datasets(graph, split, tiny_config)
    batch = collate_subgraphs([datasets["train"][0]], feature_dim=datasets["train"].feature_dim)
    assert int(batch.u_idx[0]) != int(batch.v_idx[0])
    assert int(batch.nodes[0, batch.u_idx[0]]) == int(batch.src[0])
    assert int(batch.nodes[0, batch.v_idx[0]]) == int(batch.dst[0])


def test_capped_train_split_matches_subgraph_training_pairs(tiny_config, graph):
    """Full-graph baselines must see the same training pairs as the sub-graph models."""
    from sgdiff.engine.benchmark import capped_train_split
    from sgdiff.engine.factory import training_pairs

    split = make_link_split(graph, seed=tiny_config.data.seed)
    capped = capped_train_split(split, tiny_config)
    pairs, _ = training_pairs(split, tiny_config)
    assert len(capped.train_pos) + len(capped.train_neg) == len(pairs)
    merged = np.vstack([capped.train_pos, capped.train_neg])

    def as_set(a):
        return {tuple(map(int, row)) for row in np.sort(np.asarray(a, dtype=np.int64), axis=1)}

    assert as_set(merged) == as_set(pairs)
    # val/test are untouched
    assert np.array_equal(capped.test_pos, split.test_pos)


def test_registry_covers_paper_and_2026_baselines():
    cfg = load_config(None, [])
    registry = build_registry(cfg)
    for name in ("gcn", "gat", "sage", "vgae", "seal", "sgdiff", "pencil"):
        assert name in registry
    assert registry["pencil"].kind == "subgraph"
    assert registry["sgdiff"].kind == "sgdiff"


def test_benchmark_runs_every_model_on_the_same_data(tiny_config, tmp_path):
    result = run_benchmark(
        tiny_config,
        models=["sgdiff", "seal", "pencil", "gcn"],
        seeds=[0],
        subgraph_epochs=1,
        node_pair_epochs=4,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        write=False,
        quiet=True,
        verbose=False,
    )
    assert set(result["summary"]) == {"sgdiff", "seal", "pencil", "gcn"}
    integrity = result["data_integrity"]["0"]
    for run in result["runs"]:
        assert run["split_fingerprint"] == integrity["split_fingerprint"]
        assert run["subgraph_fingerprint"] == integrity["subgraph_fingerprint"]
        for metric in ("auc", "ap", "hit@100"):
            value = run["metrics"][metric]
            assert np.isnan(value) or 0.0 <= value <= 1.0
    assert "| model |" in result["markdown"]


def test_benchmark_rejects_unknown_model(tiny_config):
    with pytest.raises(KeyError):
        run_benchmark(tiny_config, models=["nope"], seeds=[0], write=False, verbose=False)
