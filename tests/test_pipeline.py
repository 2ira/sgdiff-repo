"""End-to-end smoke tests: training loop, checkpoints and the public API."""

import numpy as np
import pytest
import torch

from sgdiff import api
from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.baseline_trainer import node_feature_matrix
from sgdiff.engine.evaluator import score_pairs
from sgdiff.engine.factory import build_model, prepare_datasets
from sgdiff.engine.trainer import Trainer, load_checkpoint
from sgdiff.models.baselines import SEALBaseline


@pytest.fixture(scope="module")
def tiny_config():
    return load_config(
        None,
        [
            "data.name=synthetic",
            "data.max_train_pairs=96",
            "data.max_eval_pairs=40",
            "subgraph.max_nodes_per_hop=8",
            "structure.num_steps=4",
            "feature.num_steps=4",
            "structure.hidden_dim=16",
            "feature.hidden_dim=16",
            "structure.num_layers=1",
            "feature.num_layers=1",
            "structure.num_mc_samples=2",
            "feature.num_mc_samples=2",
            "train.epochs=2",
            "train.pretrain_epochs=1",
            "train.batch_size=16",
            "train.eval_batch_size=16",
            "train.patience=5",
            "train.device=cpu",
        ],
    )


@pytest.fixture(scope="module")
def prepared(tiny_config, tmp_path_factory):
    graph = make_synthetic(num_nodes=200, num_features=12, seed=0)
    split = make_link_split(graph, seed=0)
    datasets = prepare_datasets(graph, split, tiny_config)
    model = build_model(tiny_config, datasets["train"].feature_dim)
    tiny_config.train.run_dir = str(tmp_path_factory.mktemp("runs"))
    trainer = Trainer(model, tiny_config)
    result = trainer.fit(datasets["train"], datasets["val"], datasets["test"])
    return {"graph": graph, "split": split, "datasets": datasets, "model": model,
            "trainer": trainer, "result": result, "config": tiny_config}


def test_training_produces_valid_metrics(prepared):
    result = prepared["result"]
    assert "test" in result
    for metric in ("auc", "ap", "hit@100"):
        value = result["test"][metric]
        assert np.isnan(value) or 0.0 <= value <= 1.0


def test_metrics_jsonl_and_checkpoint_exist(prepared):
    trainer = prepared["trainer"]
    assert (trainer.run_dir / "metrics.jsonl").exists()
    assert trainer.best_path.exists()


def test_checkpoint_roundtrip(prepared):
    model, _ = load_checkpoint(prepared["trainer"].best_path, device="cpu")
    dataset = prepared["datasets"]["test"]
    pairs = dataset.pairs[:12]
    before = score_pairs(prepared["model"], dataset.extractor, pairs, batch_size=8)
    after = score_pairs(model, dataset.extractor, pairs, batch_size=8)
    assert np.allclose(before, after, atol=1e-5)


def test_predict_proba_range(prepared):
    dataset = prepared["datasets"]["test"]
    probs = api.predict(
        prepared["model"], prepared["graph"], dataset.pairs[:8], prepared["config"],
        eval_graph=dataset.extractor.graph,
    )
    assert probs.shape == (8,)
    assert np.all(np.isfinite(probs))


def test_api_evaluate(prepared):
    dataset = prepared["datasets"]["test"]
    n = 10
    metrics = api.evaluate(
        prepared["model"],
        prepared["graph"],
        dataset.pairs[dataset.labels == 1][:n],
        dataset.pairs[dataset.labels == 0][:n],
        prepared["config"],
        eval_graph=dataset.extractor.graph,
    )
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["num_pos"] == n


def test_seal_baseline_runs(prepared):
    dataset = prepared["datasets"]["train"]
    model = SEALBaseline(dataset.feature_dim, hidden_dim=16, num_layers=1)
    batch = next(iter(torch.utils.data.DataLoader(
        dataset, batch_size=4,
        collate_fn=lambda items: __import__("sgdiff.engine.batch", fromlist=["x"]).collate_subgraphs(
            items, dataset.feature_dim),
    )))
    logits = model(batch)
    assert logits.shape == (4, 2)


def test_node_features_for_featureless_graph():
    graph = make_synthetic(num_nodes=60, num_features=8, seed=1)
    graph.x = None
    feats = node_feature_matrix(graph)
    assert feats.shape[0] == 60
    assert feats.shape[1] == 17  # 16 degree bins + scaled degree
