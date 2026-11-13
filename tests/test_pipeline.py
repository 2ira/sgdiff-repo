"""End-to-end smoke tests: training loop, checkpoints and the public API."""

import numpy as np
import pytest
import torch

from sgdiff import api
from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.baseline_trainer import node_feature_matrix
from sgdiff.engine.evaluator import _evaluation_context, score_pairs, score_pairs_logits
from sgdiff.engine.factory import build_model, initialize_training_marginals, prepare_datasets
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
            "train.finetune_epochs=2",
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


def test_validation_context_contains_no_held_out_positive_edges(tiny_config):
    graph = make_synthetic(num_nodes=80, num_features=8, seed=17)
    split = make_link_split(graph, seed=17)
    datasets = prepare_datasets(graph, split, tiny_config, train_graph=graph, eval_graph=graph)
    val_edges = datasets["val"].extractor.graph.edge_set()
    test_edges = datasets["test"].extractor.graph.edge_set()
    train_edges = {tuple(pair) for pair in split.train_pos}
    held_out = {tuple(pair) for pair in np.vstack([split.val_pos, split.test_pos])}
    assert val_edges == train_edges
    assert not val_edges & held_out
    assert test_edges == train_edges | {tuple(pair) for pair in split.val_pos}
    assert not test_edges & {tuple(pair) for pair in split.test_pos}
    assert split.meta["preprocessing"]["feature_mode"] == "attribute"
    assert split.meta["preprocessing"]["feature_normalization"] == "none"
    assert datasets["train"].extractor.graph.meta["preprocessing"]["val_context"] == "train"


def test_factory_forwards_and_records_explicit_feature_normalization(tiny_config):
    import copy

    cfg = copy.deepcopy(tiny_config)
    cfg.subgraph.feature_normalization = "row"
    graph = make_synthetic(num_nodes=40, num_features=4, seed=19)
    split = make_link_split(graph, seed=19)
    datasets = prepare_datasets(graph, split, cfg)
    assert split.meta["preprocessing"]["feature_normalization"] == "row"
    for dataset in datasets.values():
        assert dataset.extractor.feature_normalization == "row"
        sub = dataset.extractor.build(*graph.edges[0])
        norm = np.abs(sub.x).sum(axis=1)
        assert np.allclose(norm, 1.0)


def test_training_marginals_count_valid_nodes_and_undirected_edges_only():
    from types import SimpleNamespace

    from sgdiff.data.graph import GraphData
    from sgdiff.data.subgraph import SubgraphExtractor
    from sgdiff.engine.batch import PairSubgraphDataset

    graph = GraphData("path", 5, np.array([[0, 2], [2, 3], [3, 1], [1, 4]]))
    extractor = SubgraphExtractor(graph, hops=1, max_nodes_per_hop=None, feature_mode="structural")
    dataset = PairSubgraphDataset(np.array([[0, 1]]), np.array([0]), extractor)
    captured = {}

    def set_marginals(node, edge):
        captured.update(node=node, edge=edge)

    model = SimpleNamespace(structure=SimpleNamespace(num_labels=17, set_marginals=set_marginals))
    metadata = initialize_training_marginals(model, dataset)
    assert metadata["num_pairs"] == 1
    assert metadata["node_counts"][:4] == [1, 2, 0, 2]
    assert sum(metadata["node_counts"]) == 5
    assert metadata["edge_counts"] == [6, 4]
    assert captured["node"][0].item() == pytest.approx(0.2)
    assert captured["edge"].tolist() == pytest.approx([0.6, 0.4])
    assert captured["node"].sum().item() == pytest.approx(1.0)
    assert initialize_training_marginals(SimpleNamespace(), dataset) == {}


class _RandomScorer(torch.nn.Module):
    def __init__(self, device="cpu", fail=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device=device))
        self.dropout = torch.nn.Dropout()
        self.fail = fail

    def predict_logit(self, batch, num_samples=None):
        values = torch.rand(batch.batch_size, device=self.anchor.device)
        if self.fail:
            raise RuntimeError("scoring failure")
        return values

    def predict_proba(self, batch, num_samples=None):
        return self.predict_logit(batch, num_samples)

    def forward(self, batch, num_samples=None):
        values = self.predict_logit(batch, num_samples)
        return {"logits": torch.stack([-values, values], dim=-1)}


def _rng_states():
    states = [torch.get_rng_state()]
    if torch.cuda.is_available():
        states.extend(torch.cuda.get_rng_state_all())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        states.append(torch.mps.get_rng_state())
    return states


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("score_fn", [score_pairs, score_pairs_logits])
def test_scoring_restores_all_rngs_and_mixed_training_modes(device, score_fn):
    from sgdiff.data.subgraph import SubgraphExtractor

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS unavailable")
    graph = make_synthetic(num_nodes=40, num_features=4, seed=9)
    extractor = SubgraphExtractor(graph, hops=1, max_nodes_per_hop=4)
    model = _RandomScorer(device).train()
    model.dropout.eval()
    before = _rng_states()
    first = score_fn(model, extractor, graph.edges[:3], device=device, num_samples=3)
    assert all(torch.equal(a, b) for a, b in zip(before, _rng_states()))
    second = score_fn(model, extractor, graph.edges[:3], device=device, num_samples=3)
    assert np.array_equal(first, second)
    assert model.training and not model.dropout.training
    assert model.anchor.requires_grad
    model.fail = True
    with pytest.raises(RuntimeError, match="scoring failure"):
        score_fn(model, extractor, graph.edges[:3], device=device)
    assert all(torch.equal(a, b) for a, b in zip(before, _rng_states()))
    assert model.training and not model.dropout.training


def test_evaluation_finally_restores_mocked_accelerator_rngs(monkeypatch):
    states = {"cuda": [torch.tensor([11], dtype=torch.uint8)], "mps": torch.tensor([23], dtype=torch.uint8)}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in states["cuda"]])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda value: states.update(cuda=value))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: states["mps"].clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda value: states.update(mps=value))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    model = _RandomScorer().train()
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="failure"):
        with _evaluation_context(model, seed=4):
            torch.rand(2)
            states["cuda"][0].fill_(99)
            states["mps"].fill_(99)
            raise RuntimeError("failure")
    assert torch.equal(before, torch.get_rng_state())
    assert states["cuda"][0].item() == 11
    assert states["mps"].item() == 23
    assert model.training


def test_factory_forwards_scoring_and_structure_method_variants(tiny_config):
    import copy

    cfg = copy.deepcopy(tiny_config)
    cfg.structure.noise_schedule = "linear"
    cfg.structure.loss_mode = "hybrid"
    cfg.feature.normalize_score = True
    model = build_model(cfg, feature_dim=4)
    assert model.structure.noise_schedule == "linear"
    assert model.structure.loss_mode == "hybrid"
    assert model.feature.normalize_score is True
    cfg.structure.noise_schedule = "cosine"
    cfg.structure.loss_mode = "ce"
    cfg.feature.normalize_score = False
    model = build_model(cfg, feature_dim=4)
    assert model.structure.noise_schedule == "cosine"
    assert model.structure.loss_mode == "ce"
    assert model.feature.normalize_score is False
