"""Tests for the likelihood ablations (GCCM shortcut check on SGDiff)."""

import numpy as np
import pytest
import torch

from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.ablation import (
    AblatedScorer,
    Ablation,
    evaluate_ablation,
    run_ablation,
    standard_ablations,
)
from sgdiff.engine.batch import collate_subgraphs
from sgdiff.engine.factory import build_model, prepare_datasets


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory):
    return load_config(
        None,
        [
            "data.name=synthetic",
            "data.root=" + str(tmp_path_factory.mktemp("data")),
            "data.max_train_pairs=48",
            "data.max_eval_pairs=24",
            "subgraph.max_nodes_per_hop=6",
            "structure.num_steps=4",
            "feature.num_steps=4",
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
            "train.device=cpu",
        ],
    )


@pytest.fixture(scope="module")
def prepared(tiny_config):
    graph = make_synthetic(num_nodes=150, num_features=10, seed=0)
    split = make_link_split(graph, seed=0)
    datasets = prepare_datasets(graph, split, tiny_config)
    model = build_model(tiny_config, datasets["train"].feature_dim)
    batch = collate_subgraphs(
        [datasets["train"][i] for i in range(4)], feature_dim=datasets["train"].feature_dim
    )
    return {"datasets": datasets, "model": model, "batch": batch}


def test_ablation_validates_fields():
    with pytest.raises(ValueError):
        Ablation("bad", condition="nope")
    with pytest.raises(ValueError):
        Ablation("bad", labels="nope")
    with pytest.raises(ValueError):
        Ablation("bad", stream="nope")
    with pytest.raises(ValueError):
        Ablation("bad", timestep_frac=0.0)


def test_condition_variants(prepared):
    batch = prepared["batch"]
    model = prepared["model"]
    num_classes = model.fusion.num_classes

    true_y = Ablation("t").y_for(batch, cls=1, num_classes=num_classes)
    assert torch.all(true_y == 1)

    same_y = Ablation("s", condition="same").y_for(batch, cls=1, num_classes=num_classes)
    assert torch.all(same_y == 0)

    flip_y = Ablation("f", condition="flip").y_for(batch, cls=0, num_classes=num_classes)
    assert torch.all(flip_y == 1)

    ablation = Ablation("r", condition="random", seed=3)
    first = ablation.y_for(batch, cls=0, num_classes=num_classes)
    second = ablation.y_for(batch, cls=1, num_classes=num_classes)
    assert torch.equal(first, second)  # one draw shared by every class
    assert set(first.tolist()) <= {0, 1}
    assert torch.equal(first, Ablation("r2", condition="random", seed=3).y_for(batch, 0, num_classes))


def test_label_shuffle_keeps_multiset_and_padding(prepared):
    batch = prepared["batch"]
    shuffled = Ablation("l", labels="shuffle", seed=1).labels_for(batch)
    assert shuffled.shape == batch.labels.shape
    for i in range(batch.labels.shape[0]):
        valid = batch.mask[i]
        assert torch.equal(
            torch.sort(shuffled[i][valid]).values, torch.sort(batch.labels[i][valid]).values
        )
        assert torch.all(shuffled[i][~valid] == 0)


def test_ablated_scorer_shapes(prepared):
    batch = prepared["batch"]
    model = prepared["model"]
    scorer = AblatedScorer(model, Ablation("ref"))
    scorer.eval()
    logits = scorer(batch)["logits"]
    assert logits.shape == (4, 2)
    assert scorer.predict_logit(batch).shape == (4,)
    proba = scorer.predict_proba(batch)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_same_condition_removes_class_signal(prepared):
    """With one condition for both classes the two likelihoods are draws of the same thing."""
    batch = prepared["batch"]
    model = prepared["model"]
    scorer = AblatedScorer(model, Ablation("same", condition="same")).eval()
    feat, struct = scorer.class_scores(batch)
    assert torch.equal(feat[:, 0], feat[:, 1])
    assert torch.equal(struct[:, 0], struct[:, 1])
    logits = scorer(batch)["logits"]
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()
    assert (feat - struct).abs().max() > 0  # two streams exist


def test_single_timestep_matches_shape(prepared):
    batch = prepared["batch"]
    model = prepared["model"]
    for fraction in (0.05, 0.5, 1.0):
        scorer = AblatedScorer(model, Ablation(f"t{fraction}", timestep_frac=fraction)).eval()
        assert scorer.predict_logit(batch).shape == (4,)


def test_stream_ablation_zeroes_one_likelihood(prepared):
    batch = prepared["batch"]
    model = prepared["model"]
    feat, struct = AblatedScorer(model, Ablation("st", stream="structure")).eval().class_scores(batch)
    assert torch.all(feat == 0)
    assert not torch.all(struct == 0)
    feat, struct = AblatedScorer(model, Ablation("fe", stream="feature")).eval().class_scores(batch)
    assert torch.all(struct == 0)
    assert not torch.all(feat == 0)


def test_stream_contributions_split_the_logit_variance(prepared):
    from sgdiff.engine.ablation import stream_contributions

    contributions = stream_contributions(
        prepared["model"], prepared["datasets"]["test"], batch_size=8, device="cpu"
    )
    assert contributions["feature_stream"] is True
    assert contributions["share_feature"] + contributions["share_structure"] == pytest.approx(1.0)
    assert contributions["std_structure_term"] > 0.0
    # the weights are read straight from the fusion module
    assert contributions["eta1_feature"] == pytest.approx(
        float(prepared["model"].fusion.eta1.detach())
    )


def test_standard_suite_covers_the_documented_variants():
    names = {ablation.name for ablation in standard_ablations()}
    for name in ("reference", "same_condition", "label_shuffle", "single_t_mid", "structure_only"):
        assert name in names


def test_evaluate_ablation_returns_metrics(prepared):
    metrics = evaluate_ablation(
        prepared["model"], prepared["datasets"]["test"], Ablation("ref"), batch_size=8, device="cpu"
    )
    for key in ("auc", "ap", "hit@100", "score_std"):
        assert np.isfinite(metrics[key]) or np.isnan(metrics[key])


def test_run_ablation_end_to_end(tiny_config, tmp_path):
    result = run_ablation(
        tiny_config,
        seed=0,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        ablations=standard_ablations()[:3],
        train_without_condition=True,
        unfrozen_variant=False,
        write=False,
        verbose=False,
    )
    names = [row["variant"] for row in result["rows"]]
    assert "reference" in names and "trained_without_condition" in names
    assert len(result["rows"]) == 4
    assert "| variant |" in result["markdown"]
    # fusion weights + per-stream contribution are recorded, not just AUC
    assert result["fusion"], "fusion parameters must be reported"
    row = result["fusion"][0]
    for key in ("eta1_feature", "eta2_structure", "share_feature", "share_structure"):
        assert key in row
    assert 0.0 <= row["share_feature"] <= 1.0
    assert "eta_1 (feature)" in result["markdown"]


def test_ablated_scorer_preserves_parameter_gradient_flags(prepared):
    model = prepared["model"]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    scorer = AblatedScorer(model, Ablation("gradient_flags"))
    scorer.predict_logit(prepared["batch"])
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert any(flags)


@pytest.mark.parametrize("entry", ["forward", "predict_logit", "predict_proba", "class_scores"])
def test_ablated_scorer_forwards_monte_carlo_samples(prepared, monkeypatch, entry):
    model = prepared["model"]
    batch = prepared["batch"]
    calls = []

    def class_scores(*args, num_samples=None, conditions=None, **kwargs):
        calls.append(num_samples)
        assert conditions.shape == (batch.batch_size, 2)
        return torch.zeros(conditions.shape, device=batch.y.device)

    monkeypatch.setattr(model.structure, "class_scores", class_scores)
    monkeypatch.setattr(model.feature, "class_scores", class_scores)
    scorer = AblatedScorer(model, Ablation("samples"))
    getattr(scorer, entry)(batch, num_samples=7)
    assert calls == [7, 7]


def test_evaluate_ablation_preserves_rng_and_training_mode(prepared):
    model = prepared["model"].train()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    before = torch.get_rng_state()
    evaluate_ablation(model, prepared["datasets"]["test"], Ablation("state"), batch_size=8)
    assert torch.equal(before, torch.get_rng_state())
    assert model.training
    assert [parameter.requires_grad for parameter in model.parameters()] == flags


def test_reference_ablation_matches_model_branch_scores_exactly(prepared):
    model = prepared["model"].eval()
    batch = prepared["batch"]
    torch.manual_seed(91)
    expected = model.class_scores(batch, num_samples=2)
    torch.manual_seed(91)
    actual = AblatedScorer(model, Ablation("reference")).class_scores(batch, num_samples=2)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.parametrize("condition", ["same", "random"])
@pytest.mark.parametrize("fraction", [None, 0.5])
def test_equal_condition_classes_share_noise(prepared, condition, fraction):
    scorer = AblatedScorer(
        prepared["model"], Ablation("shared", condition=condition, timestep_frac=fraction, seed=7)
    ).eval()
    feature, structure = scorer.class_scores(prepared["batch"], num_samples=2)
    assert torch.equal(feature[:, 0], feature[:, 1])
    assert torch.equal(structure[:, 0], structure[:, 1])


def test_single_timestep_uses_score_norm_and_shared_feature_noise(prepared, monkeypatch):
    batch = prepared["batch"]
    model = prepared["model"]
    seen = []

    def score_mse(x0, adj, labels, mask, y, t=None, noise=None):
        seen.append((t.clone(), noise))
        return torch.full((batch.batch_size,), 11.0, device=batch.y.device)

    def training_mse(*args, **kwargs):
        raise AssertionError("training MSE reduction must not be used for scoring")

    monkeypatch.setattr(model.feature, "score_mse", score_mse)
    monkeypatch.setattr(model.feature, "compute_mse", training_mse)
    scorer = AblatedScorer(model, Ablation("single", timestep_frac=0.5, stream="feature")).eval()
    feature, structure = scorer.class_scores(batch)
    assert torch.all(feature == -11.0)
    assert torch.all(structure == 0)
    assert len(seen) == 2
    assert torch.all(seen[0][0] == 2)
    assert seen[0][1] is seen[1][1]
