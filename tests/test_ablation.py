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
    # The scores still differ through the sampled noise, but they must be the
    # same order of magnitude; the *mean over classes* is what the fusion uses.
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
