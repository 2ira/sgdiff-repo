"""Phase isolation, shared scoring and epoch-boundary checkpoint recovery."""

import pytest
import torch

from sgdiff.config import load_config
from sgdiff.data.datasets import make_synthetic
from sgdiff.data.splits import make_link_split
from sgdiff.engine.factory import build_model, prepare_datasets
from sgdiff.engine.trainer import Trainer, load_checkpoint
from sgdiff.utils.seed import set_seed
from sgdiff.version import METHOD_VERSION


def prepare(tmp_path, name):
    cfg = load_config(None, [
        "data.name=synthetic", "data.max_train_pairs=24", "data.max_eval_pairs=8",
        "subgraph.max_nodes_per_hop=4", "structure.arch=edge_gnn", "structure.hidden_dim=8",
        "structure.num_layers=1", "structure.num_steps=3", "structure.num_mc_samples=2",
        "feature.hidden_dim=8", "feature.num_steps=3", "feature.num_mc_samples=2",
        "train.epochs=2", "train.pretrain_epochs=1", "train.finetune_epochs=2",
        "train.finetune_mc_samples=2", "train.batch_size=8", "train.eval_batch_size=8",
        "train.patience=0", "train.fusion_patience=0", "train.device=cpu",
        f"train.run_dir={tmp_path}", f"train.run_name={name}",
    ])
    set_seed(11)
    graph = make_synthetic(num_nodes=40, num_features=6, seed=11)
    data = prepare_datasets(graph, make_link_split(graph, seed=11), cfg)
    model = build_model(cfg, data["train"].feature_dim)
    return cfg, data, model


def test_frozen_fusion_never_updates_denoisers(tmp_path):
    cfg, data, model = prepare(tmp_path, "freeze")
    trainer = Trainer(model, cfg)
    main, fusion = trainer._optimizers()
    before = {k: v.clone() for k, v in model.structure.state_dict().items()}
    features = {k: v.clone() for k, v in model.feature.state_dict().items()}
    trainer._train_epoch(trainer._loader(data["train"], 8, False, 0), main, fusion, "finetune")
    assert all(torch.equal(v, model.structure.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, model.feature.state_dict()[k]) for k, v in features.items())
    assert not model.feature.training and not model.structure.training
    assert not main.state
    assert fusion.state


def test_train_logits_and_inference_share_the_scoring_objective(tmp_path):
    cfg, data, model = prepare(tmp_path, "scoring")
    model.eval()
    batch = next(iter(Trainer._loader(data["train"], 8, False, 0)))
    torch.manual_seed(4)
    expected = model(batch, num_samples=2)
    torch.manual_seed(4)
    logits, scores = model.train_logits(batch, num_samples=2)
    assert torch.equal(logits, expected["logits"])
    assert torch.equal(scores["feature_scores"], expected["feature_scores"])
    assert torch.equal(scores["structure_scores"], expected["structure_scores"])


@pytest.mark.parametrize("interrupted_stage", ["pretrain", "finetune"])
def test_resume_matches_uninterrupted_training(tmp_path, interrupted_stage):
    cfg, data, model = prepare(tmp_path, "reference")
    reference = Trainer(model, cfg)
    reference.fit(data["train"], data["val"], data["test"])
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    cfg, data, model = prepare(tmp_path, "interrupted")
    interrupted = Trainer(model, cfg)
    original = interrupted.save_checkpoint

    def fail_after_checkpoint(path, epoch=-1, metrics=None, training_state=None):
        result = original(path, epoch, metrics, training_state)
        if training_state and training_state["stage"] == interrupted_stage and epoch == 0:
            raise RuntimeError("simulated interruption")
        return result

    interrupted.save_checkpoint = fail_after_checkpoint
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.fit(data["train"], data["val"], data["test"])
    cfg.train.resume = str(interrupted.last_path)
    restored = build_model(cfg, data["train"].feature_dim)
    resumed = Trainer(restored, cfg)
    resumed.fit(data["train"], data["val"], data["test"])
    assert all(torch.equal(expected[k], restored.state_dict()[k]) for k in expected)
    loaded, _ = load_checkpoint(resumed.best_path)
    assert all(torch.equal(expected[k], loaded.state_dict()[k]) for k in expected)
    payload = torch.load(resumed.last_path, weights_only=True)
    assert payload["method_version"] == METHOD_VERSION
    assert payload["training_state"]["stage"] == "finetune"


def test_legacy_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"config": {}, "model_state": {}}, path)
    with pytest.raises(ValueError, match="legacy SGDiff"):
        load_checkpoint(path)
