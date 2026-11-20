"""Clean-score controls for interpreting the team study's diagnostic profile."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.benchmark import build_registry, fit_model
from sgdiff.engine.course_baselines import fit_course_model, score_course_model
from sgdiff.engine.evaluator import score_pairs
from sgdiff.engine.factory import load_split, prepare_datasets
from sgdiff.engine.robustness import _cap_pairs
from sgdiff.engine.team_study import settings
from sgdiff.engine.trainer import load_checkpoint
from sgdiff.utils.metrics import roc_auc
from sgdiff.utils.seed import set_seed


def sp4lp_control(cfg):
    graph, split = load_split(cfg)
    original = prepare_datasets(graph, split, cfg)
    pos, neg = _cap_pairs(split, 500)
    pairs = np.vstack((pos, neg))
    labels = np.repeat([1, 0], [len(pos), len(neg)])
    registry = build_registry(cfg)
    for train_size in (256, 2000):
        cfg.data.max_train_pairs = train_size
        datasets = prepare_datasets(graph, split, cfg)
        for part in ("val", "test"):
            np.testing.assert_array_equal(original[part].pairs, datasets[part].pairs)
        for epochs in (30, 300):
            for name in ("sp4lp", "sp4lp_endpoints", "sp4lp_featureless"):
                model, metrics, _ = fit_course_model(
                    registry[name], cfg, graph, datasets, cfg.data.seed, epochs, "cpu"
                )
                scores = score_course_model(model, datasets["test"].extractor.graph, pairs)
                yield dict(seed=cfg.data.seed, train=train_size, epoch_cap=epochs, model=name,
                           auc128=metrics["auc"], auc1000=roc_auc(labels, scores))


def sgdiff_control(cfg, checkpoint, train_size, run_root):
    if checkpoint is not None:
        model, stored = load_checkpoint(checkpoint)
        expected = cfg.to_dict()
        actual = stored.to_dict()
        for config in (expected, actual):
            config["data"].pop("root")
            for key in ("device", "run_name", "run_dir"):
                config["train"].pop(key)
        if expected != actual:
            raise ValueError("use a clean local-profile checkpoint for the requested dataset and seed")
    else:
        cfg.data.max_train_pairs = train_size
    graph, split = load_split(cfg)
    datasets = prepare_datasets(graph, split, cfg)
    if checkpoint is None:
        set_seed(cfg.data.seed)
        model, _, _ = fit_model(build_registry(cfg)["sgdiff"], cfg, graph, split, datasets,
                                cfg.data.seed, device="cpu",
                                run_root=str(Path(run_root) / cfg.data.name / f"train_{train_size}"))
    d = datasets["test"]
    published = Path("results/team-study/local") / cfg.data.name / f"seed_{cfg.data.seed}"
    with np.load(published / "queries_and_scores.npz") as reference:
        np.testing.assert_array_equal(reference["test_pairs"], d.pairs)
        np.testing.assert_array_equal(reference["test_labels"], d.labels)
        for samples in (2, 4, "all"):
            for stream in (model.structure, model.feature):
                stream.num_mc_samples = stream.num_steps if samples == "all" else samples
            scores = score_pairs(model, d.extractor, d.pairs, cfg.train.eval_batch_size,
                                 device="cpu", seed=cfg.data.seed)
            if checkpoint and samples == 2:
                np.testing.assert_allclose(scores, reference["clean__sgdiff"], rtol=0, atol=1e-5)
            yield dict(dataset=cfg.data.name, seed=cfg.data.seed, n_train=cfg.data.max_train_pairs,
                       mc=samples, auc=roc_auc(d.labels, scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["sgdiff", "sp4lp"])
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-pairs", type=int, choices=[256, 2000], default=256)
    parser.add_argument("--run-root", default="runs/budget-audit-replay")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint and (args.model != "sgdiff" or args.train_pairs != 256):
        parser.error("checkpoint replay applies to the 256-query SGDiff diagnostic")
    torch.set_num_threads(2 if args.model == "sp4lp" else 3)
    cfg = settings(args.dataset, args.seed, "local", args.data_root, "cpu")
    data = (sp4lp_control(cfg) if args.model == "sp4lp"
            else sgdiff_control(cfg, args.checkpoint, args.train_pairs, args.run_root))
    values = list(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(values, indent=2) + "\n")
