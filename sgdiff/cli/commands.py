"""Implementations of the CLI sub-commands."""

from __future__ import annotations

from pathlib import Path
from ..config import ExperimentConfig, load_config, save_config
from ..data.attack import dice_attack, random_perturbation
from ..data.datasets import DATASETS, available_datasets, describe_dataset, load_graph
from ..data.splits import make_link_split
from ..engine.evaluator import evaluate_link_prediction
from ..engine.factory import build_extractor, build_model, prepare_datasets
from ..engine.trainer import Trainer, load_checkpoint
from ..utils.io import save_json
from ..utils.logging import get_logger
from ..utils.seed import resolve_device, set_seed

LOGGER = get_logger(__name__)


def _load(cfg: ExperimentConfig):
    graph = load_graph(cfg.data.name, root=cfg.data.root, download=cfg.data.download)
    split = make_link_split(
        graph,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        neg_ratio=cfg.data.neg_ratio,
        seed=cfg.data.seed,
    )
    return graph, split


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def cmd_prepare(args) -> dict:
    names = args.datasets or available_datasets()
    report = {}
    for name in names:
        if name not in DATASETS:
            LOGGER.warning("skip unknown dataset %s", name)
            continue
        spec = DATASETS[name]
        try:
            graph = load_graph(name, root=args.root, download=not args.no_download)
            report[name] = {
                "status": "ok",
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "num_features": graph.num_features,
                "paper": describe_dataset(name),
            }
        except Exception as exc:  # noqa: BLE001 - report per-dataset failures
            report[name] = {"status": "failed", "error": str(exc), "note": spec.note}
            LOGGER.error("%s: %s", name, exc)
    save_json(report, Path(args.root) / "dataset_report.json")
    return report


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args) -> dict:
    cfg = load_config(args.config, args.set)
    set_seed(cfg.train.seed)
    graph, split = _load(cfg)
    datasets = prepare_datasets(graph, split, cfg)
    model = build_model(cfg, datasets["train"].feature_dim)
    if args.run_dir:
        cfg.train.run_dir = args.run_dir
    trainer = Trainer(model, cfg)
    save_config(cfg, trainer.run_dir / "config.yaml")
    result = trainer.fit(
        datasets["train"], datasets["val"], datasets["test"], max_steps_per_epoch=args.max_steps
    )
    result["split"] = split.summary()
    save_json(result, trainer.run_dir / "summary.json")
    return result


# --------------------------------------------------------------------------- #
# evaluate / transfer
# --------------------------------------------------------------------------- #
def cmd_evaluate(args) -> dict:
    prepared_device = resolve_device(args.device or "auto")
    model, cfg = load_checkpoint(args.checkpoint, device=prepared_device)
    if args.device is None:
        prepared_device = resolve_device(cfg.train.device)
        model = model.to(prepared_device)
    if args.dataset:
        cfg.data.name = args.dataset
    graph, split = _load(cfg)
    split_name = args.split
    pos = split.positives(split_name)
    neg = split.negatives(split_name)
    if args.max_pairs:
        half = args.max_pairs // 2
        pos, neg = pos[:half], neg[:half]

    eval_graph = graph.drop_edges(split.test_pos.tolist())
    attack = None
    if args.attack in {"random", "dice"}:
        attack = (
            random_perturbation(graph, ratio=args.ratio, mode=args.attack_mode, seed=cfg.data.seed)
            if args.attack == "random"
            else dice_attack(graph, ratio=args.ratio, seed=cfg.data.seed)
        )
        eval_graph = attack

    extractor = build_extractor(eval_graph, cfg)
    metrics = evaluate_link_prediction(
        model,
        extractor,
        pos,
        neg,
        batch_size=cfg.train.eval_batch_size,
        device=prepared_device,
        seed=cfg.train.seed,
    )
    metrics.update({"dataset": cfg.data.name, "split": split_name})
    if attack is not None:
        metrics["attack"] = {"method": args.attack, "ratio": args.ratio}
    if args.output:
        save_json(metrics, args.output)
    return metrics


def cmd_transfer(args) -> dict:
    prepared_device = resolve_device(args.device or "auto")
    model, source_cfg = load_checkpoint(args.checkpoint, device=prepared_device)
    results = {}
    targets = args.targets or [n for n in available_datasets() if n != source_cfg.data.name]
    for target in targets:
        cfg = ExperimentConfig.from_dict(source_cfg.to_dict())
        cfg.data.name = target
        graph = load_graph(target, root=args.root, download=True)
        split = make_link_split(
            graph,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            neg_ratio=cfg.data.neg_ratio,
            seed=cfg.data.seed,
        )
        eval_graph = graph.drop_edges(split.test_pos.tolist())
        extractor = build_extractor(
            eval_graph, cfg, seed=cfg.data.seed + 1,
            target_feature_dim=int(model.feature.feature_dim),
        )
        pos, neg = split.test_pos, split.test_neg
        if args.max_pairs:
            half = args.max_pairs // 2
            pos, neg = pos[:half], neg[:half]
        metrics = evaluate_link_prediction(
            model, extractor, pos, neg, batch_size=cfg.train.eval_batch_size,
            device=prepared_device, seed=cfg.train.seed,
        )
        results[target] = metrics
        LOGGER.info("transfer %s -> %s: %s", source_cfg.data.name, target, metrics)
    if args.output:
        save_json(results, args.output)
    return results
