#!/usr/bin/env python
"""Run the paper's experiment grid and aggregate metrics over seeds.

Examples
--------
    python scripts/run_experiments.py --datasets cora citeseer usair ns --seeds 0 1 2 --profile cpu
    python scripts/run_experiments.py --datasets cora --seeds 0 1 2 --profile paper
    python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
    python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
    python scripts/run_experiments.py --attack cora --ratios 0.25 0.5

Every run writes ``results/{tag}.json``; aggregates go to ``results/summary_*.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff import api  # noqa: E402
from sgdiff.data.attack import dice_attack, random_perturbation  # noqa: E402
from sgdiff.data.splits import make_link_split  # noqa: E402
from sgdiff.engine.evaluator import evaluate_link_prediction  # noqa: E402
from sgdiff.engine.factory import build_extractor  # noqa: E402
from sgdiff.engine.trainer import load_checkpoint  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

RESULTS = Path("results")

# `paper` = Table 4 / Sec. 4.1 exactly; `cpu` keeps the protocol and
# architecture but shrinks compute so a laptop can finish a run.
PROFILES = {
    "paper": {},
    "cpu": {
        "data.max_train_pairs": 2000,
        "data.max_eval_pairs": 500,
        "train.epochs": 8,
        "train.pretrain_epochs": 3,
        "structure.num_mc_samples": 4,
        "feature.num_mc_samples": 4,
        "subgraph.max_nodes_total": 40,
    },
}


def _aggregate(runs: list[dict]) -> dict:
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {"num_seeds": len(runs)}
    for key in keys:
        values = np.asarray([r[key] for r in runs], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.round(6).tolist(),
        }
    return out


def _train(dataset: str, seed: int, profile: str, extra: list[str] | None = None):
    tag = f"{dataset}_seed{seed}_{profile}"
    overrides = [f"data.seed={seed}", f"train.seed={seed}", f"train.run_name={tag}"]
    overrides += [f"{k}={v}" for k, v in PROFILES[profile].items()]
    overrides += list(extra or [])
    _, result, cfg = api.train(config_path=f"configs/sgdiff_{dataset}.yaml", overrides=overrides)
    return result, cfg, tag


def run_indomain(datasets, seeds, profile) -> dict:
    summary = {}
    for dataset in datasets:
        runs = []
        for seed in seeds:
            started = time.time()
            result, _cfg, tag = _train(dataset, seed, profile)
            metrics = dict(result["test"])
            metrics.update({"dataset": dataset, "seed": seed, "profile": profile,
                            "seconds": round(time.time() - started, 1)})
            save_json(metrics, RESULTS / f"{tag}.json")
            runs.append(metrics)
            print(f"[{dataset} seed={seed}] auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
                  f"hit@100={metrics['hit@100']:.4f} ({metrics['seconds']}s)", flush=True)
        summary[dataset] = _aggregate(runs)
    return summary


def run_transfer(source: str, targets: list[str], profile: str, seed: int, max_pairs: int = 500) -> dict:
    """RQ1: train once (structure-only), evaluate on every target without retraining."""
    result, _cfg, tag = _train(source, seed, profile, ["feature.enabled=false"])
    model, cfg = load_checkpoint(f"{result['run_dir']}/best.pt", device=resolve_device("auto"))
    out = {}
    for target in targets:
        if target == source:
            continue
        graph = api.load_graph(target)
        split = make_link_split(graph, seed=cfg.data.seed)
        extractor = build_extractor(graph.drop_edges(split.test_pos.tolist()), cfg, seed=cfg.data.seed + 1)
        metrics = evaluate_link_prediction(
            model, extractor, split.test_pos[:max_pairs], split.test_neg[:max_pairs],
            batch_size=cfg.train.eval_batch_size, seed=cfg.train.seed,
        )
        metrics["source"] = source
        out[target] = metrics
        print(f"[transfer {source}->{target}] auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}", flush=True)
    save_json(out, RESULTS / f"transfer_{tag}.json")
    return out


def run_lowdata(dataset: str, seeds, profile, ratio: float = 0.01) -> dict:
    """RQ2: only ``ratio`` of the training sub-graphs are used."""
    runs = []
    for seed in seeds:
        # With 1% of the pairs (a few dozen sub-graphs) the 8-epoch CPU budget
        # is far too short: every sample is seen ~2 times.  Give the small-data
        # setting its own, longer schedule.
        extra = [f"data.train_ratio={ratio}"]
        if ratio <= 0.02:
            extra += ["train.epochs=40", "train.pretrain_epochs=10", "train.patience=40"]
        result, _cfg, tag = _train(dataset, seed, profile, extra)
        metrics = dict(result["test"])
        metrics.update({"dataset": dataset, "seed": seed, "ratio": ratio})
        save_json(metrics, RESULTS / f"lowdata_{tag}.json")
        runs.append(metrics)
        print(f"[lowdata {dataset} seed={seed}] auc={metrics['auc']:.4f}", flush=True)
    return {dataset: _aggregate(runs)}


def run_attack(dataset: str, ratios: list[float], seed: int, profile: str, max_pairs: int = 500) -> dict:
    """RQ3: random flip (RF) and DICE perturbations of the inference graph."""
    result, _cfg, tag = _train(dataset, seed, profile)
    model, cfg = load_checkpoint(f"{result['run_dir']}/best.pt", device=resolve_device("auto"))
    graph = api.load_graph(dataset)
    split = make_link_split(graph, seed=cfg.data.seed)
    pos, neg = split.test_pos[:max_pairs], split.test_neg[:max_pairs]
    out = {}
    for ratio in ratios:
        variants = {
            "clean": graph,
            "rf": random_perturbation(graph, ratio=ratio, mode="flip", seed=seed),
            "dice": dice_attack(graph, ratio=ratio, seed=seed),
        }
        for name, attacked in variants.items():
            extractor = build_extractor(attacked.drop_edges(split.test_pos.tolist()), cfg,
                                        seed=cfg.data.seed + 1)
            metrics = evaluate_link_prediction(model, extractor, pos, neg,
                                               batch_size=cfg.train.eval_batch_size, seed=seed)
            out[f"{name}@{ratio}"] = metrics
            print(f"[attack {dataset} {name} {ratio}] auc={metrics['auc']:.4f}", flush=True)
    save_json(out, RESULTS / f"attack_{tag}.json")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["cora"])
    parser.add_argument("--targets", nargs="*", default=["citeseer", "pubmed", "usair", "ns"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--profile", default="cpu", choices=sorted(PROFILES))
    parser.add_argument("--transfer", default=None, help="source dataset for RQ1")
    parser.add_argument("--lowdata", default=None, help="dataset for RQ2")
    parser.add_argument("--attack", default=None, help="dataset for RQ3")
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.25, 0.5])
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    set_seed(args.seeds[0])

    if args.transfer:
        run_transfer(args.transfer, args.targets, args.profile, args.seeds[0], args.max_pairs)
        return 0
    if args.lowdata:
        save_json(run_lowdata(args.lowdata, args.seeds, args.profile),
                  RESULTS / f"summary_lowdata_{args.lowdata}.json")
        return 0
    if args.attack:
        save_json(run_attack(args.attack, args.ratios, args.seeds[0], args.profile, args.max_pairs),
                  RESULTS / f"summary_attack_{args.attack}.json")
        return 0

    summary = run_indomain(args.datasets, args.seeds, args.profile)
    save_json(summary, RESULTS / f"summary_indomain_{args.profile}.json")
    print(json.dumps({d: {m: round(v["mean"], 4) for m, v in mats.items() if isinstance(v, dict)}
                      for d, mats in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
