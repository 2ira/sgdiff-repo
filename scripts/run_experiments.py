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
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff import api  # noqa: E402
from sgdiff.config import ExperimentConfig, load_config  # noqa: E402
from sgdiff.data.attack import apply_attack  # noqa: E402
from sgdiff.data.splits import make_link_split  # noqa: E402
from sgdiff.engine.evaluator import evaluate_link_prediction  # noqa: E402
from sgdiff.engine.factory import build_extractor  # noqa: E402
from sgdiff.engine.trainer import load_checkpoint  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402
from sgdiff.version import METHOD_VERSION  # noqa: E402

RESULTS = Path("results") / METHOD_VERSION

# Table 4 specifies architecture and diffusion steps, not optimizer epochs.
PROFILES = {
    "paper": {
        "data.max_train_pairs": None,
        "data.max_eval_pairs": None,
        "subgraph.max_nodes_total": None,
        "structure.num_mc_samples": None,
        "feature.num_mc_samples": None,
        "train.finetune_mc_samples": None,
        "train.freeze_diffusion": True,
    },
    # `full` = Table-4 data/hyper-parameters, longer schedule than `cpu`;
    # still CPU-feasible (8 MC steps, 60-node cap, evaluation every 4 epochs).
    "full": {
        "data.max_train_pairs": None,
        "data.max_eval_pairs": 500,
        "train.epochs": 20,
        "train.pretrain_epochs": 4,
        "train.finetune_epochs": 10,
        "train.finetune_mc_samples": 8,
        "train.eval_every": 4,
        "structure.num_mc_samples": 8,
        "feature.num_mc_samples": 8,
        "subgraph.max_nodes_total": 60,
    },
    "cpu": {
        "data.max_train_pairs": 2000,
        "data.max_eval_pairs": 500,
        "train.epochs": 8,
        "train.pretrain_epochs": 3,
        "train.finetune_epochs": 4,
        "train.finetune_mc_samples": 4,
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
    overrides = [f"data.seed={seed}", f"train.seed={seed}", f"train.run_name={tag}",
                 f"train.run_dir=runs/{METHOD_VERSION}"]
    overrides += [f"{k}={v}" for k, v in PROFILES[profile].items()]
    overrides += list(extra or [])
    _, result, cfg = api.train(config_path=f"configs/sgdiff_{dataset}.yaml", overrides=overrides)
    return result, cfg, tag


def run_indomain(datasets, seeds, profile, extra: list[str] | None = None) -> dict:
    summary = {}
    for dataset in datasets:
        runs = []
        for seed in seeds:
            started = time.time()
            result, _cfg, tag = _train(dataset, seed, profile, extra)
            metrics = dict(result["test"])
            metrics.update({"dataset": dataset, "seed": seed, "profile": profile,
                            "method_version": METHOD_VERSION, "protocol": result["protocol"],
                            "seconds": round(time.time() - started, 1)})
            save_json(metrics, RESULTS / f"{tag}.json")
            runs.append(metrics)
            print(f"[{dataset} seed={seed}] auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
                  f"hit@100={metrics['hit@100']:.4f} ({metrics['seconds']}s)", flush=True)
        summary[dataset] = {"method_version": METHOD_VERSION, **_aggregate(runs)}
    return summary


def _eval_transfer(model, cfg, target: str, max_pairs: Optional[int] = None, extract_cfg=None) -> dict:
    """Score ``model`` on ``target`` using the target's own test split.

    ``extract_cfg`` overrides the source configuration **only** for sub-graph
    extraction (the Table 4 ``ns`` question); ``max_label`` stays the source's so
    that the DRNL embedding size still matches the checkpoint.
    """
    graph = api.load_graph(target)
    split = make_link_split(graph, seed=cfg.data.seed)
    extractor = build_extractor(
        graph.drop_edges(split.test_pos.tolist()), extract_cfg or cfg, seed=cfg.data.seed + 1
    )
    pos, neg = split.test_pos, split.test_neg
    if max_pairs:  # per-class cap; None = the full test split
        pos, neg = pos[:max_pairs], neg[:max_pairs]
    metrics = evaluate_link_prediction(
        model, extractor, pos, neg,
        batch_size=cfg.train.eval_batch_size, seed=cfg.train.seed,
    )
    metrics["source"] = cfg.data.name
    return metrics


def experiment_copy(cfg: ExperimentConfig) -> ExperimentConfig:
    """Deep copy of a config (dataclasses -> dict -> dataclasses)."""
    return ExperimentConfig.from_dict(cfg.to_dict())


def _table4_ns(target: str):
    """Table 4 per-hop sampling limit ``ns`` of ``target`` (``None`` == -1)."""
    path = Path("configs") / f"sgdiff_{target}.yaml"
    if not path.exists():
        return None
    return load_config(path, []).subgraph.max_nodes_per_hop


def _transfer_train(source: str, seed: int, profile: str):
    """Train the source once, structure-only, in its own run directory.

    The run name is *not* the in-domain tag: transfer models must not overwrite
    ``runs/<dataset>_seed<k>_<profile>`` (different stage-2 schedule, no features).
    """
    return _train(
        source, seed, profile,
        ["feature.enabled=false", f"train.run_name=transfer_src_{source}_seed{seed}_{profile}"],
    )


def run_transfer(source: str, targets: list[str], profile: str, seed: int,
                 max_pairs: Optional[int] = None) -> dict:
    """RQ1: train once (structure-only), evaluate on every target without retraining."""
    result, _cfg, tag = _transfer_train(source, seed, profile)
    model, cfg = load_checkpoint(f"{result['run_dir']}/best.pt", device=resolve_device("auto"))
    out = {}
    for target in targets:
        if target == source:
            continue
        metrics = _eval_transfer(model, cfg, target, max_pairs)
        out[target] = metrics
        print(f"[transfer {source}->{target}] auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
              f"n={metrics.get('num_pos')}+{metrics.get('num_neg')}", flush=True)
    save_json(out, RESULTS / f"transfer_{tag}.json")
    return out


VARIANTS = ("source_ns", "target_ns")


def run_transfer_matrix(sources, targets, profile: str, seeds, max_pairs: Optional[int] = None) -> dict:
    """RQ1 the way the paper reports it.

    * every source is trained once per seed and evaluated on every target;
    * two extraction variants are reported: ``source_ns`` (the source's Table 4
      ``ns`` applies everywhere -- true zero-shot) and ``target_ns`` (the target's
      own ``ns``, i.e. assuming the target's sub-graph statistics are known);
    * the diagonal (source == target) is in-domain, so it is taken from the
      in-domain grid when available and **excluded** from both averages.
    """
    seeds = list(seeds)
    raw: dict[str, dict[str, dict[str, list]]] = {
        variant: {src: {} for src in sources} for variant in VARIANTS
    }
    counts: dict[str, dict] = {}

    for source in sources:
        for seed in seeds:
            result, _cfg, _tag = _transfer_train(source, seed, profile)
            model, cfg = load_checkpoint(f"{result['run_dir']}/best.pt", device=resolve_device("auto"))
            for target in targets:
                if target == source:
                    continue
                target_cfg = experiment_copy(cfg)
                target_cfg.subgraph.max_nodes_per_hop = _table4_ns(target)
                for variant, ext_cfg in (("source_ns", cfg), ("target_ns", target_cfg)):
                    metrics = _eval_transfer(model, cfg, target, max_pairs, extract_cfg=ext_cfg)
                    raw[variant][source].setdefault(target, []).append(float(metrics["auc"]))
                    counts[target] = {"num_pos": metrics.get("num_pos"), "num_neg": metrics.get("num_neg")}
                    print(f"[{variant} {source}(s{seed})->{target}] auc={metrics['auc']:.4f} "
                          f"n={metrics.get('num_pos')}+{metrics.get('num_neg')}", flush=True)
            save_json(raw, RESULTS / f"transfer_matrix_partial_{profile}.json")  # resumable

    summary: dict = {
        "profile": profile,
        "seeds": seeds,
        "sources": list(sources),
        "targets": [t for t in targets],
        "max_pairs_per_class": max_pairs,
        "test_set_size": counts,
        "variants": {},
        "diagonal_in_domain": {},
        "note": ("Router is not redistributable, so the matrix covers 5 of the paper's 6 "
                 "sources/targets; the diagonal is in-domain and excluded from the averages."),
    }
    for variant in VARIANTS:
        cells = {
            src: {
                tgt: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": [round(v, 6) for v in vals],
                }
                for tgt, vals in raw[variant][src].items()
            }
            for src in sources
        }
        source_avg = {
            src: float(np.mean([c["mean"] for t, c in row.items() if t != src]))
            for src, row in cells.items()
            if any(t != src for t in row)
        }
        target_avg = {
            tgt: float(np.mean([row[tgt]["mean"] for src, row in cells.items() if tgt in row and src != tgt]))
            for tgt in targets
            if any(tgt in row and src != tgt for src, row in cells.items())
        }
        summary["variants"][variant] = {
            "auc": cells,
            "source_avg_excl_diagonal": source_avg,
            "target_avg_excl_diagonal": target_avg,
        }

    for name in sources:
        vals = []
        for seed in seeds:
            path = RESULTS / f"{name}_seed{seed}_{profile}.json"
            if path.exists():
                vals.append(float(json.loads(path.read_text())["auc"]))
        if vals:
            summary["diagonal_in_domain"][name] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals,
            }

    save_json(summary, RESULTS / f"transfer_matrix_{profile}.json")
    for variant in VARIANTS:
        print(f"\n### variant: {variant}")
        print("| source | " + " | ".join(targets) + " | source avg |")
        print("|---|" + "---:|" * (len(targets) + 1))
        for src in sources:
            row = summary["variants"][variant]["auc"][src]
            cells_s = [("—" if t == src else f"{row[t]['mean']:.3f}±{row[t]['std']:.3f}") for t in targets]
            avg = summary["variants"][variant]["source_avg_excl_diagonal"][src]
            print(f"| {src} | " + " | ".join(cells_s) + f" | {avg:.3f} |")
        tavg = summary["variants"][variant]["target_avg_excl_diagonal"]
        print("| **target avg** | " + " | ".join(f"{tavg[t]:.3f}" for t in targets if t in tavg) + " | |")
    return summary


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


def run_attack(
    dataset: str,
    ratios: list[float],
    seed: int,
    profile: str,
    max_pairs: int = 500,
    attacks: list[str] | None = None,
) -> dict:
    """RQ3: RF / DICE / EA perturbations of the inference graph (SGDiff, evasion)."""
    result, _cfg, tag = _train(dataset, seed, profile)
    model, cfg = load_checkpoint(f"{result['run_dir']}/best.pt", device=resolve_device("auto"))
    graph = api.load_graph(dataset)
    split = make_link_split(graph, seed=cfg.data.seed)
    pos, neg = split.test_pos[:max_pairs], split.test_neg[:max_pairs]
    methods = ["rf", "dice", "ea"] if attacks is None else list(attacks)
    out = {}
    for ratio in ratios:
        variants: list[tuple[str, object]] = [("clean", graph)]
        for method in methods:
            try:
                variants.append(
                    (method, apply_attack(graph, method, ratio=ratio, mode="flip", seed=seed))
                )
            except ValueError as exc:  # dense EA on a large graph
                out[f"{method}@{ratio}"] = {"skipped": str(exc)}
                print(f"[attack {dataset} {method} {ratio}] skipped: {exc}", flush=True)
        for name, attacked in variants:
            extractor = build_extractor(attacked.drop_edges(split.test_pos.tolist()), cfg,
                                        seed=cfg.data.seed + 1)
            metrics = evaluate_link_prediction(model, extractor, pos, neg,
                                               batch_size=cfg.train.eval_batch_size, seed=seed)
            metrics["perturbation"] = attacked.meta.get("attack")
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
    parser.add_argument("--transfer-matrix", action="store_true",
                        help="RQ1 matrix: every --datasets entry as source, every --targets entry as target")
    parser.add_argument("--lowdata", default=None, help="dataset for RQ2")
    parser.add_argument("--attack", default=None, help="dataset for RQ3")
    parser.add_argument("--attack-methods", nargs="*", default=["rf", "dice", "ea"],
                        choices=["rf", "dice", "ea"], help="RQ3 attacks to run")
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.25, 0.5])
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="cap per class (positives and negatives separately); "
                             "default = the full test split")
    parser.add_argument("--set", dest="sets", nargs="*", default=[], metavar="KEY=VALUE",
                        help="extra config overrides for the in-domain grid")
    args = parser.parse_args()

    if not args.seeds:
        parser.error("at least one seed is required")
    if args.sets and (args.transfer or args.transfer_matrix or args.lowdata or args.attack):
        parser.error("--set currently applies only to the in-domain grid; use the training CLI for other overrides")
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seeds[0])

    if args.transfer_matrix:
        run_transfer_matrix(args.datasets, args.targets, args.profile, args.seeds, args.max_pairs)
        return 0
    if args.transfer:
        run_transfer(args.transfer, args.targets, args.profile, args.seeds[0], args.max_pairs)
        return 0
    if args.lowdata:
        save_json(run_lowdata(args.lowdata, args.seeds, args.profile),
                  RESULTS / f"summary_lowdata_{args.lowdata}.json")
        return 0
    if args.attack:
        save_json(run_attack(args.attack, args.ratios, args.seeds[0], args.profile,
                             args.max_pairs or 500, args.attack_methods),
                  RESULTS / f"summary_attack_{args.attack}.json")
        return 0

    summary = run_indomain(args.datasets, args.seeds, args.profile, args.sets)
    save_json(summary, RESULTS / f"summary_indomain_{args.profile}.json")
    print(json.dumps({d: {m: round(v["mean"], 4) for m, v in mats.items() if isinstance(v, dict)}
                      for d, mats in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
