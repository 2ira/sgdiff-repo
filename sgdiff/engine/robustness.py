"""Robustness (RQ3) harness: RF / DICE / EA against every model in the zoo.

Protocol (paper Sec. 4.4, *evasion* setting)
-------------------------------------------
1. train every model once on the clean graph, through
   :func:`~sgdiff.engine.benchmark.fit_model` — the same split, the same
   enclosing sub-graphs and the same training pairs as the benchmark;
2. perturb only the **inference** graph with a budget of ``ratio * |E|`` flips
   (:mod:`sgdiff.data.attack` gives RF, DICE and EA the same interface);
3. score the **same** test pair set with the frozen model.

That is what the paper does: the attacked graph is the graph the model is asked
about, the weights are untouched.  Node features are standardised once on the
clean evaluation graph, so for featureless datasets (USAir/NS) the degree
encoding does not silently absorb the perturbation — structure is the only
variable.
"""

from __future__ import annotations

import copy
import time
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.attack import EA_MAX_NODES, apply_attack, supports_attack
from ..data.splits import LinkSplit
from ..utils.io import save_json
from ..utils.seed import resolve_device, set_seed
from .baseline_trainer import (
    _eval_subgraph,
    _edge_index,
    evaluate_node_pair_baseline,
    node_feature_matrix,
)
from .batch import PairSubgraphDataset
from .benchmark import METRICS, build_registry, fit_model
from .evaluator import evaluate_link_prediction
from .factory import build_extractor, load_split, prepare_datasets

#: Attack names as accepted on the command line (``clean`` == no perturbation).
ATTACKS = ("clean", "rf", "dice", "ea")

CLEAN = "clean"


def variant_name(attack: str, ratio: float) -> str:
    """``clean`` or ``<attack>@<percent>`` (``rf@10%``)."""
    if attack in (None, CLEAN):
        return CLEAN
    return f"{attack}@{ratio * 100:g}%"


def _cap_pairs(split: LinkSplit, max_pairs: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """The scored test pairs: per-class cap, ``None`` = the full test split."""
    pos = np.asarray(split.test_pos, dtype=np.int64).reshape(-1, 2)
    neg = np.asarray(split.test_neg, dtype=np.int64).reshape(-1, 2)
    if max_pairs:
        cap = int(max_pairs)
        pos, neg = pos[:cap], neg[:cap]
    return pos, neg


def score_variant(
    kind: str,
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    eval_graph,
    pos: np.ndarray,
    neg: np.ndarray,
    device,
    seed: int,
    clean_features: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Score one (model, perturbed graph) pair with the shared metric code."""
    if kind == "node_pair":
        edge_index = _edge_index(eval_graph, device)
        x = torch.as_tensor(clean_features, device=device)
        metrics = evaluate_node_pair_baseline(
            model, x, edge_index, eval_graph.num_nodes, pos, neg, device
        )
        return {k: float(metrics[k]) for k in METRICS if k in metrics}

    extractor = build_extractor(eval_graph, cfg, seed=cfg.data.seed + 1)
    if kind == "subgraph":
        pairs = np.vstack([pos, neg]) if len(neg) else pos
        labels = np.concatenate(
            [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
        )
        metrics = _eval_subgraph(model, PairSubgraphDataset(pairs, labels, extractor), device)
        return {k: float(v) for k, v in metrics.items() if k in METRICS}

    metrics = evaluate_link_prediction(
        model,
        extractor,
        pos,
        neg,
        batch_size=cfg.train.eval_batch_size,
        device=device,
        seed=seed,
    )
    return {k: float(metrics[k]) for k in METRICS if k in metrics}


def _aggregate(runs: Sequence[dict], models: Sequence[str], variants: Sequence[str]) -> dict:
    summary: dict[str, dict[str, dict]] = {}
    for name in models:
        per_variant: dict[str, dict] = {}
        for variant in variants:
            subset = [r for r in runs if r["model"] == name and r["variant"] == variant]
            entry: dict[str, dict] = {}
            for metric in METRICS:
                values = [r["metrics"][metric] for r in subset if metric in r["metrics"]]
                values = [v for v in values if np.isfinite(v)]
                if not values:
                    continue
                entry[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
                drops = [r["drop"][metric] for r in subset if metric in r.get("drop", {})]
                drops = [d for d in drops if np.isfinite(d)]
                if variant != CLEAN and drops:
                    entry[metric]["drop_mean"] = float(np.mean(drops))
            if entry:
                per_variant[variant] = entry
        summary[name] = per_variant
    return summary


def format_markdown(result: dict) -> str:
    """One table: rows = models, columns = clean + every attack budget (AUC)."""
    summary = result["summary"]
    variants = [CLEAN] + [v for v in result["variants"] if v != CLEAN]
    header = ["model", "#params"] + [f"AUC {v}" for v in variants]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    ranked = sorted(
        summary.items(), key=lambda kv: -kv[1].get(CLEAN, {}).get("auc", {}).get("mean", -1.0)
    )
    for name, per_variant in ranked:
        params = next((r["num_params"] for r in result["runs"] if r["model"] == name), 0)
        cells = [name, f"{params:,}"]
        for variant in variants:
            entry = per_variant.get(variant, {}).get("auc")
            if entry is None:
                cells.append("-")
            elif entry["n"] > 1:
                cells.append(f"{entry['mean']:.3f} ± {entry['std']:.3f}")
            else:
                cells.append(f"{entry['mean']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_robustness(
    config: ExperimentConfig,
    models: Optional[Sequence[str]] = None,
    attacks: Sequence[str] = ("rf", "dice", "ea"),
    ratios: Sequence[float] = (0.1, 0.25, 0.5),
    seeds: Sequence[int] = (0,),
    sgdiff_epochs: Optional[int] = None,
    subgraph_epochs: int = 40,
    node_pair_epochs: int = 300,
    device: str = "auto",
    max_pairs: Optional[int] = None,
    output_dir: str = "results",
    run_root: str = "runs/robustness",
    ea_max_nodes: int = EA_MAX_NODES,
    write: bool = True,
    quiet: bool = False,
    verbose: bool = True,
) -> dict:
    """Train every model once per seed and re-score it on attacked inference graphs.

    ``attacks`` may contain ``"clean"`` explicitly; it is always evaluated (the
    reference column) and never duplicated.  EA is skipped with a note when the
    graph exceeds ``ea_max_nodes`` (its meta-gradient is dense).
    """
    if quiet:
        import logging

        logging.getLogger("sgdiff").setLevel(logging.WARNING)
    device = resolve_device(device)
    set_seed(int(seeds[0]))

    registry = build_registry(config)
    models = list(models) if models else ["sgdiff", "seal", "pencil", "gcn", "gat", "sage", "vgae"]
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise KeyError(f"unknown models {unknown}; available: {sorted(registry)}")
    attacks = [a for a in dict.fromkeys([CLEAN] + [str(a).lower() for a in attacks])]
    ratios = [float(r) for r in ratios]
    # (variant, attack, ratio): the clean column first, then every attack budget
    plan: list[tuple[str, str, float]] = [(CLEAN, CLEAN, 0.0)] + [
        (variant_name(a, r), a, r) for r in ratios for a in attacks if a != CLEAN
    ]
    variants = [variant for variant, _attack, _ratio in plan]

    runs: list[dict] = []
    attack_stats: dict[str, dict] = {}
    skipped: list[dict] = []
    num_pos = num_neg = 0
    total = len(seeds) * len(models)

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.data.seed = int(seed)
        cfg.train.seed = int(seed)
        set_seed(int(seed))

        graph, split = load_split(cfg)
        datasets = prepare_datasets(graph, split, cfg)
        pos, neg = _cap_pairs(split, max_pairs)
        num_pos, num_neg = int(len(pos)), int(len(neg))
        test_edges = split.test_pos.tolist()
        clean_eval_graph = graph.drop_edges(test_edges)
        clean_features = (
            node_feature_matrix(clean_eval_graph)
            if any(registry[m].kind == "node_pair" for m in models)
            else None
        )

        # Build every attacked inference graph once per seed, then reuse it for
        # all models: identical perturbation for identical comparison.
        perturbed: dict[str, object] = {CLEAN: clean_eval_graph}
        for variant, attack, ratio in plan:
            if attack == CLEAN:
                continue
            if not supports_attack(attack, graph.num_nodes, ea_max_nodes):
                skipped.append(
                    {
                        "seed": int(seed),
                        "variant": variant,
                        "reason": f"{attack} is dense: {graph.num_nodes} nodes > {ea_max_nodes}",
                    }
                )
                continue
            attacked = apply_attack(graph, attack, ratio=ratio, seed=int(seed), device=str(device))
            perturbed[variant] = attacked.drop_edges(test_edges)
            attack_stats[variant] = {"ratio": ratio, **attacked.meta.get("attack", {})}

        for index, name in enumerate(models, start=1):
            spec = registry[name]
            started = time.perf_counter()
            model, _train_metrics, num_params = fit_model(
                spec,
                cfg,
                graph,
                split,
                datasets,
                int(seed),
                sgdiff_epochs=sgdiff_epochs,
                subgraph_epochs=subgraph_epochs,
                node_pair_epochs=node_pair_epochs,
                device=device,
                run_root=f"{run_root}/{cfg.data.name}",
            )
            fit_seconds = time.perf_counter() - started
            clean_metrics: dict[str, float] = {}

            for variant, attack, ratio in plan:
                eval_graph = perturbed.get(variant)
                if eval_graph is None:  # skipped attack (EA on a big graph)
                    continue
                metrics = score_variant(
                    spec.kind,
                    model,
                    cfg,
                    eval_graph,
                    pos,
                    neg,
                    device,
                    int(seed),
                    clean_features=clean_features,
                )
                if variant == CLEAN:
                    clean_metrics = metrics
                runs.append(
                    {
                        "model": name,
                        "seed": int(seed),
                        "attack": attack,
                        "ratio": float(ratio),
                        "variant": variant,
                        "metrics": metrics,
                        "num_params": int(num_params),
                        "fit_seconds": float(fit_seconds),
                    }
                )
            # drop w.r.t. the clean column of the same seed (higher = worse)
            for run in runs:
                if run["model"] != name or run["seed"] != int(seed) or run["variant"] == CLEAN:
                    continue
                run["drop"] = {
                    k: clean_metrics[k] - v for k, v in run["metrics"].items() if k in clean_metrics
                }
            if verbose:
                drops = " ".join(
                    f"{r['variant']}={r['metrics']['auc']:.3f}"
                    for r in runs
                    if r["model"] == name and r["seed"] == int(seed)
                )
                print(
                    f"[{index}/{total}] {cfg.data.name} seed={seed} {name}: {drops} "
                    f"({fit_seconds:.1f}s fit)",
                    flush=True,
                )

    result = {
        "dataset": config.data.name,
        "models": models,
        "seeds": [int(s) for s in seeds],
        "attacks": [a for a in attacks if a != CLEAN],
        "ratios": ratios,
        "variants": variants,
        "protocol": {
            "setting": "evasion: weights trained on the clean graph, attack applied to the inference graph",
            "scored_pairs": "test split (per-class cap: %s)" % ("full" if not max_pairs else max_pairs),
            "num_pos": num_pos,
            "num_neg": num_neg,
            "subgraph": {
                "hops": config.subgraph.hops,
                "max_nodes_per_hop": config.subgraph.max_nodes_per_hop,
                "max_nodes_total": config.subgraph.max_nodes_total,
                "remove_target_edge": config.subgraph.remove_target_edge,
            },
            "metrics": list(METRICS),
            "epochs": {
                "sgdiff": sgdiff_epochs if sgdiff_epochs is not None else config.train.epochs,
                "subgraph": subgraph_epochs,
                "node_pair": node_pair_epochs,
            },
            "device": str(device),
        },
        "attack_stats": attack_stats,
        "skipped": skipped,
        "runs": runs,
        "summary": _aggregate(runs, models, variants),
    }
    result["markdown"] = format_markdown(result)

    if write:
        stem = f"{output_dir}/robustness_{config.data.name}"
        save_json(result, f"{stem}.json")
        with open(f"{stem}.md", "w", encoding="utf-8") as fh:
            fh.write(_header(result) + result["markdown"] + "\n")
    return result


def _header(result: dict) -> str:
    return "\n".join(
        [
            f"# Robustness: {result['dataset']}",
            "",
            "Generated by `scripts/run_robustness.py`. Every model is trained once on",
            "the clean graph and then re-scored on the *inference* graph perturbed by",
            "RF / DICE / EA at the given budgets (evasion setting). `drop_mean` in the",
            "JSON is the AUC lost against the model's own clean column.",
            "",
            f"* models: {', '.join(result['models'])}",
            f"* seeds: {result['seeds']}",
            f"* ratios: {', '.join(f'{r * 100:g}%' for r in result['ratios'])}",
            f"* metric: AUC (equal-size random negatives)",
            "",
            "",
        ]
    )
