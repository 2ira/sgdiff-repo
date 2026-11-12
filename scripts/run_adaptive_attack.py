#!/usr/bin/env python
"""Is the measured damage an upper or a lower bound? Adaptive edge deletion.

Every probe in this repo is *non-adaptive*: it deletes a random or degree-selected
half of the graph and never looks at the model.  The attribution claims therefore
have an unstated caveat -- they describe what *these* perturbations do, not what
the cheapest perturbation against the model would do.  This script closes that gap
with a greedy, score-feedback attacker at a *matched budget*:

* ``random``   -- delete random edges (replicates the probe's mechanism, not its ratio),
* ``hub``      -- delete the edges with the highest endpoint degree,
* ``adaptive`` -- at each step, sample candidate edges from the **evidence pool**
  (the edges that are the unique shared neighbour of some test positive), score
  the working set with the candidate removed, and delete the one that costs the
  model the most separation.

The working set is a fixed 200 + 200 test pairs, so all three strategies are
compared on identical pairs and identical budgets (5 % / 10 % of |E|).

    python scripts/run_adaptive_attack.py --dataset cora --models gcn sage
    python scripts/run_adaptive_attack.py --dataset usair --models gcn --budgets 0.05

Output: ``results/adaptive_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.engine.baseline_trainer import node_feature_matrix  # noqa: E402
from sgdiff.engine.benchmark import build_registry, fit_model  # noqa: E402
from sgdiff.engine.factory import load_split, prepare_datasets  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402


def degrees(graph) -> np.ndarray:
    deg = np.zeros(graph.num_nodes, dtype=np.int64)
    edges = np.asarray(graph.edges, dtype=np.int64)
    if len(edges):
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    return deg


def cn_matrix(graph):
    adj = graph.adjacency().tocsr().astype(np.float64)
    return (adj @ adj).tocsr()


def evidence_pool(graph, pos_pairs: np.ndarray) -> np.ndarray:
    """Load-bearing edges of the *current* graph, for the given positive pairs.

    Two tiers: (1) the edges that carry the shared neighbour of a pair with
    ``CN >= 1`` -- deleting them is what removes the evidence; (2) failing that,
    any edge incident to a positive endpoint, so the attacker never runs out of
    moves before the budget is spent.
    """
    neighbours = graph.neighbors()
    tier1: list[tuple[int, int]] = []
    tier2: list[tuple[int, int]] = []
    for u, v in pos_pairs:
        nu, nv = neighbours[u], neighbours[v]
        shared = set(nu) & set(nv)
        if shared:
            for w in shared:
                tier1.append((min(u, int(w)), max(u, int(w))))
                tier1.append((min(v, int(w)), max(v, int(w))))
        else:
            for w in nu:
                tier2.append((min(u, int(w)), max(u, int(w))))
            for w in nv:
                tier2.append((min(v, int(w)), max(v, int(w))))
    pool = tier1 if tier1 else tier2
    if not pool:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pool, dtype=np.int64), axis=0)


def pick_random(graph, rng, taken: set) -> tuple[int, int] | None:
    edges = np.asarray(graph.edges, dtype=np.int64)
    for _ in range(50):
        e = edges[rng.integers(len(edges))]
        key = (int(min(e)), int(max(e)))
        if key not in taken:
            return key
    return None


def pick_hub(graph, taken: set) -> tuple[int, int] | None:
    deg = degrees(graph)
    edges = np.asarray(graph.edges, dtype=np.int64)
    score = np.maximum(deg[edges[:, 0]], deg[edges[:, 1]])
    for idx in np.argsort(-score):
        key = (int(edges[idx, 0]), int(edges[idx, 1]))
        if key not in taken:
            return key
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--models", nargs="*", default=["gcn", "sage"])
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.05, 0.10])
    parser.add_argument("--pool", type=int, default=32, help="candidate edges scored per adaptive step")
    parser.add_argument("--pairs", type=int, default=200, help="working set size per class")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"] + rp.PROFILES[args.profile] + args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)
    device = resolve_device(args.device)
    registry = build_registry(cfg)

    results: dict[str, dict] = {}
    for seed in args.seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        cfg_seed.train.device = str(device)
        set_seed(int(seed))

        graph, split = load_split(cfg_seed)
        datasets = prepare_datasets(graph, split, cfg_seed)
        pos = np.asarray(split.test_pos, dtype=np.int64)[: args.pairs]
        neg = np.asarray(split.test_neg, dtype=np.int64)[: args.pairs]
        pairs = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()
        clean = graph.drop_edges(test_edges)
        features = node_feature_matrix(clean)  # frozen attributes, as in the probes
        n_edges = int(clean.num_edges)

        def score(g) -> np.ndarray:
            return rp.score_model("node_pair", model, cfg_seed, g, pairs, device, int(seed), features=features)

        def auc_of(g) -> float:
            return float(summarize_scores(labels, score(g))["auc"])

        for name in args.models:
            spec = registry[name]
            set_seed(int(seed))
            started = time.perf_counter()
            model, _m, _p = fit_model(
                spec, cfg_seed, graph, split, datasets, int(seed), device=device,
                run_root=f"runs/adaptive/{cfg_seed.data.name}",
            )
            fit_seconds = time.perf_counter() - started
            base_scores = score(clean)
            base_auc = float(summarize_scores(labels, base_scores)["auc"])

            for budget in args.budgets:
                steps = max(1, int(round(budget * n_edges)))
                for strategy in ("random", "hub", "adaptive"):
                    rng = np.random.default_rng(int(seed) + 7919)
                    g = clean
                    taken: set[tuple[int, int]] = set()
                    deleted_cn: list[float] = []
                    step_started = time.perf_counter()
                    for _ in range(steps):
                        if strategy == "random":
                            edge = pick_random(g, rng, taken)
                        elif strategy == "hub":
                            edge = pick_hub(g, taken)
                        else:
                            # the pool is rebuilt on the current graph: as evidence
                            # paths are destroyed, new load-bearing edges appear
                            pool = evidence_pool(g, pos)
                            if len(pool) == 0:
                                break
                            fresh = [tuple(int(x) for x in row) for row in pool if tuple(int(x) for x in row) not in taken]
                            if not fresh:
                                break
                            idx = rng.choice(len(fresh), size=min(args.pool, len(fresh)), replace=False)
                            candidates = [fresh[i] for i in idx]
                            best, best_auc = None, np.inf
                            for cand in candidates:
                                trial_auc = float(summarize_scores(labels, score(g.drop_edges([cand])))["auc"])
                                if trial_auc < best_auc:
                                    best, best_auc = cand, trial_auc
                            edge = best
                        if edge is None:
                            break
                        a2 = cn_matrix(g)
                        deleted_cn.append(float(a2[edge[0], edge[1]]))
                        taken.add(edge)
                        g = g.drop_edges([edge])
                    elapsed = time.perf_counter() - step_started
                    auc = auc_of(g)
                    key = f"{name}"
                    per_model = results.setdefault(key, {"fit_seconds": fit_seconds, "clean_auc": base_auc, "seeds": {}})
                    per_model["seeds"].setdefault(int(seed), {})
                    per_model["seeds"][int(seed)].setdefault(f"{budget:.2f}", {})[strategy] = {
                        "auc": auc,
                        "damage": base_auc - auc,
                        "seconds": elapsed,
                        "edges_deleted": len(taken),
                        "deleted_cn_mean": float(np.mean(deleted_cn)) if deleted_cn else float("nan"),
                        "deleted_bridge_frac": float(np.mean(np.asarray(deleted_cn) == 0)) if deleted_cn else float("nan"),
                    }
                    print(
                        f"[{cfg_seed.data.name} s{seed}] {name} budget={budget:.2f} {strategy}: "
                        f"auc={auc:.3f} (Δ{base_auc - auc:+.3f}) {len(taken)} edges in {elapsed:.1f}s",
                        flush=True,
                    )

    summary = _summarise(args.dataset, args.budgets, results)
    stem = f"{args.output}/adaptive_{args.dataset}"
    save_json(summary, f"{stem}.json")
    Path(f"{stem}.md").write_text(to_markdown(summary) + "\n", encoding="utf-8")
    print(to_markdown(summary))
    print(f"\nwrote {stem}.json and .md")
    return 0


def _summarise(dataset: str, budgets: list[float], results: dict) -> dict:
    out = {"dataset": dataset, "budgets": list(budgets), "models": {}}
    for model, payload in results.items():
        agg: dict[str, dict[str, dict]] = {}
        for budget_key in payload["seeds"][next(iter(payload["seeds"]))]:
            agg[budget_key] = {}
            for strategy in ("random", "hub", "adaptive"):
                rows = [per[budget_key][strategy] for per in payload["seeds"].values() if strategy in per.get(budget_key, {})]
                if not rows:
                    continue
                agg[budget_key][strategy] = {
                    k: float(np.nanmean([r[k] for r in rows])) for k in ("auc", "damage", "seconds", "edges_deleted", "deleted_cn_mean", "deleted_bridge_frac")
                }
        out["models"][model] = {"clean_auc": payload["clean_auc"], "fit_seconds": payload["fit_seconds"], "by_budget": agg}
    return out


def to_markdown(result: dict) -> str:
    lines = [f"# Adaptive vs non-adaptive deletion: {result['dataset']}", ""]
    lines += [
        "Same model, same working pairs, same number of deleted edges; only the *choice* of edges differs.",
        "`adaptive` rebuilds the evidence pool on the current graph and deletes the candidate that costs the most AUC.",
        "",
        "| model | budget | random ΔAUC | hub ΔAUC | adaptive ΔAUC | extra damage vs random | deleted CN (adaptive) | bridge share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, payload in result["models"].items():
        for budget_key, per_strategy in sorted(payload["by_budget"].items()):
            rnd = per_strategy.get("random", {})
            hub = per_strategy.get("hub", {})
            ada = per_strategy.get("adaptive", {})
            if not ada:
                continue
            gain = ada["damage"] - rnd.get("damage", float("nan"))
            lines.append(
                f"| {model} | {float(budget_key):.0%} | {rnd.get('damage', float('nan')):+.3f} | "
                f"{hub.get('damage', float('nan')):+.3f} | {ada['damage']:+.3f} | {gain:+.3f} | "
                f"{ada['deleted_cn_mean']:.2f} | {ada['deleted_bridge_frac']:.2f} |"
            )
    lines += ["", "`clean` AUC per model: " + ", ".join(f"{m} {p['clean_auc']:.3f}" for m, p in result["models"].items()) + "."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
