#!/usr/bin/env python
"""A repair that follows from §7.7: route on evidence, instead of getting more of it.

§7.5/§7.6 showed that giving the sub-graph family 2-hop sub-graphs *globally*
raises the ceiling and leaves the deletion gap unchanged. §7.7 then showed why the
average is the wrong object: the loss is concentrated on the pairs whose shared
neighbour was deleted, and zero (or negative) on the pairs that kept it. That
suggests a *conditional* repair rather than a bigger model:

    score(u, v) = k2_model(u, v)   if the scored graph has no evidence for (u, v)
                  k1_model(u, v)   otherwise

i.e. train both extractors, keep both models, and route per pair on a signal that
is observable at deployment time (``CN == 0`` on the graph being scored). This
script measures the router against its two components, and against three things
the single-point version could not answer:

* **is there a threshold that works?**  The gate is swept over ``CN <= tau`` for
  ``tau = 0, 1, 2`` (with "always k2" as the endpoint), because a repair that only
  fires on low-evidence pairs is only as good as its best operating point.
* **how large is each difference?**  Every arm difference gets a paired bootstrap
  interval over the scored pairs (clean and probe resampled jointly, which is the
  quantity "does the router spend damage" is actually about), so the 0.02-0.06
  scale claims are intervals, not point estimates.
* **can CN completion even fire on those pairs?**  For every scored pair with
  ``CN == 0`` the script rebuilds the deleted edges with the CN prior used in
  §7.5 and re-counts common neighbours: a completion repair that cannot create
  evidence for the pairs it targets is not a repair, whatever its AUC.

    python scripts/run_gated_repair.py --dataset cora --models seal pencil --seeds 0 1

Output: ``results/gated_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from run_bootstrap_ci import fast_auc  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.engine.baseline_trainer import node_feature_matrix  # noqa: E402
from sgdiff.engine.benchmark import build_registry, fit_model  # noqa: E402
from sgdiff.engine.factory import load_split, prepare_datasets  # noqa: E402
from sgdiff.engine.robustness import _cap_pairs  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

VARIANTS = ("clean", "del@50%", "del_hub@50%", "ins_hub@50%")


def cn_of(graph, pairs: np.ndarray) -> np.ndarray:
    """Common neighbours of each pair on the graph being scored (deployment signal)."""
    adj = graph.adjacency().tocsr().astype(np.float64)
    a2 = (adj @ adj).tocsr()
    return np.asarray(a2[pairs[:, 0], pairs[:, 1]]).ravel()


def arm_name(tau: int | None) -> str:
    return "k2" if tau is None else f"route<={tau}"


def arm_label(tau: int | None) -> str:
    return "always k2" if tau is None else f"`CN ≤ {tau}`"


def routed_scores(s1: np.ndarray, s2: np.ndarray, cn: np.ndarray, tau: int | None) -> np.ndarray:
    if tau is None:
        return s2
    return np.where(cn <= tau, s2, s1)


def damage_delta_bootstrap(
    labels: np.ndarray,
    arms_clean: dict[str, np.ndarray],
    arms_probe: dict[str, np.ndarray],
    reps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired-bootstrap interval for ``damage(k1) - damage(arm)``.

    Positive = the arm spends *less* damage than plain k = 1. The clean and the
    probe scores are resampled on the **same** pairs, and the clean/probe AUC
    difference is taken inside every resample, so the interval is about the damage
    difference itself rather than about an AUC difference on one graph.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    arms = list(arms_clean)

    def statistic(idx: np.ndarray | None) -> dict[str, float]:
        lab = labels if idx is None else labels[idx]
        auc = {arm: fast_auc(lab, arms_clean[arm] if idx is None else arms_clean[arm][idx]) for arm in arms}
        auc_probe = {arm: fast_auc(lab, arms_probe[arm] if idx is None else arms_probe[arm][idx]) for arm in arms}
        base = auc["k1"] - auc_probe["k1"]
        return {arm: float(base - (auc[arm] - auc_probe[arm])) for arm in arms}

    base = statistic(None)
    draws = {arm: np.empty(reps, dtype=np.float64) for arm in arms}
    for i in range(reps):
        idx = rng.integers(0, n, n)
        point = statistic(idx)
        for arm in arms:
            draws[arm][i] = point[arm]
    out: dict[str, dict[str, float]] = {}
    for arm in arms:
        lo, hi = np.percentile(draws[arm], [2.5, 97.5])
        wrong = draws[arm] <= 0.0 if base[arm] > 0 else draws[arm] >= 0.0
        out[arm] = {
            "delta": float(base[arm]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_wrong_sign": float(wrong.mean()),
        }
    return out


def completion_reach(graph, scored_graph, pairs: np.ndarray, forbidden: set[int]) -> dict[str, float] | None:
    """Can the CN-completion repair (as used in §7.5) reach the low-evidence pairs?

    Rebuild the deleted edges with the CN prior, then re-count common neighbours on
    the completed graph. A pair with ``CN = 0`` on the scored graph only benefits if
    the completion creates a shared neighbour for it. Insertion graphs have nothing
    to rebuild, so they return ``None``.
    """
    from run_deep_probe import complete_edges  # local import: heavy module

    removed = int(graph.num_edges - scored_graph.num_edges)
    if removed <= 0:
        return None
    completed = complete_edges(scored_graph, removed, "cn", forbidden)
    before = cn_of(scored_graph, pairs)
    after = cn_of(completed, pairs)
    low = before == 0
    if not low.any():
        return {"low_share": 0.0, "recovered_share": 0.0, "added_edges": int(removed)}
    return {
        "low_share": float(low.mean()),
        "recovered_share": float((after[low] > 0).mean()),
        "added_edges": int(removed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--models", nargs="*", default=["seal", "pencil"])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--thresholds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    thresholds: list[int | None] = [int(t) for t in args.thresholds] + [None]
    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"] + rp.PROFILES[args.profile] + args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)
    device = resolve_device(args.device)
    registry = build_registry(cfg)

    raw: dict[tuple[str, str, str, int], float] = {}
    boot: dict[tuple[str, str, str, int], dict] = {}
    gates: dict[tuple[str, int], dict] = {}
    reach: dict[tuple[str, int], dict] = {}

    for seed in args.seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        cfg_seed.train.device = str(device)
        set_seed(int(seed))

        graph, split = load_split(cfg_seed)
        datasets = prepare_datasets(graph, split, cfg_seed)
        pos, neg = _cap_pairs(split, args.max_pairs)
        pairs = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()
        clean_graph = graph.drop_edges(test_edges)
        clean_features = node_feature_matrix(clean_graph)
        forbidden = {int(min(u, v) * graph.num_nodes + max(u, v)) for u, v in np.asarray(pairs)}

        graphs: dict[str, object] = {}
        for variant in args.variants:
            if variant == "clean":
                graphs[variant] = clean_graph
                continue
            try:
                graphs[variant] = PERTURBATIONS[variant](graph, int(seed)).drop_edges(test_edges)
            except ValueError as exc:
                print(f"skip {variant}: {exc}", flush=True)

        # ---- deployment signals and the reach of a CN-completion repair
        for variant, eval_graph in graphs.items():
            cn = cn_of(eval_graph, pairs)
            gates[(variant, int(seed))] = {
                "gate_share": {str(t): float((cn <= int(t)).mean()) for t in args.thresholds},
            }
            if variant != "clean":
                found = completion_reach(clean_graph, eval_graph, pairs, forbidden)
                if found is not None:
                    reach[(variant, int(seed))] = found

        for name in args.models:
            spec = registry[name]
            if spec.kind != "subgraph":
                print(f"skip {name}: gated routing needs a sub-graph model, got {spec.kind}", flush=True)
                continue
            started = time.perf_counter()
            set_seed(int(seed))
            model_k1, _m, _p = rp.fit_model(
                spec, cfg_seed, graph, split, datasets, int(seed), device=device,
                run_root=f"runs/gated/{cfg_seed.data.name}/k1",
            )
            cfg_k2 = copy.deepcopy(cfg_seed)
            cfg_k2.subgraph.hops = 2
            datasets_k2 = rp.prepare_datasets(graph, split, cfg_k2)
            set_seed(int(seed))
            model_k2, _m2, _p2 = rp.fit_model(
                spec, cfg_k2, graph, split, datasets_k2, int(seed), device=device,
                run_root=f"runs/gated/{cfg_seed.data.name}/k2",
            )
            model_k1.eval()
            model_k2.eval()

            scored: dict[str, dict[str, np.ndarray]] = {}
            for variant, eval_graph in graphs.items():
                set_seed(int(seed))
                s1 = rp.score_model(spec.kind, model_k1, cfg_seed, eval_graph, pairs, device, int(seed), clean_features)
                set_seed(int(seed))
                s2 = rp.score_model(spec.kind, model_k2, cfg_k2, eval_graph, pairs, device, int(seed), clean_features)
                cn = cn_of(eval_graph, pairs)
                scored[variant] = {"s1": s1, "s2": s2, "cn": cn}
                raw[(name, "k1", variant, int(seed))] = float(summarize_scores(labels, s1)["auc"])
                for tau in thresholds:
                    arm = arm_name(tau)
                    scores = routed_scores(s1, s2, cn, tau)
                    raw[(name, arm, variant, int(seed))] = float(summarize_scores(labels, scores)["auc"])
                print(
                    f"[{cfg_seed.data.name} s{seed}] {name} {variant}: "
                    f"k1={raw[(name, 'k1', variant, int(seed))]:.3f} "
                    f"k2={raw[(name, 'k2', variant, int(seed))]:.3f} "
                    f"routed={raw[(name, 'route<=0', variant, int(seed))]:.3f} "
                    f"(gate on {int((cn == 0).sum())}/{len(cn)})",
                    flush=True,
                )

            # ---- paired bootstrap on the *damage* difference, clean vs probe
            clean = scored.get("clean")
            if clean is not None:
                arm_taus: dict[str, int] = {"k1": -1}
                for tau in args.thresholds:
                    arm_taus[arm_name(tau)] = int(tau)

                def _arms(bucket: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                    out = {"k1": bucket["s1"]}
                    for arm, tau in arm_taus.items():
                        if arm == "k1":
                            continue
                        out[arm] = routed_scores(bucket["s1"], bucket["s2"], bucket["cn"], tau)
                    return out

                for variant, probe in scored.items():
                    if variant == "clean":
                        continue
                    found = damage_delta_bootstrap(
                        labels, _arms(clean), _arms(probe), args.reps, int(seed) + 1000
                    )
                    for arm, payload in found.items():
                        if arm == "k1":
                            continue
                        boot[(name, arm, variant, int(seed))] = payload
            print(f"[{cfg_seed.data.name} s{seed}] {name}: fits {time.perf_counter() - started:.0f}s", flush=True)

    result = _summarise(args.dataset, raw, boot, gates, reach, args.models, args.thresholds, args.reps)
    stem = f"{args.output}/gated_{args.dataset}"
    save_json(result, f"{stem}.json")
    Path(f"{stem}.md").write_text(to_markdown(result) + "\n", encoding="utf-8")
    print(to_markdown(result))
    print(f"\nwrote {stem}.json and .md")
    return 0


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _summarise(
    dataset: str,
    raw: dict,
    boot: dict,
    gates: dict,
    reach: dict,
    models: list[str],
    thresholds: list[int],
    reps: int,
) -> dict:
    arms = ["k1", "k2"] + [arm_name(t) for t in thresholds]
    out: dict[str, dict] = {
        "dataset": dataset,
        "thresholds": thresholds,
        "reps": reps,
        "models": {},
        "gate": {},
        "completion": {},
        "bootstrap": {},
    }
    for name in models:
        cells: dict[str, dict[str, float]] = {}
        for variant in VARIANTS:
            per_arm: dict[str, float] = {}
            for arm in arms:
                vals = [v for (m, a, var, _s), v in raw.items() if m == name and a == arm and var == variant]
                if vals:
                    per_arm[arm] = _mean(vals)
            if per_arm:
                cells[variant] = per_arm
        if cells:
            out["models"][name] = cells

    for variant in VARIANTS:
        shares = [g["gate_share"] for (var, _s), g in gates.items() if var == variant]
        if shares:
            out["gate"][variant] = {
                "share": {str(t): _mean([s[str(t)] for s in shares]) for t in thresholds},
            }
    for variant in VARIANTS:
        rows = [r for (var, _s), r in reach.items() if var == variant]
        if rows:
            out["completion"][variant] = {k: _mean([r[k] for r in rows]) for k in rows[0]}

    for name in models:
        for variant in VARIANTS:
            per: dict[str, dict[str, float]] = {}
            for arm in [arm_name(t) for t in thresholds] + ["k2"]:
                entries = [b for (m, a, var, _s), b in boot.items() if m == name and a == arm and var == variant]
                if not entries:
                    continue
                per[f"{arm}-k1"] = {
                    "delta": _mean([e["delta"] for e in entries]),
                    "ci_low": _mean([e["ci_low"] for e in entries]),
                    "ci_high": _mean([e["ci_high"] for e in entries]),
                    "p_wrong_sign": _mean([e["p_wrong_sign"] for e in entries]),
                    "seeds": len(entries),
                }
            if per:
                out["bootstrap"].setdefault(name, {})[variant] = per
    return out


def to_markdown(result: dict) -> str:
    thresholds = result.get("thresholds", [0])
    base_arm = f"route<={thresholds[0]}"
    lines = [f"# Evidence-gated routing repair: {result['dataset']}", ""]
    lines += [
        "Two sub-graph models are trained (k = 1 and k = 2); the router keeps the k2 score",
        f"for pairs with `CN <= {thresholds[0]}` on the graph being scored. Damage = clean − probe.",
        "",
        f"| model | probe | k = 1 | k = 2 (global) | routed (`CN ≤ {thresholds[0]}`) | routed − k1 | 95 % CI (paired) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model, cells in result["models"].items():
        clean = cells.get("clean", {})
        for variant, arms in cells.items():
            ci = result.get("bootstrap", {}).get(model, {}).get(variant, {}).get(f"{base_arm}-k1")
            ci_txt = f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]" if ci else "—"
            if variant == "clean":
                routed = arms.get(base_arm, float("nan"))
                k1 = arms.get("k1", float("nan"))
                lines.append(
                    f"| {model} | clean | {k1:.3f} | {arms.get('k2', float('nan')):.3f} | "
                    f"{routed:.3f} | {routed - k1:+.3f} | {ci_txt} |"
                )
                continue
            dmg = {arm: clean.get(arm, float("nan")) - arms.get(arm, float("nan")) for arm in ("k1", "k2", base_arm)}
            lines.append(
                f"| {model} | {variant} | {dmg['k1']:+.3f} | {dmg['k2']:+.3f} | {dmg[base_arm]:+.3f} | "
                f"{dmg['k1'] - dmg[base_arm]:+.3f} | {ci_txt} |"
            )
    lines += [
        "",
        "Positive `routed − k1` on a damage row = the router shrinks the damage; on the clean row it is the",
        "ceiling the router buys (or costs). The interval is a paired bootstrap over the scored pairs for the",
        "*damage difference* (clean and probe resampled jointly); it covers pair sampling, not training seeds.",
        "",
        "## Routing frontier: is there a threshold that works?",
        "",
    ]
    arms = [f"route<={t}" for t in thresholds]
    lines.append("| model | probe | " + " | ".join(arms) + " | always k2 | k = 1 |")
    lines.append("|---|---|" + "---:|" * (len(arms) + 2))
    for model, cells in result["models"].items():
        clean = cells.get("clean", {})
        for variant, row in cells.items():
            if variant == "clean":
                continue
            vals = []
            for arm in arms + ["k2", "k1"]:
                vals.append(f"{clean.get(arm, float('nan')) - row.get(arm, float('nan')):+.3f}")
            lines.append(f"| {model} | {variant} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "Share of scored pairs the gate fires on (higher = more routing):",
        "",
        "| probe | " + " | ".join(f"`CN ≤ {t}`" for t in thresholds) + " |",
        "|---|" + "---:|" * len(thresholds),
    ]
    for variant, gate in result.get("gate", {}).items():
        cells = " | ".join(f"{gate['share'].get(str(t), float('nan')):.1%}" for t in thresholds)
        lines.append(f"| {variant} | {cells} |")
    if result.get("completion"):
        lines += [
            "",
            "## Can a CN-completion repair reach the low-evidence pairs?",
            "",
            "Deleted edges are rebuilt with the CN prior of §7.5 (same count as removed); a `CN = 0` pair",
            "benefits only if the completion creates a shared neighbour for *that pair*.",
            "",
            "| probe | `CN = 0` share | recovered by completion | edges added |",
            "|---|---:|---:|---:|",
        ]
        for variant, row in result["completion"].items():
            lines.append(
                f"| {variant} | {row['low_share']:.1%} | {row['recovered_share']:.1%} | {int(row['added_edges'])} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
