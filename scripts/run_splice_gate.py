#!/usr/bin/env python
"""The gate §7.11 asks for: ask what the model *would* do with evidence.

Every gate tested so far observes *state* — "does this pair have a shared neighbour
on the scored graph" — and every one of them fails, because on a sparse citation
graph ``CN = 0`` is the normal state of the data, not an anomaly. This script
implements the alternative: a **counterfactual splice**.

For a pair ``(u, v)`` it adds one synthetic shared neighbour ``w`` (edges
``u-w`` and ``v-w``, DRNL label 2, features taken from ``w``'s own attributes) to
that pair's enclosing sub-graph, re-scores it with the *same* k = 1 model, and
records

    Δ(u, v) = score(spliced) − score(plain)

Δ asks what the model does when evidence appears, so it is defined for every pair
and does not degenerate under attack. The gate routes to the k = 2 model when
``Δ > 0`` — i.e. when the pair is *responsive to evidence* rather than merely
missing it.

Three things are measured:

* does the splice routing repair the deletion gap (clean AUC, `del`, `del_hub`,
  `ins_hub`) against plain k = 1, always k = 2, and the `CN = 0` gate of §7.11;
* is the gate signal itself informative — AUC of ``Δ`` at predicting which pairs
  lose score under deletion, computed on the scored graph only;
* how often it fires before vs after the attack (a gate that fires on 90 % of
  pairs after deletion is the k = 2 model wearing a hat).

    python scripts/run_splice_gate.py --dataset cora --models seal pencil --seeds 0 1

Output: ``results/splice_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from run_bootstrap_ci import fast_auc  # noqa: E402
from run_gated_repair import cn_of, damage_delta_bootstrap  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.data.subgraph import Subgraph, drnl_label  # noqa: E402
from sgdiff.engine.baseline_trainer import node_feature_matrix  # noqa: E402
from sgdiff.engine.batch import PairSubgraphDataset, collate_subgraphs  # noqa: E402
from sgdiff.engine.benchmark import build_registry, fit_model  # noqa: E402
from sgdiff.engine.factory import build_extractor, load_split, prepare_datasets  # noqa: E402
from sgdiff.engine.robustness import _cap_pairs  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

VARIANTS = ("clean", "del@50%", "del_hub@50%", "ins_hub@50%")
QUANTILES = (0.5, 0.7, 0.9)


def splice_subgraph(sub: Subgraph, w: int, row: np.ndarray | None, max_label: int) -> Subgraph:
    """Add a synthetic shared neighbour ``w`` to the pair of ``sub``."""
    n = sub.num_nodes
    adj = np.zeros((n + 1, n + 1), dtype=np.float32)
    adj[:n, :n] = sub.adj
    u, v = sub.u_idx, sub.v_idx
    adj[u, n] = adj[n, u] = 1.0
    adj[v, n] = adj[n, v] = 1.0
    labels = np.append(sub.labels, drnl_label(1, 1, max_label))
    x = None
    if sub.x is not None:
        fill = row if row is not None else np.zeros(sub.x.shape[1], dtype=np.float32)
        x = np.vstack([sub.x, fill[None, :].astype(np.float32)])
    return Subgraph(src=sub.src, dst=sub.dst, nodes=np.append(sub.nodes, w), adj=adj, labels=labels, x=x)


def _forward(model, subs: list[Subgraph], feature_dim: int, batch_size: int, device) -> np.ndarray:
    loader = DataLoader(
        list(range(len(subs))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda idx: collate_subgraphs([(subs[i], 0) for i in idx], feature_dim=feature_dim),
    )
    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out.append(model.predict_logit(batch.to(device)).detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def spliced_deltas(
    model,
    eval_graph,
    cfg,
    pairs: np.ndarray,
    feature_dim_hint: int | None,
    seed: int,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every pair plain and with one synthetic shared neighbour spliced in."""
    extractor = build_extractor(eval_graph, cfg, seed=cfg.data.seed + 1)
    dataset = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
    batch_size = int(getattr(cfg.train, "eval_batch_size", 64))
    feature_dim = feature_dim_hint or dataset.feature_dim
    base = _forward(model, [dataset[i][0] for i in range(len(dataset))], feature_dim, batch_size, device)

    rng = np.random.default_rng(int(seed) + 17)
    spliced: list[Subgraph] = []
    blank: list[Subgraph] = []
    for i in range(len(dataset)):
        sub, _ = dataset[i]
        excluded = set(int(n) for n in sub.nodes)
        w = 0
        for _ in range(20):
            w = int(rng.integers(0, eval_graph.num_nodes))
            if w not in excluded:
                break
        row = None
        if extractor._features is not None:
            row = np.asarray(extractor._features[w], dtype=np.float32)
        spliced.append(splice_subgraph(sub, w, row, extractor.max_label))
        blank.append(splice_subgraph(sub, w, None, extractor.max_label))
    with_splice = _forward(model, spliced, feature_dim, batch_size, device)
    with_blank = _forward(model, blank, feature_dim, batch_size, device)
    return base, with_splice, with_blank, extractor._features is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--models", nargs="*", default=["seal", "pencil"])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--max-pairs", type=int, default=500)
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

    raw: dict[tuple[str, str, str, int], float] = {}
    boot: dict[tuple[str, str, str, int], dict] = {}
    diag: dict[tuple[str, str, int], dict] = {}

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

        graphs: dict[str, object] = {}
        for variant in args.variants:
            if variant == "clean":
                graphs[variant] = clean_graph
                continue
            try:
                graphs[variant] = PERTURBATIONS[variant](graph, int(seed)).drop_edges(test_edges)
            except ValueError as exc:
                print(f"skip {variant}: {exc}", flush=True)

        for name in args.models:
            spec = registry[name]
            if spec.kind != "subgraph":
                print(f"skip {name}: splice gate needs a sub-graph model, got {spec.kind}", flush=True)
                continue
            started = time.perf_counter()
            set_seed(int(seed))
            model_k1, _m, _p = rp.fit_model(
                spec, cfg_seed, graph, split, datasets, int(seed), device=device,
                run_root=f"runs/splice/{cfg_seed.data.name}/k1",
            )
            cfg_k2 = copy.deepcopy(cfg_seed)
            cfg_k2.subgraph.hops = 2
            datasets_k2 = rp.prepare_datasets(graph, split, cfg_k2)
            set_seed(int(seed))
            model_k2, _m2, _p2 = rp.fit_model(
                spec, cfg_k2, graph, split, datasets_k2, int(seed), device=device,
                run_root=f"runs/splice/{cfg_seed.data.name}/k2",
            )
            model_k1.eval()
            model_k2.eval()

            scored: dict[str, dict[str, np.ndarray]] = {}
            for variant, eval_graph in graphs.items():
                set_seed(int(seed))
                s1 = rp.score_model(spec.kind, model_k1, cfg_seed, eval_graph, pairs, device, int(seed), clean_features)
                set_seed(int(seed))
                s2 = rp.score_model(spec.kind, model_k2, cfg_k2, eval_graph, pairs, device, int(seed), clean_features)
                plain, spliced, spliced_blank, has_features = spliced_deltas(
                    model_k1, eval_graph, cfg_seed, pairs, None, int(seed), device
                )
                if has_features:
                    # the two forward passes must agree on the un-spliced pairs
                    assert np.allclose(plain, s1, atol=1e-4), "splice scorer disagrees with score_model"
                delta = spliced - plain
                delta_blank = spliced_blank - plain
                cn = cn_of(eval_graph, pairs)
                scored[variant] = {"s1": s1, "s2": s2, "cn": cn, "delta": delta, "delta_blank": delta_blank}
                raw[(name, "k1", variant, int(seed))] = float(summarize_scores(labels, s1)["auc"])
                raw[(name, "k2", variant, int(seed))] = float(summarize_scores(labels, s2)["auc"])
                raw[(name, "cn_gate", variant, int(seed))] = float(
                    summarize_scores(labels, np.where(cn == 0, s2, s1))["auc"]
                )
                raw[(name, "splice_sign", variant, int(seed))] = float(
                    summarize_scores(labels, np.where(delta > 0, s2, s1))["auc"]
                )
                for q in QUANTILES:
                    arm = f"splice_q{int(q * 100)}"
                    gate = delta >= np.quantile(delta, q)
                    raw[(name, arm, variant, int(seed))] = float(
                        summarize_scores(labels, np.where(gate, s2, s1))["auc"]
                    )
                print(
                    f"[{cfg_seed.data.name} s{seed}] {name} {variant}: "
                    f"k1={raw[(name, 'k1', variant, int(seed))]:.3f} "
                    f"q50={raw[(name, 'splice_q50', variant, int(seed))]:.3f} "
                    f"k2={raw[(name, 'k2', variant, int(seed))]:.3f} "
                    f"(sign gate fires {(delta > 0).mean():.1%}, Δ median {np.median(delta):+.4f})",
                    flush=True,
                )

            clean = scored["clean"]
            for variant, probe in scored.items():
                if variant == "clean":
                    continue
                drop = clean["s1"] - probe["s1"]
                q50_probe = probe["delta"] >= np.quantile(probe["delta"], 0.5)
                q50_clean = clean["delta"] >= np.quantile(clean["delta"], 0.5)
                diag[(name, variant, int(seed))] = {
                    "fire_clean_sign": float((clean["delta"] > 0).mean()),
                    "fire_probe_sign": float((probe["delta"] > 0).mean()),
                    "fire_clean_q50": float(q50_clean.mean()),
                    "fire_probe_q50": float(q50_probe.mean()),
                    "cn_zero_clean": float((clean["cn"] == 0).mean()),
                    "cn_zero_probe": float((probe["cn"] == 0).mean()),
                    "auc_delta_predicts_drop": float(fast_auc((drop > np.median(drop)).astype(np.int64), probe["delta"])),
                    "overlap_q50_with_cn_gate": float((q50_probe & (probe["cn"] == 0)).sum() / max(int(q50_probe.sum()), 1)),
                    "delta_median": float(np.median(probe["delta"])),
                    "delta_positive_share": float((probe["delta"] > 0).mean()),
                    "delta_blank_positive_share": float((probe["delta_blank"] > 0).mean()),
                    "corr_own_vs_blank": _safe_corr(probe["delta"], probe["delta_blank"]),
                }
                arms_clean = {"k1": clean["s1"], "k2": clean["s2"], "splice_q50": np.where(q50_clean, clean["s2"], clean["s1"])}
                arms_probe = {"k1": probe["s1"], "k2": probe["s2"], "splice_q50": np.where(q50_probe, probe["s2"], probe["s1"])}
                found = damage_delta_bootstrap(labels, arms_clean, arms_probe, args.reps, int(seed) + 2000)
                for arm, payload in found.items():
                    if arm == "k1":
                        continue
                    boot[(name, arm, variant, int(seed))] = payload
            print(f"[{cfg_seed.data.name} s{seed}] {name}: fits+scores {time.perf_counter() - started:.0f}s", flush=True)

    result = _summarise(args.dataset, raw, boot, diag, args.models, args.reps)
    stem = f"{args.output}/splice_{args.dataset}"
    save_json(result, f"{stem}.json")
    Path(f"{stem}.md").write_text(to_markdown(result) + "\n", encoding="utf-8")
    print(to_markdown(result))
    print(f"\nwrote {stem}.json and .md")
    return 0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _summarise(dataset: str, raw: dict, boot: dict, diag: dict, models: list[str], reps: int) -> dict:
    arms = ("k1", "k2", "cn_gate", "splice_sign") + tuple(f"splice_q{int(q * 100)}" for q in QUANTILES)
    out: dict[str, dict] = {"dataset": dataset, "reps": reps, "models": {}, "bootstrap": {}, "gate": {}}
    for name in models:
        cells: dict[str, dict[str, float]] = {}
        for variant in VARIANTS:
            per_arm = {}
            for arm in arms:
                vals = [v for (m, a, var, _s), v in raw.items() if m == name and a == arm and var == variant]
                if vals:
                    per_arm[arm] = _mean(vals)
            if per_arm:
                cells[variant] = per_arm
        if cells:
            out["models"][name] = cells
        per_gate: dict[str, dict[str, float]] = {}
        for variant in VARIANTS:
            rows = [d for (m, var, _s), d in diag.items() if m == name and var == variant]
            if rows:
                per_gate[variant] = {k: _mean([r[k] for r in rows]) for k in rows[0]}
        if per_gate:
            out["gate"][name] = per_gate
        per_boot: dict[str, dict] = {}
        for variant in VARIANTS:
            for arm in ("k2", "splice_q50"):
                entries = [b for (m, a, var, _s), b in boot.items() if m == name and a == arm and var == variant]
                if not entries:
                    continue
                per_boot.setdefault(variant, {})[f"{arm}-k1"] = {
                    "delta": _mean([e["delta"] for e in entries]),
                    "ci_low": _mean([e["ci_low"] for e in entries]),
                    "ci_high": _mean([e["ci_high"] for e in entries]),
                    "p_wrong_sign": _mean([e["p_wrong_sign"] for e in entries]),
                    "seeds": len(entries),
                }
        if per_boot:
            out["bootstrap"][name] = per_boot
    return out


def to_markdown(result: dict) -> str:
    lines = [f"# Counterfactual splice gate: {result['dataset']}", ""]
    lines += [
        "For every pair the k = 1 model is scored twice: once on its enclosing sub-graph, once with a",
        "synthetic shared neighbour `w` spliced in (`u-w`, `v-w`, DRNL label 2). The gate routes to the",
        "k = 2 model for the pairs whose score the splice raises most. Damage = clean − probe.",
        "",
        "| model | probe | k = 1 | k = 2 (global) | `CN = 0` gate | splice gate (top 50 %) | splice − k1 | 95 % CI (paired) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, cells in result["models"].items():
        clean = cells.get("clean", {})
        for variant, row in cells.items():
            ci = result.get("bootstrap", {}).get(model, {}).get(variant, {}).get("splice_q50-k1")
            ci_txt = f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]" if ci else "—"
            if variant == "clean":
                k1 = clean.get("k1", float("nan"))
                routed = clean.get("splice_q50", float("nan"))
                lines.append(
                    f"| {model} | clean | {k1:.3f} | {clean.get('k2', float('nan')):.3f} | "
                    f"{clean.get('cn_gate', float('nan')):.3f} | {routed:.3f} | {routed - k1:+.3f} | {ci_txt} |"
                )
                continue
            dmg = {
                arm: clean.get(arm, float("nan")) - row.get(arm, float("nan"))
                for arm in ("k1", "k2", "cn_gate", "splice_q50")
            }
            lines.append(
                f"| {model} | {variant} | {dmg['k1']:+.3f} | {dmg['k2']:+.3f} | {dmg['cn_gate']:+.3f} | "
                f"{dmg['splice_q50']:+.3f} | {dmg['k1'] - dmg['splice_q50']:+.3f} | {ci_txt} |"
            )
    lines += [
        "",
        "Positive `splice − k1` on a damage row = the splice gate shrinks the damage; on the clean row it is",
        "the ceiling it buys. The interval is a paired bootstrap over scored pairs for the damage difference;",
        "it covers pair sampling, not training seeds.",
        "",
        "## Gate frontier: sign is useless, magnitude is not",
        "",
        "The splice raises the score for almost every pair, so the *sign* gate degenerates into \"always k2\".",
        "The quantile gates route only the pairs with the largest rise; `CN = 0` is the §7.11 gate, shown for",
        "reference.",
        "",
        "| model | probe | `CN = 0` | sign(Δ) | top 50 % | top 30 % | top 10 % | always k2 | k = 1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, cells in result["models"].items():
        clean = cells.get("clean", {})
        for variant, row in cells.items():
            if variant == "clean":
                continue
            arms = ["cn_gate", "splice_sign", "splice_q50", "splice_q70", "splice_q90", "k2", "k1"]
            vals = " | ".join(
                f"{clean.get(a, float('nan')) - row.get(a, float('nan')):+.3f}" for a in arms
            )
            lines.append(f"| {model} | {variant} | {vals} |")
    lines += [
        "",
        "## Is the gate signal informative?",
        "",
        "`Δ` is computed on the graph being scored (deployment-time signal). `AUC(Δ → drop)` asks whether the",
        "magnitude ranks the pairs that lose score under deletion above the ones that do not; 0.5 is chance.",
        "",
        "| model | probe | sign gate fires | top 50 % fires | `CN = 0` share | Δ > 0 (own feats) | Δ > 0 (blank feats) | corr(own, blank) | AUC(Δ → drop) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, gates in result["gate"].items():
        for variant, row in gates.items():
            if variant == "clean":
                continue
            lines.append(
                f"| {model} | {variant} | {row['fire_probe_sign']:.1%} | {row['fire_probe_q50']:.1%} | "
                f"{row['cn_zero_probe']:.1%} | {row['delta_positive_share']:.1%} | "
                f"{row['delta_blank_positive_share']:.1%} | {row['corr_own_vs_blank']:.3f} | "
                f"{row['auc_delta_predicts_drop']:.3f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
