#!/usr/bin/env python
"""Is the damage in the ranking, or only at the operating point?

``run_probe.py`` reports one number per cell: AUC.  AUC is a *global* ranking
statistic and it is blind to the two things a deployed link predictor is judged
on -- the head of the ranking and the score margin.  This script re-reads the
per-pair score dumps of ``run_score_dump.py`` and answers:

1. **Operating point.**  ``ap`` (AUPRC), ``hit@100``, ``tpr@fpr=1%`` next to AUC.
   If a probe costs 0.05 AUC but 0.15 hit@100, the damage is in the head, i.e. the
   part that would actually be shipped as a top-k recommendation.
2. **Ranking vs distribution.**  ``d'`` (separation of the pos / neg score
   distributions), the global score shift, Kendall tau against the clean scores,
   top-100 retention, and the *inversion rate*: the fraction of (positive,
   negative) pairs that the clean model ordered correctly and the probe does not.
   AUC damage can come from a global shift (harmless for ranking), from
   compression, or from actual reordering -- these three are separable here.
3. **Where the damage lives.**  The same comparison inside strata of the scored
   graph: common-neighbour bucket (0 / 1 / 2+), endpoint-degree tertile, and
   same-label vs cross-label pairs.

    python scripts/run_operating_point.py --dataset cora
    python scripts/run_operating_point.py --dataset cora citeseer usair

Output: ``results/operating_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402

PRETTY = {
    "clean": "clean",
    "del@50%": "del",
    "del_hub@50%": "del_hub",
    "ins_hub@50%": "ins_hub",
    "feature_shuffle": "feat_shuffle",
}


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    """TPR at a fixed false-positive rate (threshold from the negatives)."""
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if not len(neg) or not len(pos):
        return float("nan")
    threshold = np.quantile(neg, 1.0 - fpr)
    return float((pos > threshold).mean())


def dprime(labels: np.ndarray, scores: np.ndarray) -> float:
    """Separation of the pos / neg score distributions, in pooled std units."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    pooled = np.sqrt(0.5 * (pos.var() + neg.var())) or 1e-12
    return float((pos.mean() - neg.mean()) / pooled)


def topk_retention(clean: np.ndarray, probe: np.ndarray, k: int = 100) -> float:
    a = np.argsort(-clean)[:k]
    b = np.argsort(-probe)[:k]
    return float(len(set(a) & set(b)) / k)


def inversion_rate(labels: np.ndarray, clean: np.ndarray, probe: np.ndarray) -> float:
    """Of the pos/neg pairs the clean model orders correctly, how many flip?"""
    pos_c, neg_c = clean[labels == 1], clean[labels == 0]
    pos_p, neg_p = probe[labels == 1], probe[labels == 0]
    if not len(pos_c) or not len(neg_c):
        return float("nan")
    correct = pos_c[:, None] > neg_c[None, :]
    kept = pos_p[:, None] > neg_p[None, :]
    total = correct.sum()
    if total == 0:
        return float("nan")
    return float((correct & ~kept).sum() / total)


def cell_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    out = {k: float(v) for k, v in summarize_scores(labels, scores).items()}
    out["tpr@fpr1%"] = tpr_at_fpr(labels, scores)
    out["dprime"] = dprime(labels, scores)
    return out


def strata(payload: dict, variant: str, clean_scores: np.ndarray, probe_scores: np.ndarray) -> dict:
    """AUC inside *clean-defined* strata, plus the pairs that lost their evidence.

    The partition uses the clean graph on purpose: grouping by the probe graph's
    own structure would condition on the outcome ("pairs that ended up with no
    shared neighbour") and call it an explanation.
    """
    labels = payload["labels"]
    clean_feat = {k.split("__")[-1]: v for k, v in payload.items() if k.startswith("feat__clean__")}
    probe_feat = {k.split("__")[-1]: v for k, v in payload.items() if k.startswith(f"feat__{variant}__")}
    if "cn" not in clean_feat:
        return {}
    cn = clean_feat["cn"]
    max_deg = np.maximum(clean_feat["deg_u"], clean_feat["deg_v"])
    groups: dict[str, np.ndarray] = {
        "cn=0": cn == 0,
        "cn=1": cn == 1,
        "cn>=2": cn >= 2,
        "deg:low": max_deg <= np.quantile(max_deg, 1 / 3),
        "deg:mid": (max_deg > np.quantile(max_deg, 1 / 3)) & (max_deg <= np.quantile(max_deg, 2 / 3)),
        "deg:high": max_deg > np.quantile(max_deg, 2 / 3),
    }
    if "cn" in probe_feat:
        groups["lost_evidence (CN≥1 → 0)"] = (cn >= 1) & (probe_feat["cn"] == 0)
        groups["kept_evidence (CN≥1 → ≥1)"] = (cn >= 1) & (probe_feat["cn"] >= 1)
    if "same_label" in clean_feat:
        groups["same_label"] = clean_feat["same_label"] == 1
        groups["cross_label"] = clean_feat["same_label"] == 0
    out: dict[str, dict] = {}
    scale = float(clean_scores.std() or 1.0)
    for name, mask in groups.items():
        n_pos = int((mask & (labels == 1)).sum())
        n_neg = int((mask & (labels == 0)).sum())
        if n_pos < 10:
            continue
        entry: dict[str, float] = {"n_pos": n_pos, "n_neg": n_neg}
        # score drop of each class inside the stratum, in clean-std units
        pos_mask, neg_mask = mask & (labels == 1), mask & (labels == 0)
        entry["d_pos"] = float((clean_scores[pos_mask] - probe_scores[pos_mask]).mean() / scale)
        entry["d_neg"] = (
            float((clean_scores[neg_mask] - probe_scores[neg_mask]).mean() / scale) if n_neg else float("nan")
        )
        if n_pos >= 20 and n_neg >= 20:
            clean_auc = float(summarize_scores(labels[mask], clean_scores[mask])["auc"])
            probe_auc = float(summarize_scores(labels[mask], probe_scores[mask])["auc"])
            entry.update({"clean": clean_auc, "probe": probe_auc, "damage": clean_auc - probe_auc})
        out[name] = entry
    return out


def analyse(dataset: str, scores_dir: Path) -> dict:
    files = sorted(scores_dir.glob(f"{dataset}_s*.npz"))
    if not files:
        raise SystemExit(f"no dumps for {dataset} in {scores_dir}; run scripts/run_score_dump.py first")
    dumps = [dict(np.load(f)) for f in files]
    models = sorted({k.split("__")[1] for k in dumps[0] if k.startswith("score__")})
    variants = [v for v in PRETTY if any(f"score__{models[0]}__{v}" in d for d in dumps)]

    metrics: dict[str, dict[str, dict]] = {m: {} for m in models}
    decomp: dict[str, dict[str, dict]] = {m: {} for m in models}
    strat: dict[str, dict[str, dict]] = {m: {} for m in models}

    for model in models:
        for variant in variants:
            per_seed_metrics, per_seed_decomp = [], []
            for payload in dumps:
                key = f"score__{model}__{variant}"
                if key not in payload:
                    continue
                labels, probe = payload["labels"], payload[key]
                clean = payload[f"score__{model}__clean"]
                per_seed_metrics.append(cell_metrics(labels, probe))
                scale = clean.std() or 1e-12
                per_seed_decomp.append(
                    {
                        "shift": float((probe - clean).mean() / scale),
                        "shift_pos": float((clean[labels == 1] - probe[labels == 1]).mean() / scale),
                        "shift_neg": float((clean[labels == 0] - probe[labels == 0]).mean() / scale),
                        "tau": float(kendalltau(clean, probe).statistic),
                        "top100_retention": topk_retention(clean, probe),
                        "inversion": inversion_rate(labels, clean, probe),
                    }
                )
                if variant != "clean":
                    per = strata(payload, variant, clean, probe)
                    if per:
                        strat[model].setdefault(variant, []).append(per)
            metrics[model][variant] = {k: float(np.nanmean([m[k] for m in per_seed_metrics])) for k in per_seed_metrics[0]}
            decomp[model][variant] = {k: float(np.nanmean([d[k] for d in per_seed_decomp])) for k in per_seed_decomp[0]}

    # average the strata over seeds
    strat_avg: dict[str, dict[str, dict]] = {}
    for model, per_variant in strat.items():
        strat_avg[model] = {}
        for variant, runs in per_variant.items():
            names = set().union(*[set(r) for r in runs])
            strat_avg[model][variant] = {}
            for name in names:
                rows = [r[name] for r in runs if name in r]
                keys = set().union(*[set(row) for row in rows])
                strat_avg[model][variant][name] = {
                    k: float(np.nanmean([row[k] for row in rows if k in row])) for k in keys
                }

    return {
        "dataset": dataset,
        "seeds": [f.name for f in files],
        "models": models,
        "variants": variants,
        "metrics": metrics,
        "decomposition": decomp,
        "strata": strat_avg,
    }


def to_markdown(result: dict) -> str:
    models = result["models"]
    probes = [v for v in result["variants"] if v != "clean"]
    lines = [f"# Operating point and ranking decomposition: {result['dataset']}", ""]

    lines += [
        "## 1. The same attack, four different verdicts",
        "",
        "Δ vs clean. `hit@100` = precision in the top-100; `tpr@fpr1%` = recall at a 1 % false-positive budget.",
        "",
        "| model | probe | ΔAUC | ΔAP | Δhit@100 | ΔTPR@1%FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        base = result["metrics"][model].get("clean")
        for variant in probes:
            cur = result["metrics"][model].get(variant)
            if not base or not cur:
                continue
            lines.append(
                f"| {model} | {PRETTY.get(variant, variant)} | "
                f"{base['auc'] - cur['auc']:+.3f} | {base['ap'] - cur['ap']:+.3f} | "
                f"{base['hit@100'] - cur['hit@100']:+.3f} | {base['tpr@fpr1%'] - cur['tpr@fpr1%']:+.3f} |"
            )

    lines += [
        "",
        "## 2. Is it a shift, a compression, or a reordering?",
        "",
        "`Δpos` / `Δneg` = mean score *drop* of each class in clean-std units (positive = lost score);",
        "`tau` = Kendall rank correlation with the clean scores;",
        "`top100` = overlap of the clean and probe top-100; `inversion` = of the pairs the clean model orders",
        "correctly, the fraction the probe reverses.",
        "",
        "| model | probe | d′ | Δpos | Δneg | tau | top100 | inversion |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for variant in ["clean"] + probes:
            d = result["decomposition"][model].get(variant)
            m = result["metrics"][model].get(variant)
            if not d or not m:
                continue
            lines.append(
                f"| {model} | {PRETTY.get(variant, variant)} | {m['dprime']:.2f} | "
                f"{d.get('shift_pos', float('nan')):+.2f} | {d.get('shift_neg', float('nan')):+.2f} | "
                f"{d['tau']:.3f} | {d['top100_retention']:.2f} | {d['inversion']:.2f} |"
            )

    lines += ["", "## 3. Where the damage lives (AUC inside strata of the scored graph)", ""]
    for model, per_variant in result["strata"].items():
        for variant, rows in per_variant.items():
            names = list(rows)
            if not names:
                continue
            lines.append(f"**{model} · {PRETTY.get(variant, variant)}**")
            lines.append("")
            lines.append("| stratum | clean AUC | probe AUC | damage | Δ score pos | Δ score neg | pairs (pos+neg) |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for name in names:
                row = rows[name]
                has_auc = "clean" in row and np.isfinite(row.get("clean", np.nan))
                auc_cells = (
                    f"{row['clean']:.3f} | {row['probe']:.3f} | {row['damage']:+.3f}"
                    if has_auc
                    else "- | - | -"
                )
                lines.append(
                    f"| {name} | {auc_cells} | {row.get('d_pos', float('nan')):+.2f} | "
                    f"{row.get('d_neg', float('nan')):+.2f} | {int(row['n_pos'])}+{int(row['n_neg'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=["cora"])
    parser.add_argument("--scores", default="results/scores")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    for dataset in args.dataset:
        result = analyse(dataset, Path(args.scores))
        result["markdown"] = to_markdown(result)
        stem = f"{args.output}/operating_{dataset}"
        save_json(result, f"{stem}.json")
        Path(f"{stem}.md").write_text(result["markdown"] + "\n", encoding="utf-8")
        print(result["markdown"][:1200])
        print(f"\nwrote {stem}.json and .md\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
