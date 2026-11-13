#!/usr/bin/env python
"""Bootstrap confidence intervals for every probe damage, with paired resampling.

The tables in ``docs/attribution.md`` report means over seeds and compare them to
"the seed spread". That is the wrong uncertainty: the number of seeds is 2-8,
while the number of scored pairs is 1000. The quantity the claims are about —
"this probe costs this model that much AUC" — is estimated from the pair sample,
so its interval comes from resampling **pairs**, and because the clean and the
probe scores are computed on the *same* pairs the resampling must be **paired**.

For each (model, probe) this reports:

* ``ΔAUC`` with a 95 % paired-bootstrap interval,
* ``p``: the fraction of resamples in which the damage is ≤ 0, i.e. a one-sided
  bootstrap p-value for "the probe does not hurt".

    python scripts/run_bootstrap_ci.py --dataset cora citeseer usair

Output: ``results/significance_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sgdiff.utils.io import save_json  # noqa: E402

PRETTY = {"del@50%": "del", "del_hub@50%": "del_hub", "ins_hub@50%": "ins_hub", "feature_shuffle": "feat_shuffle"}
REPS = 2000


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (ties get average ranks)."""
    n_pos = float((labels == 1).sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def analyse(dataset: str, scores_dir: Path, reps: int = REPS, seed: int = 0) -> dict:
    files = sorted(scores_dir.glob(f"{dataset}_s*.npz"))
    if not files:
        raise SystemExit(f"no dumps for {dataset}; run scripts/run_score_dump.py first")
    dumps = [dict(np.load(f)) for f in files]
    models = sorted({k.split("__")[1] for k in dumps[0] if k.startswith("score__")})
    variants = [v for v in PRETTY if any(f"score__{models[0]}__{v}" in d for d in dumps)]
    rng = np.random.default_rng(seed)

    out: dict[str, dict] = {}
    for model in models:
        out[model] = {}
        for variant in variants:
            dmg_samples: list[np.ndarray] = []
            auc_point, clean_point, damage_point = [], [], []
            for payload in dumps:
                key = f"score__{model}__{variant}"
                if key not in payload:
                    continue
                labels = payload["labels"]
                clean = payload[f"score__{model}__clean"].astype(np.float64)
                probe = payload[key].astype(np.float64)
                n = len(labels)
                auc_c, auc_p = fast_auc(labels, clean), fast_auc(labels, probe)
                clean_point.append(auc_c)
                auc_point.append(auc_p)
                damage_point.append(auc_c - auc_p)
                idx = rng.integers(0, n, size=(reps, n))
                dmg_samples.append(
                    np.array(
                        [fast_auc(labels[b], clean[b]) - fast_auc(labels[b], probe[b]) for b in idx]
                    )
                )
            if not damage_point:
                continue
            samples = np.concatenate(dmg_samples)
            out[model][variant] = {
                "auc": float(np.mean(auc_point)),
                "clean_auc": float(np.mean(clean_point)),
                "damage": float(np.mean(damage_point)),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_no_damage": float((samples <= 0).mean()),
            }
    return {
        "dataset": dataset,
        "files": [f.name for f in files],
        "seeds": len(files),
        "pairs": int(len(dumps[0]["labels"])),
        "reps": reps,
        "results": out,
    }


def to_markdown(result: dict) -> str:
    lines = [f"# Paired bootstrap: {result['dataset']}", ""]
    lines += [
        f"{result['reps']} resamples of the {result.get('pairs', 1000)} scored pairs per seed, pooled over",
        f"{result.get('seeds', len(result['files']))} seeds; clean and probe are resampled together.",
        "`p` is the one-sided bootstrap probability that the damage is ≤ 0.",
        "",
        "| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |",
        "|---|---|---:|---:|---|---:|",
    ]
    rows = []
    for model, per_variant in result["results"].items():
        for variant, cell in per_variant.items():
            if variant == "clean":
                continue
            rows.append((cell["damage"], model, variant, cell))
    for _dmg, model, variant, cell in sorted(rows, key=lambda r: -abs(r[0])):
        lines.append(
            f"| {model} | {PRETTY.get(variant, variant)} | {cell['clean_auc']:.3f} | {cell['auc']:.3f} | "
            f"{cell['damage']:+.3f} [{cell['ci_low']:+.3f}, {cell['ci_high']:+.3f}] | {cell['p_no_damage']:.4f} |"
        )
    lines += [
        "",
        "### What this does and does not cover",
        "",
        "* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).",
        f"* It does **not** cover training-seed uncertainty; the dumps are {result.get('seeds', len(result['files']))}",
        "  seeds, so a claim about the population of trained models still rests on the seed spreads in",
        "  the other tables. Pooling seeds widens the interval when they disagree; it does not turn",
        "  a pair-level claim into a model-population claim.",
        "* Effects whose interval includes 0 should not be reported as an effect, however",
        "  consistent their sign looks across the eight seeds of §7.6.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=["cora"])
    parser.add_argument("--scores", default="results/scores")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    for dataset in args.dataset:
        result = analyse(dataset, Path(args.scores), reps=args.reps)
        result["markdown"] = to_markdown(result)
        stem = f"{args.output}/significance_{dataset}"
        save_json(result, f"{stem}.json")
        Path(f"{stem}.md").write_text(result["markdown"] + "\n", encoding="utf-8")
        print(f"wrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
