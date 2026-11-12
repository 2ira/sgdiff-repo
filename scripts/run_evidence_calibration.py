#!/usr/bin/env python
"""Can the *scoring rule* be fixed without retraining? Null-model calibration.

§7.6 of ``docs/attribution.md`` leaves one lever untested: every model turns
observations into a score with a rule that does not know what "no evidence" looks
like, so when the graph is thinned the score simply moves. A monotone rescaling
cannot fix that -- AUC is rank-invariant -- but a **non-monotone** correction can:
subtract the structural prior the score already contains, estimated from a
degree-preserving null model.

Two corrections, both fitted on the **clean** graph only and then applied with the
features of whatever graph is scored (a deployed model cannot see the clean graph
either):

``residual``
    ridge-regress the clean scores on per-pair structure
    ``[cn, log(1+null_cn), d_u, d_v, d_u d_v]`` (``null_cn = d_u d_v / 2m`` is the
    configuration-model expectation), then score with ``s - λ·ŝ``.
``nullsub``
    the cheaper parametric version, ``s - λ·z(null_cn)`` (no fitting).

λ = 0 reproduces the baseline, λ = 1 is the full correction. Everything is
measured on the same per-pair dumps as ``run_operating_point.py``.

    python scripts/run_evidence_calibration.py --dataset cora citeseer

Output: ``results/calibration_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402

PRETTY = {"clean": "clean", "del@50%": "del", "del_hub@50%": "del_hub", "ins_hub@50%": "ins_hub", "feature_shuffle": "feat_shuffle"}
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def feature_matrix(feat: dict, stats: dict) -> np.ndarray:
    """Standardise with *clean* statistics, so the transform is fixed at deploy time."""
    cols = [
        (feat["cn"] - stats["cn"][0]) / stats["cn"][1],
        (np.log1p(feat["null_cn"]) - stats["lcn"][0]) / stats["lcn"][1],
        (feat["deg_u"] - stats["du"][0]) / stats["du"][1],
        (feat["deg_v"] - stats["dv"][0]) / stats["dv"][1],
        (feat["deg_u"] * feat["deg_v"] - stats["prod"][0]) / stats["prod"][1],
    ]
    x = np.stack(cols, axis=1)
    return np.concatenate([np.ones((len(x), 1)), x], axis=1)


def feat_of(payload: dict, variant: str) -> dict:
    return {k.split("__")[-1]: v.astype(np.float64) for k, v in payload.items() if k.startswith(f"feat__{variant}__")}


def clean_stats(feat: dict) -> dict:
    lcn = np.log1p(feat["null_cn"])
    prod = feat["deg_u"] * feat["deg_v"]
    sd = lambda a: float(a.std() or 1.0)  # noqa: E731
    return {
        "cn": (float(feat["cn"].mean()), sd(feat["cn"])),
        "lcn": (float(lcn.mean()), sd(lcn)),
        "du": (float(feat["deg_u"].mean()), sd(feat["deg_u"])),
        "dv": (float(feat["deg_v"].mean()), sd(feat["deg_v"])),
        "prod": (float(prod.mean()), sd(prod)),
    }


def ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n, d = x.shape
    reg = alpha * np.eye(d)
    reg[0, 0] = 0.0  # do not penalise the intercept
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def analyse(dataset: str, scores_dir: Path) -> dict:
    files = sorted(scores_dir.glob(f"{dataset}_s*.npz"))
    if not files:
        raise SystemExit(f"no dumps for {dataset}; run scripts/run_score_dump.py first")
    dumps = [dict(np.load(f)) for f in files]
    models = sorted({k.split("__")[1] for k in dumps[0] if k.startswith("score__")})
    variants = [v for v in PRETTY if any(f"score__{models[0]}__{v}" in d for d in dumps)]

    out: dict[str, dict] = {}
    for model in models:
        per_model: dict[str, object] = {"fit": {}, "residual": {}, "nullsub": {}}
        r2_runs = []
        for payload in dumps:
            labels = payload["labels"]
            clean_scores = payload[f"score__{model}__clean"].astype(np.float64)
            stats = clean_stats(feat_of(payload, "clean"))
            x_clean = feature_matrix(feat_of(payload, "clean"), stats)
            w = ridge(x_clean, clean_scores)
            s_hat_clean = x_clean @ w
            resid = clean_scores - s_hat_clean
            ss_res = float((resid**2).sum())
            ss_tot = float(((clean_scores - clean_scores.mean()) ** 2).sum() or 1.0)
            r2_runs.append(1.0 - ss_res / ss_tot)

            for variant in variants:
                key = f"score__{model}__{variant}"
                if key not in payload:
                    continue
                s = payload[key].astype(np.float64)
                feat = feat_of(payload, variant)
                x = feature_matrix(feat, stats)
                s_hat = x @ w
                lcn = np.log1p(feat["null_cn"])
                z_lcn = (lcn - stats["lcn"][0]) / stats["lcn"][1]
                for lam in LAMBDAS:
                    for family, corrected in (
                        ("residual", s - lam * s_hat),
                        ("nullsub", s - lam * z_lcn),
                    ):
                        auc = float(summarize_scores(labels, corrected)["auc"])
                        per_model[family].setdefault(variant, {}).setdefault(lam, []).append(auc)
        per_model["fit"]["r2"] = float(np.mean(r2_runs))
        for family in ("residual", "nullsub"):
            per_model[family] = {
                variant: {lam: float(np.mean(vals)) for lam, vals in per_lam.items()}
                for variant, per_lam in per_model[family].items()
            }
        out[model] = per_model
    return {"dataset": dataset, "files": [f.name for f in files], "models": models, "variants": variants, "results": out}


def to_markdown(result: dict) -> str:
    models, variants = result["models"], result["variants"]
    probes = [v for v in variants if v != "clean"]
    lines = [f"# Null-model calibration: {result['dataset']}", ""]
    lines += [
        "The correction is fitted on the clean graph and applied with the scored graph's own features.",
        "Damage = clean(λ=0) − probe(λ), so the clean row also reports the accuracy the correction costs.",
        "",
        "## 1. How much of the score is structural prior?",
        "",
        "| model | R² of the clean-score ridge on [CN, E_null, degrees] |",
        "|---|---:|",
    ]
    for model in models:
        lines.append(f"| {model} | {result['results'][model]['fit']['r2']:.3f} |")

    lines += [
        "",
        "## 2. Full correction (λ = 1) vs baseline",
        "",
        "| model | clean AUC (base → corrected) | probe | damage (base → corrected) |",
        "|---|---|---|---:|",
    ]
    for model in models:
        res = result["results"][model]
        for family in ("residual", "nullsub"):
            for variant in probes:
                if variant not in res[family]:
                    continue
                base_clean = res[family]["clean"][0.0]
                corr_clean = res[family]["clean"][1.0]
                base_probe = res[family][variant][0.0]
                corr_probe = res[family][variant][1.0]
                lines.append(
                    f"| {model} | {base_clean:.3f} → {corr_clean:.3f} | {PRETTY.get(variant, variant)} | "
                    f"{base_clean - base_probe:+.3f} → {corr_clean - corr_probe:+.3f} |"
                )

    lines += [
        "",
        "## 3. Does the correction have a working budget? (residual family, `del`)",
        "",
        "| model | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        res = result["results"][model]["residual"].get("del@50%")
        if not res:
            continue
        cells = " | ".join(f"{res[lam]:.3f}" for lam in LAMBDAS)
        lines.append(f"| {model} | {cells} |")

    lines += [
        "",
        "## 4. All probes, all λ (residual family: AUC after correction)",
        "",
    ]
    for model in models:
        res = result["results"][model]["residual"]
        lines.append(f"**{model}**")
        lines.append("")
        lines.append("| probe | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for variant in ["clean"] + probes:
            if variant not in res:
                continue
            cells = " | ".join(f"{res[variant][lam]:.3f}" for lam in LAMBDAS)
            lines.append(f"| {PRETTY.get(variant, variant)} | {cells} |")
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
        stem = f"{args.output}/calibration_{dataset}"
        save_json(result, f"{stem}.json")
        Path(f"{stem}.md").write_text(result["markdown"] + "\n", encoding="utf-8")
        print(f"wrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
