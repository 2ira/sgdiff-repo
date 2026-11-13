#!/usr/bin/env python
"""Collect every ``results/*.json`` into one markdown table.

    python scripts/collect_results.py                 # print
    python scripts/collect_results.py --write         # update results/RESULTS.md
    python scripts/collect_results.py --merge full    # rebuild the mean/std summary

``run_experiments.py`` writes ``summary_indomain_<profile>.json`` once per
invocation, so running the datasets in several batches (cora, then usair/ns,
then citeseer/pubmed) overwrites the earlier aggregates.  ``--merge`` rebuilds
the file from the per-run ``<dataset>_seed<k>_<profile>.json`` files instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.version import METHOD_VERSION  # noqa: E402

RESULTS = Path("results") / METHOD_VERSION

# Runs that are not in-domain table rows.
SKIP_PREFIXES = ("summary_", "lowdata_", "attack_", "transfer_", "ablation_", "baselines_")


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _aggregate(runs: list[dict]) -> dict:
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out: dict = {"num_seeds": len(runs)}
    for key in keys:
        values = [float(r[key]) for r in runs]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        out[key] = {"mean": mean, "std": std, "values": [round(v, 6) for v in values]}
    return out


def merge_indomain(profile: str) -> dict:
    """Rebuild ``summary_indomain_<profile>.json`` from the per-seed run files."""
    grouped: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob(f"*_seed*_{profile}.json")):
        if path.name.startswith(SKIP_PREFIXES):
            continue
        payload = json.loads(path.read_text())
        if "auc" not in payload:
            continue
        if payload.get("method_version") != METHOD_VERSION:
            raise ValueError(f"cannot merge legacy/mixed method results: {path}")
        dataset = path.stem.rsplit("_seed", 1)[0]
        grouped.setdefault(dataset, []).append(payload)

    if not grouped:
        raise ValueError(f"no corrected-method {profile} runs under {RESULTS}")
    summary = {dataset: {"method_version": METHOD_VERSION, **_aggregate(runs)}
               for dataset, runs in sorted(grouped.items())}
    out = RESULTS / f"summary_indomain_{profile}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    rows = [f"| {d} | {m['num_seeds']} | {m['auc']['mean']:.4f} ± {m['auc']['std']:.4f} "
            f"| {m['ap']['mean']:.4f} | {m['hit@100']['mean']:.4f} |"
            for d, m in summary.items()]
    print(f"wrote {out} ({len(summary)} datasets)")
    print("| dataset | seeds | AUC | AP | Hit@100 |")
    print("|---|---:|---:|---:|---:|")
    print("\n".join(rows))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--merge", metavar="PROFILE", default=None,
                        help="rebuild results/summary_indomain_<PROFILE>.json from per-seed files")
    args = parser.parse_args()

    if args.merge:
        merge_indomain(args.merge)
        return 0

    rows: list[str] = ["| experiment | AUC | AP | Hit@100 |", "|---|---:|---:|---:|"]
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        payload = json.loads(path.read_text())
        if "auc" in payload:
            rows.append(
                f"| {path.stem} | {_fmt(payload.get('auc'))} | {_fmt(payload.get('ap'))} "
                f"| {_fmt(payload.get('hit@100'))} |"
            )
        else:  # nested dict: transfer / attack
            for key, metrics in payload.items():
                if isinstance(metrics, dict) and "auc" in metrics:
                    rows.append(
                        f"| {path.stem}[{key}] | {_fmt(metrics['auc'])} | {_fmt(metrics['ap'])} "
                        f"| {_fmt(metrics.get('hit@100'))} |"
                    )

    for path in sorted(RESULTS.glob("summary_*.json")):
        payload = json.loads(path.read_text())
        rows.append(f"\n### {path.stem} (mean over seeds)\n")
        rows.append("| dataset | AUC | AP | Hit@100 |")
        rows.append("|---|---:|---:|---:|")
        for dataset, metrics in payload.items():
            if isinstance(metrics, dict) and "auc" in metrics and isinstance(metrics["auc"], dict):
                rows.append(
                    f"| {dataset} | {metrics['auc']['mean']:.4f} ± {metrics['auc']['std']:.4f} "
                    f"| {metrics['ap']['mean']:.4f} | {metrics['hit@100']['mean']:.4f} |"
                )

    table = "\n".join(rows)
    if args.write:
        (RESULTS / "RESULTS.md").write_text("# Measured results\n\n" + table + "\n")
        print(f"wrote {RESULTS / 'RESULTS.md'}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
