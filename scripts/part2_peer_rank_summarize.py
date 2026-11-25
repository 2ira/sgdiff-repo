"""Aggregate Part 2 node-rank curves with the team's sample-SD conventions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    root = Path(root)
    groups, dgroups = defaultdict(list), defaultdict(list)
    audit_count, max_error = 0, 0.0
    keys = ("dataset", "model", "variant", "condition", "rank", "k")
    paths = sorted(root.glob("*_seed*/per_seed_curves.csv"))
    if len(paths) != 9:
        raise ValueError(f"Expected 9 dataset/seed curve files, got {len(paths)}")
    for path in paths:
        for row in csv.DictReader(path.open(newline="")):
            groups[tuple(row[k] for k in keys)].append(row)
        audit = json.loads(path.with_name("audit.json").read_text())
        audit_count += len(audit["records"])
        max_error = max(max_error, *(row["published_auc_error"] or 0 for row in audit["records"]))
        for row in csv.DictReader(path.with_name("paired_deltas.csv").open(newline="")):
            dgroups[tuple(row[k] for k in keys)].append(row)
    if audit_count != 882:
        raise ValueError(f"Expected 882 original evaluations, got {audit_count}")

    def aggregate(source, metric):
        result = []
        for key, rows in sorted(source.items()):
            if len(rows) != 3 or len({r["seed"] for r in rows}) != 3:
                raise ValueError(f"Incomplete or duplicate seed group: {key}")
            vals = [float(r[metric]) for r in rows if r[metric] != ""]
            result.append(dict(zip(keys, key), n_seeds_present=len(rows), n_seeds_valid=len(vals),
                               mean=float(np.mean(vals)) if vals else None,
                               sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                               min_n_pos=min(int(r["n_pos"]) for r in rows),
                               min_n_neg=min(int(r["n_neg"]) for r in rows),
                               n_low_support=sum(int(r["low_support"]) for r in rows),
                               mean_pos_coverage=float(np.mean([float(r["pos_coverage"]) for r in rows])),
                               mean_neg_coverage=float(np.mean([float(r["neg_coverage"]) for r in rows]))))
        return result

    summary = aggregate(groups, "auc")
    deltas = aggregate(dgroups, "delta_auc")
    assert len(summary) == 3 * 7 * 14 * 2 * 101
    write_csv(root / "summary.csv", summary)
    write_csv(root / "delta_summary.csv", deltas)
    write_csv(root / "key_points.csv", [r for r in summary if int(r["k"]) in (1, 5, 10, 20, 50, 100)])
    (root / "aggregation_audit.json").write_text(json.dumps(dict(
        records_recomputed=audit_count, summary_rows=len(summary),
        max_published_global_auc_error=max_error,
        seed_dispersion="sample standard deviation ddof=1; not a confidence interval"), indent=2) + "\n")
    print(f"Summarized {audit_count} original evaluations into {len(summary)} curve points")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    main(p.parse_args().results)
