"""Recompute every metric, audit paired attacks, and summarize actual completed runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .adapters import VARIANTS
from .protocol import metrics
from .reporting import write_report


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.


def summarize(root):
    records, attacks, source_records, epochs = [], {}, {}, {}
    for file in sorted((root / 'runs').glob('*/complete.json')):
        run = json.loads(file.read_text())
        cfg = run['config']
        source_records[cfg['model']] = cfg['author_sources']
        epochs[file.parent.name] = run['training']
        for row in run['evaluations']:
            name = 'clean' if row['attack'] == 'clean' else row['attack'] if row['attack'] == 'feature_shuffle' else f"{row['attack']}_{row['ratio']:g}"
            with np.load(file.parent / f'predictions_{name}.npz') as f:
                checked = metrics(f['positive'], f['negative'])
            for metric, value in checked.items():
                if not np.isclose(value, row['metrics'][metric], atol=1e-12):
                    raise RuntimeError(f'Saved metric differs from predictions: {file}:{name}:{metric}')
            key = (cfg['dataset'], cfg['seed'], row['attack'], row['ratio'])
            graph_hash = row['statistics']['graph_sha256']
            if key in attacks and attacks[key] != graph_hash:
                raise RuntimeError(f'Models were scored on different attacked graphs: {key}')
            attacks[key] = graph_hash
            records.append({k: row[k] for k in ['dataset', 'model', 'component', 'seed', 'attack', 'ratio']} |
                           {k: checked[k] for k in ['auc', 'ap', 'hits100', 'precision100']} |
                           {f'drop_{k}': row['drop'][k] for k in ['auc', 'ap', 'hits100', 'precision100']})
    if not records:
        raise RuntimeError('No completed experiments')
    with (root / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, records[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in ['dataset', 'model', 'component', 'attack', 'ratio'])].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        result = dict(zip(['dataset', 'model', 'component', 'attack', 'ratio'], key), n_seeds=len(rows))
        for metric in ['auc', 'ap', 'hits100', 'precision100', 'drop_auc', 'drop_ap']:
            result[metric + '_mean'], result[metric + '_std'] = mean_std([r[metric] for r in rows])
        summary.append(result)
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, summary[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(summary)
    by_seed = {tuple(r[k] for k in ['dataset', 'model', 'component', 'seed', 'attack', 'ratio']): r for r in records}
    paired = []
    for row in records:
        if row['component'] == 'full':
            continue
        key = (row['dataset'], row['model'], 'full', row['seed'], row['attack'], row['ratio'])
        if key not in by_seed:
            continue
        full = by_seed[key]
        paired.append({k: row[k] for k in ['dataset', 'model', 'component', 'seed', 'attack', 'ratio']} |
                      {f'full_minus_variant_{m}': full[m] - row[m] for m in ['auc', 'ap', 'hits100']} |
                      {f'variant_minus_full_drop_{m}': row[f'drop_{m}'] - full[f'drop_{m}'] for m in ['auc', 'ap']})
    if paired:
        with (root / 'paired_components.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, paired[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(paired)
        paired_groups = defaultdict(list)
        for row in paired:
            paired_groups[tuple(row[k] for k in ['dataset', 'model', 'component', 'attack', 'ratio'])].append(row)
        paired_summary = []
        for key, rows in sorted(paired_groups.items()):
            entry = dict(zip(['dataset', 'model', 'component', 'attack', 'ratio'], key), n_seeds=len(rows))
            for metric in [k for k in rows[0] if '_minus_' in k]:
                entry[metric + '_mean'], entry[metric + '_std'] = mean_std([r[metric] for r in rows])
            paired_summary.append(entry)
        with (root / 'paired_components_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, paired_summary[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(paired_summary)
    actual_grid = {tuple(r[k] for k in ['dataset', 'model', 'component', 'seed']) for r in records}
    expected_grid = {(ds, model, component, seed) for ds in ['cora', 'citeseer']
                     for model, components in VARIANTS.items() for component in components for seed in range(3)}
    audit = dict(completed_training_runs=complete_runs, evaluations=len(records),
                 prediction_metrics_recomputed=True, attacked_graphs_identical_across_models=True,
                 expected_training_runs=54, expected_evaluations=54 * 14,
                 missing_configurations=sorted(expected_grid - actual_grid),
                 extra_configurations=sorted(actual_grid - expected_grid),
                 complete=actual_grid == expected_grid and complete_runs == 54 and len(records) == 54 * 14)
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    write_report(root)
    return audit


def plot_results(root):
    """Standalone publication-style plots; source tables remain exportable CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (root / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['component'] == 'full']
    datasets = sorted({r['dataset'] for r in rows})
    models = sorted({r['model'] for r in rows})
    fig, axes = plt.subplots(len(datasets), 3, figsize=(11, 3.4 * len(datasets)), squeeze=False, sharey='row')
    for i, ds in enumerate(datasets):
        for j, attack in enumerate(['rf', 'dice', 'ea']):
            ax = axes[i, j]
            for model in models:
                selected = sorted([r for r in rows if r['dataset'] == ds and r['model'] == model and r['attack'] in ['clean', attack]], key=lambda r: float(r['ratio']))
                ax.errorbar([float(r['ratio']) * 100 for r in selected], [float(r['auc_mean']) * 100 for r in selected],
                            yerr=[float(r['auc_std']) * 100 for r in selected], marker='o', capsize=3,
                            label={'ncnc': 'NCNC', 'lpformer': 'LPFormer', 'refined_gae': 'Refined-GAE'}[model])
            ax.set(title=f'{ds.capitalize()} · {attack.upper()}', xlabel='Edited edges (% of visible graph)')
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel('Test AUC (%)')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'robustness_auc.png', dpi=200)
    fig.savefig(root / 'robustness_auc.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.output), indent=2))
    plot_results(args.output)


if __name__ == '__main__':
    main()
