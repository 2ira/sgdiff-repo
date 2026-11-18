"""Recompute every metric, audit paired attacks, and summarize actual completed runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .adapters import VARIANTS
from .protocol import metrics


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
        writer = csv.DictWriter(f, records[0].keys()); writer.writeheader(); writer.writerows(records)
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
        writer = csv.DictWriter(f, summary[0].keys()); writer.writeheader(); writer.writerows(summary)
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
            writer = csv.DictWriter(f, paired[0].keys()); writer.writeheader(); writer.writerows(paired)
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
            writer = csv.DictWriter(f, paired_summary[0].keys()); writer.writeheader(); writer.writerows(paired_summary)
    lookup = {(r['dataset'], r['model'], r['component'], r['attack'], r['ratio']): r for r in summary}
    datasets = sorted({r['dataset'] for r in records})
    models = sorted({r['model'] for r in records})
    complete_runs = len(epochs)
    report = ['# NCNC / LPFormer / Refined-GAE: measured reproduction and robustness', '',
              f'Completed {complete_runs} configurations and {len(records)} evaluations. '
              'Each configuration is one (dataset, model, component, seed).', '',
              'Author model classes are called from pinned external repositories. '
              'These are author-architecture reproductions under a shared protocol, not claims of reproducing original paper tables.', '',
              'Data: canonical Planetoid Cora/Citeseer, raw attributes; 80/5/15 positive split, balanced disjoint fixed negatives. '
              'All models use positive minibatch size 512, retaining a nonempty visible graph after target masking. '
              'Clean validation selects checkpoints. Test context is train+validation. '
              'Attacks edit this visible graph after protecting all test candidates. '
              'EA is the documented linear-surrogate transfer attack, not DeepRobust parity.', '',
              'AUC/AP are standard ranking metrics. Hits@100 compares positives strictly against the 100th-highest global negative. '
              'Precision@100 is a separate metric, with order-independent cutoff ties. '
              'Values below are percentages, mean ± sample standard deviation.', '', '## Clean full-model results', '',
              '| Dataset | Model | Seeds | AUC | AP | Hits@100 |', '|---|---|---|---|---|---|']
    def cell(row, metric='auc'):
        if row is None:
            return 'N/A'
        return f"{100 * row[metric + '_mean']:.2f} ± {100 * row[metric + '_std']:.2f}"
    for ds in datasets:
        for model in models:
            row = lookup.get((ds, model, 'full', 'clean', 0.))
            if row:
                report.append(f"| {ds} | {model} | {row['n_seeds']} | {cell(row)} | {cell(row, 'ap')} | {cell(row, 'hits100')} |")
    report += ['', '## Full-model structural evasion (AUC)', '',
               '| Dataset | Model | Clean | RF 25% | DICE 25% | EA 25% | RF 50% | DICE 50% | EA 50% |',
               '|---|---|---|---|---|---|---|---|---|']
    for ds in datasets:
        for model in models:
            vals = [lookup.get((ds, model, 'full', a, r)) for a, r in [('clean', 0.), ('rf', .25), ('dice', .25), ('ea', .25), ('rf', .5), ('dice', .5), ('ea', .5)]]
            if any(v is not None for v in vals):
                report.append('| ' + ' | '.join([ds, model] + [cell(v) for v in vals]) + ' |')
    report += ['', '## Retrained component ablations (AUC)', '',
               '| Dataset | Model | Component | Clean | RF 50% | DICE 50% | EA 50% |', '|---|---|---|---|---|---|---|']
    for ds in datasets:
        for model in models:
            for component in VARIANTS[model]:
                vals = [lookup.get((ds, model, component, a, r)) for a, r in [('clean', 0.), ('rf', .5), ('dice', .5), ('ea', .5)]]
                if vals[0] is not None:
                    report.append('| ' + ' | '.join([ds, model, component] + [cell(v) for v in vals]) + ' |')
    report += ['', '## Frozen-model mechanism probes (AUC)', '',
               '| Dataset | Model | Clean | Random deletion 50% | Hub deletion 50% | Insertion 50% | Feature shuffle |',
               '|---|---|---|---|---|---|---|']
    for ds in datasets:
        for model in models:
            vals = [lookup.get((ds, model, 'full', a, r)) for a, r in [('clean', 0.), ('delete', .5), ('delete_hub', .5), ('insert', .5), ('feature_shuffle', 0.)]]
            if vals[0] is not None:
                report.append('| ' + ' | '.join([ds, model] + [cell(v) for v in vals]) + ' |')
    report += ['', '## Interpretation limits', '',
               '- Frozen probes measure sensitivity; component rows are independent retraining experiments.',
               '- No prespecified robustness ranking is assumed. Read attacked absolute performance together with clean-to-attack drops.',
               '- Three seeds support an initial comparison, not a definitive significance or broad robustness claim.',
               '- Shared train/validation/test masks and negatives differ from the authors’ original benchmark protocols. Hyperparameters and every departure are saved in config.json.',
               '- LPFormer PPR is regenerated on every distinct attacked graph; the PE ablation retains the same neighborhood selection.',
               '- No-norm Refined-GAE retains dropout, avoiding the author encoder’s norm/dropout switch coupling.', '']
    if paired:
        report += ['Paired seed-level full-minus-variant performance and variant-minus-full degradation '
                   'are in `paired_components.csv` and `paired_components_summary.csv`. Positive performance '
                   'deltas favor the full model; positive degradation deltas mean the variant loses more.', '']
    (root / 'report.md').write_text('\n'.join(report))
    audit = dict(completed_training_runs=complete_runs, evaluations=len(records),
                 prediction_metrics_recomputed=True, attacked_graphs_identical_across_models=True,
                 expected_training_runs=54, expected_evaluations=54 * 14,
                 complete=complete_runs == 54 and len(records) == 54 * 14)
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
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
                            yerr=[float(r['auc_std']) * 100 for r in selected], marker='o', capsize=3, label=model)
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
