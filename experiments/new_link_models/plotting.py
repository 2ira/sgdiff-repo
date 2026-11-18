"""Create integrated PNG/PDF report figures from measured tables, without training."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from .reporting import CONDITIONS, DATASETS, MODELS, group_key, read_csv, validate_rows


FIGURES = ['performance_overview', 'attack_curves', 'probe_damage', 'component_effects']
COLORS = {'ncnc': '#0072B2', 'lpformer': '#D55E00', 'refined_gae': '#009E73'}
LABELS = ['Clean', 'RF\n10%', 'RF\n25%', 'RF\n50%', 'DICE\n10%', 'DICE\n25%',
          'DICE\n50%', 'EA\n10%', 'EA\n25%', 'EA\n50%', 'Delete\n50%',
          'Hub-del\n50%', 'Insert\n50%', 'Feature\nshuffle']
VARIANTS = [('ncnc', 'no_completion'), ('ncnc', 'no_cn'),
            ('lpformer', 'uniform_attention'), ('lpformer', 'no_ppr_encoding'),
            ('refined_gae', 'no_norm'), ('refined_gae', 'dot_decoder')]
SHORT_VARIANTS = ['NCNC / no completion', 'NCNC / no CN', 'LPFormer / uniform attn.',
                  'LPFormer / no PPR encoding', 'Refined-GAE / no norm', 'Refined-GAE / dot decoder']


def figure_points(root):
    """Validate every source value and export every plotted mean/SD in raw units."""
    root = Path(root)
    summary, seeds = read_csv(root, 'summary.csv'), read_csv(root, 'per_seed.csv')
    validate_rows(summary, seeds)
    seeded = {(r['dataset'], r['model'], r['component'], int(r['seed']), r['attack'], float(r['ratio'])): r
              for r in seeds}
    pairs = read_csv(root, 'paired_components_summary.csv')
    expected = {group_key(r) for r in summary if r['component'] != 'full'}
    if len(pairs) != len(expected) or {group_key(r) for r in pairs} != expected:
        raise ValueError('Incomplete or duplicate paired component summary')
    for r in pairs:
        ds, model, component, attack, ratio = group_key(r)
        members = [s for s in seeds if group_key(s) == group_key(r)]
        values = [float(seeded[(ds, model, 'full', int(s['seed']), attack, ratio)]['auc'])
                  - float(s['auc']) for s in members]
        for suffix, value in [('mean', statistics.mean(values)),
                              ('std', statistics.stdev(values) if len(values) > 1 else 0.)]:
            if not math.isclose(value, float(r[f'full_minus_variant_auc_{suffix}']), abs_tol=1e-12):
                raise ValueError(f'Stale paired component AUC: {group_key(r)}:{suffix}')
        if len(members) != int(r['n_seeds']):
            raise ValueError(f'Paired seed count differs: {group_key(r)}')
    points = []
    def add(figure, row, metric, panel):
        points.append(dict(figure=figure, **{k: row[k] for k in
                      ['dataset', 'model', 'component', 'attack', 'ratio']}, panel=panel,
                      metric=metric, mean=float(row[metric + '_mean']), std=float(row[metric + '_std'])))
    for r in summary:
        if r['component'] != 'full':
            continue
        for metric in ['auc', 'ap']:
            add('performance_overview', r, metric, metric)
            for attack in ['rf', 'dice', 'ea']:
                if r['attack'] in ['clean', attack]:
                    add('attack_curves', r, metric, f'{attack}_{metric}')
        if r['attack'] in ['delete', 'delete_hub', 'insert', 'feature_shuffle']:
            add('probe_damage', r, 'drop_auc', 'feature' if r['attack'] == 'feature_shuffle' else 'structural')
    for r in pairs:
        add('component_effects', r, 'full_minus_variant_auc',
            'feature' if r['attack'] == 'feature_shuffle' else 'structural')
    return points


def plot_figures(root):
    # Optional plotting dependencies are not imported by the Markdown generator.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    root = Path(root)
    points = figure_points(root)
    index = {(r['figure'], r['dataset'], r['model'], r['component'], r['attack'],
              float(r['ratio']), r['metric'], r['panel']): r for r in points}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

    def point(figure, ds, model, attack, ratio, metric, panel, component='full'):
        return index[(figure, ds, model, component, attack, ratio, metric, panel)]

    def save(fig, name):
        fig.savefig(root / (name + '.png'), dpi=180, bbox_inches='tight')
        fig.savefig(root / (name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})
        plt.close(fig)

    # All 14 conditions and both primary metrics, with one shared absolute scale.
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), layout='constrained')
    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(['auc', 'ap']):
            ax = axes[2 * i + j]
            values = np.array([[100 * point('performance_overview', ds, model, attack, ratio,
                                           metric, metric)['mean'] for attack, ratio in CONDITIONS]
                               for model in MODELS])
            image = ax.imshow(values, vmin=60, vmax=100, cmap='viridis', aspect='auto')
            for y, x in np.ndindex(values.shape):
                ax.text(x, y, f'{values[y, x]:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if values[y, x] < 82 else '#17212b')
            ax.set(title=f'{DATASETS[ds]} | {metric.upper()} (%)',
                   xticks=range(14), xticklabels=LABELS,
                   yticks=range(3), yticklabels=list(MODELS.values()))
            for boundary in [.5, 3.5, 6.5, 9.5, 12.5]:
                ax.axvline(boundary, color='white', linewidth=1.8)
    fig.colorbar(image, ax=axes, label='Mean performance (%) | higher is better', shrink=.85, pad=.015)
    fig.suptitle('Full models: all 14 conditions | three-seed means', fontsize=15)
    save(fig, 'performance_overview')

    # AUC and AP dose-response curves; absolute values plus sample-SD bars.
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), layout='constrained')
    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(['auc', 'ap']):
            rows_for_scale = [r for r in points if r['figure'] == 'attack_curves'
                              and r['dataset'] == ds and r['metric'] == metric]
            lower = math.floor(min(100 * (r['mean'] - r['std']) for r in rows_for_scale)) - .5
            upper = math.ceil(max(100 * (r['mean'] + r['std']) for r in rows_for_scale)) + .5
            for col, attack in enumerate(['rf', 'dice', 'ea']):
                ax = axes[2 * i + j, col]
                for model in MODELS:
                    selected = [point('attack_curves', ds, model, a, ratio, metric, f'{attack}_{metric}')
                                for a, ratio in [('clean', 0.), (attack, .1), (attack, .25), (attack, .5)]]
                    ax.errorbar([0, 10, 25, 50], [100 * r['mean'] for r in selected],
                                yerr=[100 * r['std'] for r in selected], color=COLORS[model],
                                marker='o', markersize=4, capsize=3, linewidth=1.6, label=MODELS[model])
                ax.set(title=f'{DATASETS[ds]} | {metric.upper()} | {attack.upper()}',
                       xlabel='Edit budget (% of visible edges)', ylabel=f'{metric.upper()} (%)',
                       xticks=[0, 10, 25, 50], ylim=(lower, upper))
                ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('RF / DICE / EA: mean +/- sample SD across three seeds', fontsize=15)
    save(fig, 'attack_curves')

    # Keep attribute damage on a separate axis so it cannot hide deletion effects.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained',
                             gridspec_kw={'width_ratios': [2.8, 1.2]})
    for i, ds in enumerate(DATASETS):
        for col, (attacks, panel) in enumerate([(['delete', 'delete_hub', 'insert'], 'structural'),
                                               (['feature_shuffle'], 'feature')]):
            ax = axes[i, col]
            for j, model in enumerate(MODELS):
                selected = [point('probe_damage', ds, model, a, 0. if a == 'feature_shuffle' else .5,
                                  'drop_auc', panel) for a in attacks]
                ax.errorbar(np.arange(len(attacks)) + (j - 1) * .15,
                            [100 * r['mean'] for r in selected], yerr=[100 * r['std'] for r in selected],
                            color=COLORS[model], marker='o', linestyle='none', capsize=4,
                            markersize=6, label=MODELS[model])
            ax.axhline(0, color='#64748b', linewidth=1)
            ax.set(title=f'{DATASETS[ds]} | ' + ('structure' if col == 0 else 'features'),
                   xticks=range(len(attacks)),
                   xticklabels=['Random delete\n50%', 'Degree-weighted delete\n50%', 'Random insert\n50%']
                   if col == 0 else ['Feature shuffle'], ylabel='Paired AUC drop (percentage points)',
                   xlim=(-.45, len(attacks) - .55))
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Controlled perturbations: clean minus perturbed AUC | mean +/- sample SD', fontsize=14)
    save(fig, 'probe_damage')

    # All component variants x all conditions. Separate color scales prevent the
    # large feature-shuffle effects from washing out structural differences.
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), layout='constrained',
                             gridspec_kw={'width_ratios': [13, 1.25]})
    structural = [r for r in points if r['figure'] == 'component_effects' and r['panel'] == 'structural']
    feature = [r for r in points if r['figure'] == 'component_effects' and r['panel'] == 'feature']
    limits = [max(1., math.ceil(max(abs(100 * r['mean']) for r in structural))),
              max(1., math.ceil(max(abs(100 * r['mean']) for r in feature)))]
    images = []
    for i, ds in enumerate(DATASETS):
        for col, (conditions, panel) in enumerate([(CONDITIONS[:-1], 'structural'),
                                                   ([CONDITIONS[-1]], 'feature')]):
            values = np.array([[100 * point('component_effects', ds, model, attack, ratio,
                                           'full_minus_variant_auc', panel, component)['mean']
                                for attack, ratio in conditions] for model, component in VARIANTS])
            ax = axes[i, col]
            image = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(0., -limits[col], limits[col]), aspect='auto')
            for y, x in np.ndindex(values.shape):
                label = '0.0' if abs(values[y, x]) < .05 else f'{values[y, x]:+.1f}'
                ax.text(x, y, label, ha='center', va='center', fontsize=9,
                        color='white' if abs(values[y, x]) > .6 * limits[col] else '#17212b')
            ax.set(title=f'{DATASETS[ds]} | ' + ('clean + structural conditions' if col == 0 else 'features'),
                   xticks=range(len(conditions)), xticklabels=LABELS[:-1] if col == 0 else [LABELS[-1]],
                   yticks=range(6), yticklabels=SHORT_VARIANTS if col == 0 else [])
            for boundary in [1.5, 3.5]:
                ax.axhline(boundary, color='white', linewidth=2)
            if col == 0:
                for boundary in [.5, 3.5, 6.5, 9.5]:
                    ax.axvline(boundary, color='white', linewidth=1)
            if i == 0:
                images.append(image)
    for col in range(2):
        fig.colorbar(images[col], ax=axes[:, col], orientation='horizontal', fraction=.055, pad=.04,
                     label='AUC delta (pp)' if col else 'Full minus variant AUC (pp)')
    fig.suptitle('Retrained component effects: paired means | blue = full model higher, red = variant higher', fontsize=14)
    save(fig, 'component_effects')

    with (root / 'figure_points.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, points[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(points)
    def sha(name):
        return hashlib.sha256((root / name).read_bytes()).hexdigest()
    audit = dict(source='Saved measured tables; no training or metric changes',
                 units='Raw 0-1 metric units in figure_points.csv; plots multiply by 100',
                 uncertainty='Sample SD across seeds; drops and component deltas paired within seed',
                 records={name: sum(r['figure'] == name for r in points) for name in FIGURES},
                 source_sha256={name: sha(name) for name in
                                ['summary.csv', 'per_seed.csv', 'paired_components_summary.csv']},
                 output_sha256={name: sha(name) for name in ['figure_points.csv'] +
                                [f'{name}.{ext}' for name in FIGURES for ext in ['png', 'pdf']]},
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'figure_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot_figures(args.output), indent=2))


if __name__ == '__main__':
    main()
