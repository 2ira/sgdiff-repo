"""Check report completeness and prevent silently reporting partial/stale results."""
import copy
from pathlib import Path

import pytest

from experiments.new_link_models.reporting import (
    DATASETS, METRICS, MODELS, condition, read_csv, render_report, validate_rows,
)


ARCHIVE = Path(__file__).resolve().parents[1] / 'experiments/new_link_models/benchmarks'


def test_measured_matrix_is_complete_and_aggregates_match_predictions_export():
    assert validate_rows(read_csv(ARCHIVE, 'summary.csv'), read_csv(ARCHIVE, 'per_seed.csv')) == 54


@pytest.mark.parametrize('fault', ['missing_condition', 'duplicate_seed', 'stale_mean', 'stale_std'])
def test_reject_partial_or_stale_report(fault):
    summary = copy.deepcopy(read_csv(ARCHIVE, 'summary.csv'))
    seeds = copy.deepcopy(read_csv(ARCHIVE, 'per_seed.csv'))
    if fault == 'missing_condition':
        target = summary.pop(0)
        seeds = [r for r in seeds if not all(r[k] == target[k] for k in
                 ['dataset', 'model', 'component', 'attack', 'ratio'])]
    elif fault == 'duplicate_seed':
        seeds.append(seeds[0])
    else:
        summary[0]['auc_' + fault.removeprefix('stale_')] = '0.12345'
    with pytest.raises(ValueError):
        validate_rows(summary, seeds)


def test_report_expands_all_756_seed_rows_with_correct_values():
    report = render_report(ARCHIVE)
    section = report.split('## 逐种子完整结果\n', 1)[1].split('## 结论、审计与尚未运行的设置\n', 1)[0]
    rendered = {}
    current = None
    for line in section.splitlines():
        if line.startswith('### '):
            current = next((ds, model) for ds in DATASETS for model in MODELS
                           if line == f'### {DATASETS[ds]} / {MODELS[model]} 逐种子结果')
        elif line.startswith('| ') and not line.startswith(('| 组件 |', '| ---')):
            parts = [v.strip() for v in line.strip('|').split('|')]
            key = current + tuple(parts[:3])
            assert key not in rendered
            rendered[key] = parts[3:]
    expected = {(r['dataset'], r['model'], r['component'], r['seed'], condition(r)):
                [f'{100 * float(r[m]):.4f}' for m in METRICS]
                for r in read_csv(ARCHIVE, 'per_seed.csv')}
    assert len(rendered) == 756
    assert rendered == expected


def test_report_has_every_aggregate_condition_and_component():
    report = render_report(ARCHIVE)
    section = report.split('## 完整模型：全部攻击与指标\n', 1)[1].split('## 配对组件差值\n', 1)[0]
    rendered = []
    for line in section.splitlines():
        if line.startswith('| ') and ' ± ' in line:
            parts = [v.strip() for v in line.strip('|').split('|')]
            rendered.append(parts)
    expected = [[condition(r), r['n_seeds']] + [
        f"{100 * float(r[m + '_mean']):.2f} ± {100 * float(r[m + '_std']):.2f}" for m in METRICS]
        for r in read_csv(ARCHIVE, 'summary.csv')]
    assert len(rendered) == 252
    assert sorted(rendered) == sorted(expected)


def test_plotted_values_match_measured_tables_with_paired_uncertainty():
    from collections import Counter
    from experiments.new_link_models.plotting import figure_points, group_key
    points = figure_points(ARCHIVE)
    assert Counter(r['figure'] for r in points) == {
        'performance_overview': 168, 'attack_curves': 144,
        'probe_damage': 24, 'component_effects': 168}
    sources = {group_key(r): r for r in read_csv(ARCHIVE, 'summary.csv')}
    pairs = {group_key(r): r for r in read_csv(ARCHIVE, 'paired_components_summary.csv')}
    for point in points:
        source = (pairs if point['figure'] == 'component_effects' else sources)[group_key(point)]
        for field in ['mean', 'std']:
            assert point[field] == float(source[point['metric'] + '_' + field])


def test_figure_source_validation_rejects_stale_paired_effects(tmp_path):
    import csv
    import shutil
    from experiments.new_link_models.plotting import figure_points
    for name in ['summary.csv', 'per_seed.csv']:
        shutil.copyfile(ARCHIVE / name, tmp_path / name)
    pairs = read_csv(ARCHIVE, 'paired_components_summary.csv')
    pairs[0]['full_minus_variant_auc_std'] = '0.12345'
    with (tmp_path / 'paired_components_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, pairs[0].keys())
        writer.writeheader()
        writer.writerows(pairs)
    with pytest.raises(ValueError, match='Stale paired component AUC'):
        figure_points(tmp_path)


def test_all_four_figures_precede_the_first_report_table():
    report = render_report(ARCHIVE)
    first_table = report.index('\n| ')
    for name in ['performance_overview', 'attack_curves', 'probe_damage', 'component_effects']:
        assert report.index(f']({name}.png)') < first_table
        assert (ARCHIVE / (name + '.png')).exists()
        assert (ARCHIVE / (name + '.pdf')).exists()
