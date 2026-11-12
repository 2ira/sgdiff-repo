"""Audit saved predictions and generate tables/figures from completed runs only."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import MODES, NAMES, ROOT, metrics

METRICS = ['auc', 'ap', 'hits100']
LABELS = dict(structure='Structure only', feature='Feature only', combined='Structure + feature')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        record = dict(zip(fields, key))
        if len(set(r['seed'] for r in values)) != len(values):
            raise RuntimeError(f'Duplicate seed observations: {key}')
        record['n_seeds'] = len(values)
        for metric in METRICS:
            data = [v[metric] for v in values]
            record[metric + '_mean'] = float(np.mean(data))
            record[metric + '_std'] = float(np.std(data, ddof=1)) if len(data) > 1 else None
        result.append(record)
    return result


def fmt(row, metric):
    if row is None:
        return 'N/A'
    sd = row[metric + '_std']
    return f'{row[metric + "_mean"] * 100:.2f} ± {sd * 100:.2f}' if sd is not None else f'{row[metric + "_mean"] * 100:.2f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    out = args.output
    config = json.loads((out / 'config.json').read_text())
    files = sorted((out / 'runs').glob('*/complete.json'))
    rows = []
    identities = set()
    # Validate data integrity, checkpoint selection, and metrics against saved scores.
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for item in manifest['sources']:
        assert hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest() == item['sha256']
    for file in files:
        records = json.loads(file.read_text())
        train = json.loads((file.parent / 'training.json').read_text())
        assert train['valid']['auc'] == max(item['auc'] for item in train['history'])
        for row in records:
            key = tuple(row[k] for k in ['protocol', 'source', 'target', 'mode', 'seed'])
            assert key not in identities
            identities.add(key)
            assert row['seed'] in config['seeds']
            s = np.load(out / 'splits' / f'{row["target"]}_seed{row["seed"]}.npz')
            prediction = np.load(file.parent / f'predictions_{row["target"]}.npz')
            assert len(prediction['positive']) == len(s['test'])
            assert len(prediction['negative']) == len(s['test_neg'])
            actual = metrics(prediction['positive'], prediction['negative'])
            for m in METRICS:
                assert np.isclose(actual[m], row[m], rtol=0, atol=1e-12)
            rows.append(row)
    expected = set()
    for seed in config['seeds']:
        for protocol in config['protocols']:
            for source in config['datasets']:
                for mode in config['modes']:
                    if mode != 'structure' and source not in NAMES[:3]:
                        continue
                    for target in ([source] if protocol == 'within' else config['datasets']):
                        if mode != 'structure' and target not in NAMES[:3]:
                            continue
                        expected.add((protocol, source, target, mode, seed))
    assert identities <= expected
    missing = expected - identities
    if missing and not args.allow_partial:
        raise RuntimeError(f'Experiment incomplete: {len(identities)}/{len(expected)} scores. Use --allow-partial for a labeled interim report.')
    write_csv(out / 'per_seed.csv', rows)
    summary = summarize(rows, ['protocol', 'source', 'target', 'mode'])
    write_csv(out / 'summary.csv', summary)

    # First average graph pairs within seed, then estimate variation across seeds.
    average_rows = []
    for exclude in [False, True]:
        group = defaultdict(list)
        for row in rows:
            if row['protocol'] != 'transfer' or (exclude and row['source'] == row['target']):
                continue
            for axis, graph in [('all_pairs', 'ALL'), ('source', row['source']), ('target', row['target'])]:
                group[(exclude, axis, graph, row['mode'], row['seed'])].append(row)
        for key, values in group.items():
            exclude, axis, graph, mode, seed = key
            r = dict(exclude_diagonal=exclude, axis=axis, graph=graph, mode=mode, seed=seed)
            r.update({m: float(np.mean([v[m] for v in values])) for m in METRICS})
            average_rows.append(r)
    averages = summarize(average_rows, ['exclude_diagonal', 'axis', 'graph', 'mode'])
    write_csv(out / 'transfer_averages.csv', averages)

    # Paired differences retain the common split/negative-sample pairing.
    lookup_seed = {(r['protocol'], r['source'], r['target'], r['mode'], r['seed']): r for r in rows}
    deltas = []
    for r in rows:
        if r['mode'] != 'combined':
            continue
        for baseline in ['structure', 'feature']:
            ref = lookup_seed.get((r['protocol'], r['source'], r['target'], baseline, r['seed']))
            if ref:
                item = {k: r[k] for k in ['protocol', 'source', 'target', 'seed']}
                item['comparison'] = 'combined_minus_' + baseline
                item.update({m: r[m] - ref[m] for m in METRICS})
                deltas.append(item)
    write_csv(out / 'paired_deltas.csv', summarize(deltas, ['protocol', 'source', 'target', 'comparison']))
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    lookup = {(r['protocol'], r['source'], r['target'], r['mode']): r for r in summary}
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
        for ax, mode in zip(axes, MODES):
            names = [n for n in NAMES if n in config['datasets'] and (mode == 'structure' or n in NAMES[:3])]
            data = np.full((len(names), len(names)), np.nan)
            for i, source in enumerate(names):
                for j, target in enumerate(names):
                    row = lookup.get(('transfer', source, target, mode))
                    if row:
                        data[i, j] = row[metric + '_mean'] * 100
            im = ax.imshow(data, vmin=0, vmax=100, cmap='YlGnBu')
            for i in range(len(names)):
                for j in range(len(names)):
                    if np.isfinite(data[i, j]):
                        ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', fontsize=8,
                                color='white' if data[i, j] > 65 else 'black')
            ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
            ax.set_yticks(range(len(names)), names)
            ax.set(xlabel='Target', ylabel='Source', title=LABELS[mode])
        fig.colorbar(im, ax=axes, label=f'{metric.upper()} (%)', shrink=.75)
        ns = [r['n_seeds'] for r in summary if r['protocol'] == 'transfer']
        seed_label = (f'{min(ns)} seeds per cell' if ns and min(ns) == max(ns)
                      else f'{min(ns)}-{max(ns)} seeds per completed cell' if ns else 'no completed cells')
        fig.suptitle(f'GCN zero-shot transfer: {metric.upper()} ({seed_label})' + (' [PARTIAL]' if missing else ''))
        fig.savefig(figures / f'transfer_{metric}.png', dpi=180)
        plt.close(fig)

    lines = ['# GCN 链路预测消融与跨数据集迁移实验结果', '',
             f'状态：{"完整" if not missing else "未完成 / PARTIAL"}。完成 {len(files)} 个训练单元，{len(rows)}/{len(expected)} 条单种子评估记录；随机种子 {config["seeds"]}。', '',
             '所有数值为百分数，表格给出均值 ± 样本标准差。结构-only 的图内输入为节点ID；迁移输入为训练图度数编码，因此两个协议须分开解读。', '',
             '## 数据核对', '', '| 数据集 | 节点数 | 清洗后的无向边数 | 原始特征维度 |', '|---|---:|---:|---:|']
    for d in manifest['datasets']:
        lines.append(f'| {d["name"]} | {d["nodes"]} | {d["undirected_edges"]} | {d["feature_dim"] or "无"} |')
    lines += ['', '清洗包含无向化、去重、去自环。边划分80%/5%/15%，负边与正边等量；各组共享划分和候选池。', '',
              f'训练：hidden={config["hidden"]}，embedding={config["embedding"]}，dropout={config["dropout"]}，lr={config["lr"]}，weight_decay={config["weight_decay"]}；最多{config["epochs"]}轮，每{config["eval_every"]}轮验证，{config["patience"]}轮无验证AUC改善停止。', '',
              '## 图内三组消融', '']
    for metric in METRICS:
        lines += [f'### {metric.upper()}', '', '| 数据集 | 结构-only | 特征-only | 两者结合 |', '|---|---:|---:|---:|']
        for name in config['datasets']:
            values = [fmt(lookup.get(('within', name, name, mode)), metric) for mode in MODES]
            lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## 跨数据集迁移', '',
              '行是训练源图，列是测试目标图。参数冻结，只在源图验证集选择最优epoch。对角线为相同迁移输入协议的图内对照；跨图平均排除对角线。属性投影只对齐维度，不对齐语义。', '']
    for metric in METRICS:
        lines += [f'### {metric.upper()}', '', f'![Transfer {metric}](figures/transfer_{metric}.png)', '']
        for mode in MODES:
            names = [n for n in config['datasets'] if mode == 'structure' or n in NAMES[:3]]
            lines += [f'**{LABELS[mode]}**', '', '| 源图 → 目标图 | ' + ' | '.join(names) + ' |',
                      '|---|' + '---:|' * len(names)]
            for source in names:
                lines.append('| ' + source + ' | ' + ' | '.join(fmt(lookup.get(('transfer', source, target, mode)), metric) for target in names) + ' |')
            lines.append('')
    lines += ['## 排除对角线后的跨图平均', '',
              '结构-only 跨全部6图；属性组跨3个引文图。候选组合不同，不可直接比较其全体平均。另列共同的三个引文图上的平均以便公平比较。', '',
              '| 范围 | 模型 | AUC | AP | Hits@100 |', '|---|---|---:|---:|---:|']
    for r in averages:
        if r['axis'] == 'all_pairs' and r['exclude_diagonal']:
            lines.append('| 可用全部图 | ' + LABELS[r['mode']] + ' | ' + ' | '.join(fmt(r, m) for m in METRICS) + ' |')
    common = []
    for seed in config['seeds']:
        for mode in config['modes']:
            subset = [r for r in rows if r['protocol'] == 'transfer' and r['mode'] == mode and r['seed'] == seed
                      and r['source'] in NAMES[:3] and r['target'] in NAMES[:3] and r['source'] != r['target']]
            if subset:
                common.append(dict(mode=mode, seed=seed, **{m: float(np.mean([r[m] for r in subset])) for m in METRICS}))
    common_summary = summarize(common, ['mode'])
    write_csv(out / 'transfer_common_citation_average.csv', common_summary)
    for r in common_summary:
        lines.append('| 相同3个引文图 | ' + LABELS[r['mode']] + ' | ' + ' | '.join(fmt(r, m) for m in METRICS) + ' |')
    lines += ['', '## 结果适用范围', '',
              '- Router、USAir、NS 没有原始特征，属性相关实验为 N/A。',
              '- 这是依据论文数据协议独立实现的 GCN 基线与新增消融，不是 SGDiff 模型复现，也不是作者表格数值的精确重现。论文未公开完整代码、精确划分和 GCN 超参数。',
              '- 图内 ID 输入与原始属性模型参数量不同；迁移三组参数量相同。',
              '- 跨图属性语义未对齐；较弱的迁移结果不能证明原始特征没有价值。',
              '- Hits@100 对不同图使用不同规模的全局负边池，难度不同；论文未公开具体候选池细则。本实现使用严格大于第100高负边得分的定义。',
              '- 统一训练配置；初步 Cora seed0 验证检查后采用 GAE 默认的零权重衰减，未做全面超参数优化。详细配置与协议见上级 README。',
              '- 不附加节点分类、扩散模型、对抗攻击、1%训练数据实验。', '',
              '可复查文件：`per_seed.csv`、`summary.csv`、`paired_deltas.csv`、`transfer_averages.csv`、`transfer_common_citation_average.csv`；`runs` 保存检查点/验证曲线/预测，`splits` 保存精确划分。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    audit = dict(complete=not missing, completed_training_runs=len(files),
                 completed_evaluations=len(rows), expected_evaluations=len(expected),
                 raw_hashes_verified=len(manifest['sources']),
                 all_prediction_metrics_recomputed=True, all_checkpoints_selected_on_source_validation=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
