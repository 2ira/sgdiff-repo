"""Render the complete measured Markdown report from saved CSV/JSON artifacts.

This entry point needs only the standard library; it never trains models or
changes measured artifacts. Run with --output pointing at the archive or run root.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = {'ncnc': 'NCNC', 'lpformer': 'LPFormer', 'refined_gae': 'Refined-GAE'}
DATASETS = {'cora': 'Cora', 'citeseer': 'Citeseer'}
CONDITIONS = [('clean', 0.)] + [
    (attack, ratio) for attack in ['rf', 'dice', 'ea'] for ratio in [.1, .25, .5]
] + [('delete', .5), ('delete_hub', .5), ('insert', .5), ('feature_shuffle', 0.)]
METRICS = ['auc', 'ap', 'hits100', 'precision100', 'drop_auc', 'drop_ap']


def read_csv(root, name):
    with (root / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def condition(row):
    attack = row['attack']
    label = {'clean': '干净图', 'delete': '随机删边', 'delete_hub': '按度加权删边',
             'insert': '随机插边', 'feature_shuffle': '特征打乱'}.get(attack, attack.upper())
    return label if attack in ['clean', 'feature_shuffle'] else f"{label} {100 * float(row['ratio']):g}%"


def group_key(row):
    return row['dataset'], row['model'], row['component'], row['attack'], float(row['ratio'])


def validate_rows(summary, per_seed):
    """Refuse reports that hide missing conditions, duplicate seeds or stale means."""
    groups = defaultdict(list)
    seen = set()
    for row in per_seed:
        key = group_key(row)
        seeded_key = key + (int(row['seed']),)
        if seeded_key in seen:
            raise ValueError(f'Duplicate seed evaluation: {seeded_key}')
        seen.add(seeded_key)
        groups[key].append(row)
    if len({group_key(row) for row in summary}) != len(summary):
        raise ValueError('Duplicate aggregate evaluation')
    if set(groups) != {group_key(row) for row in summary}:
        raise ValueError('Summary and seed-level evaluation coverage differs')
    configs = defaultdict(set)
    for row in per_seed:
        configs[(row['dataset'], row['model'], row['component'], int(row['seed']))].add(
            (row['attack'], float(row['ratio'])))
    for key, conditions in configs.items():
        if conditions != set(CONDITIONS):
            raise ValueError(f'Incomplete 14-condition evaluation: {key}')
    for row in summary:
        members = groups[group_key(row)]
        if len(members) != int(row['n_seeds']):
            raise ValueError(f'Seed count differs: {group_key(row)}')
        for metric in METRICS:
            values = [float(r[metric]) for r in members]
            for suffix, actual in [('mean', statistics.mean(values)),
                                   ('std', statistics.stdev(values) if len(values) > 1 else 0.)]:
                if not math.isclose(actual, float(row[f'{metric}_{suffix}']), abs_tol=1e-12):
                    raise ValueError(f'Stale aggregate: {group_key(row)}:{metric}_{suffix}')
    return len(configs)


def cell(row, metric='auc'):
    return f"{100 * float(row[metric + '_mean']):.2f} ± {100 * float(row[metric + '_std']):.2f}"


def table(lines, headings, rows):
    lines.extend(['| ' + ' | '.join(headings) + ' |',
                  '| ' + ' | '.join(['---'] * len(headings)) + ' |'])
    for row in rows:
        lines.append('| ' + ' | '.join(str(value).replace('|', '\\|') for value in row) + ' |')
    lines.append('')


def render_report(root):
    root = Path(root)
    summary, seeds = read_csv(root, 'summary.csv'), read_csv(root, 'per_seed.csv')
    n_configs = validate_rows(summary, seeds)
    lookup = {group_key(row): row for row in summary}
    datasets = [d for d in DATASETS if any(r['dataset'] == d for r in summary)]
    models = [m for m in MODELS if any(r['model'] == m for r in summary)]
    seed_ids = sorted({int(r['seed']) for r in seeds})
    lines = ['# NCNC、LPFormer、Refined-GAE：完整复现、鲁棒性与组件消融实验报告', '',
             f'本报告收录 **{n_configs} 组训练、{len(seeds)} 次评估、{len(summary)} 行跨种子汇总**。'
             f'随机种子：{", ".join(map(str, seed_ids))}。所有表格由已保存的 CSV/JSON 自动生成，'
             '未用新训练替代已有结果。指标和汇总在生成前逐项核对。', '',
             '结果位于 `feat/ncnc-lpformer-refined-gae` 分支。仓库历史 SGDiff 报告此前未接入这组独立实验；'
             '旧版三模型报告只展示部分汇总，把删边放在英文“mechanism probes”一节，'
             '且遗漏了 10% 攻击及消融模型的完整扰动表。本版将这些结果集中呈现。', '',
             '## 阅读索引', '',
             '- [实验范围与协议](#实验范围与协议)',
             '- [完整模型：干净图、删边与插边概览](#完整模型干净图删边与插边概览)',
             '- [完整模型：全部攻击与指标](#完整模型全部攻击与指标)',
             '- [组件消融：全部条件与指标](#组件消融全部条件与指标)',
             '- [配对组件差值](#配对组件差值)',
             '- [实际编辑预算](#实际编辑预算)',
             '- [训练记录](#训练记录)',
             '- [逐种子完整结果](#逐种子完整结果)',
             '- [结论、审计与尚未运行的设置](#结论审计与尚未运行的设置)', '',
             '## 实验范围与协议', '',
             '本轮是统一协议下的作者架构复现，调用固定版本的作者模型；'
             '不代表逐项复现三个原论文的基准表格，也不能与仓库历史 SGDiff 表格直接排名。'
             '详细参数、源码版本及协议差异见 [实验说明](../README.md)、'
             '[模型与消融说明](../MODEL_GUIDE.zh-CN.md) 和 [运行配置](run_manifest.json)。', '',
             '实验矩阵为 Cora/Citeseer × 3 模型 × 每模型 3 个组件版本 × 3 种子；'
             '每组训练对应下列 14 个条件。所有组件变体独立重训，随后冻结参数评估。', '']
    table(lines, ['条件', '已运行比例', '操作与含义'], [
        ['干净图', '0%', '对照'], ['RF', '10% / 25% / 50%', '随机删除与插入，合计为编辑预算'],
        ['DICE', '10% / 25% / 50%', '删除同类边、插入异类边，类别只供攻击器使用'],
        ['EA', '10% / 25% / 50%', '线性代理迁移攻击，单次梯度排序，非目标模型白盒攻击'],
        ['随机删边', '50%', '全部预算用于均匀随机删除现有边'],
        ['按度加权删边', '50%', '无放回采样，概率正比于端点的干净度数之和'],
        ['随机插边', '50%', '全部预算用于插入非边'],
        ['特征打乱', '一次共享置换', '打乱特征行与节点的对应关系，保持边与特征集合']])
    lines += [
        '- **数据与划分**：Planetoid 原始属性，正边按 80/5/15 划分；训练、验证、测试使用固定、'
        '平衡且互斥的负边，负边不在完整干净图中。',
        '- **训练**：共同正样本 batch size 为 512；每批目标正边从消息传播图和局部邻域移除。'
        'LPFormer 训练沿用作者的完整训练图 PPR 缓存，仍可能包含训练目标信息；验证/测试目标不进入该缓存。',
        '- **选点**：每 5 epoch 检查干净验证 AUC，最多 100 epoch，耐心 30 epoch；'
        '验证上下文只有训练正边，测试上下文是训练加验证正边。',
        '- **攻击方式**：干净图训练后冻结权重，仅在推理时扰动图，属于 evasion；'
        '不是在扰动图上重训的投毒实验。所有模型与组件共享同一数据集、种子下的攻击图。',
        '- **预算与保护**：预算为 `round(比例 × 可见推理图无向边数)`；保护全部测试正负候选点对及验证负边。'
        'RF/DICE 将约一半预算用于删除，其余用于插入，因此“50% 编辑”不等于“删边50%”。',
        '- **LPFormer**：每张不同的受扰图重新计算 PPR；去位置编码消融仍保留 PPR 候选筛选。',
        '- **指标**：AUC、AP、Hits@100、Precision@100 均以百分数展示；跨种子为均值 ± 样本标准差。'
        'Hits@100 使用全局负样本池第 100 高分作为严格阈值，Precision@100 单独计算并平均处理边界并列。',
        '- **降幅**：同一种子的干净指标减受扰指标，再计算均值和标准差，单位为百分点；'
        '负降幅表示受扰指标提高。不能用两个独立标准差相减得到降幅标准差。', '']
    lines += ['## 完整模型：干净图、删边与插边概览', '',
              '以下为完整模型 AUC；全部 AP、Hits@100、Precision@100 和降幅紧接在下一节。', '']
    table(lines, ['数据集', '模型', '干净图', '随机删边50%', '按度加权删边50%', '随机插边50%', '特征打乱'],
          [[DATASETS[ds], MODELS[model]] +
           [cell(lookup[(ds, model, 'full', attack, ratio)]) for attack, ratio in
            [('clean', 0.), ('delete', .5), ('delete_hub', .5), ('insert', .5), ('feature_shuffle', 0.)]]
           for ds in datasets for model in models])
    headings = ['条件', '种子数', 'AUC', 'AP', 'Hits@100', 'Precision@100', 'AUC降幅', 'AP降幅']
    def metric_rows(ds, model, component):
        for attack, ratio in CONDITIONS:
            row = lookup.get((ds, model, component, attack, ratio))
            if row is not None:
                yield [condition(row), row['n_seeds']] + [cell(row, m) for m in METRICS]
    lines += ['## 完整模型：全部攻击与指标', '',
              '每个模型均完整列出 14 个条件，包含 RF/DICE/EA 的 10%、25%、50% 三档预算。', '']
    for ds in datasets:
        for model in models:
            lines += [f'### {DATASETS[ds]} / {MODELS[model]} / 完整模型', '']
            table(lines, headings, metric_rows(ds, model, 'full'))
    lines += ['## 组件消融：全部条件与指标', '',
              '每个变体从头独立训练，训练后采用与完整模型相同的 14 条件评估。'
              '完整模型见上一节；本节列出全部消融版本，不将特征打乱称为结构-only 重训。', '']
    table(lines, ['模型', '变体', '具体改变'], [
        ['NCNC', 'no_completion', '关闭补全，作者 depth-0 NCN'],
        ['NCNC', 'no_cn', '作者 CN0 端点预测器；仍保留 GCN(A,X)，不是特征-only'],
        ['LPFormer', 'uniform_attention', '冻结为均匀注意力，保留投影和候选'],
        ['LPFormer', 'no_ppr_encoding', '位置编码置零，保留 PPR 候选选择和计数'],
        ['Refined-GAE', 'no_norm', '编码器和解码器 LayerNorm 改为 Identity，保留 dropout'],
        ['Refined-GAE', 'dot_decoder', '保留编码器，改为作者点积预测器']])
    for ds in datasets:
        for model in models:
            components = sorted({r['component'] for r in summary
                                 if r['dataset'] == ds and r['model'] == model and r['component'] != 'full'})
            for component in components:
                lines += [f'### {DATASETS[ds]} / {MODELS[model]} / {component}', '']
                table(lines, headings, metric_rows(ds, model, component))
    paired_path = root / 'paired_components_summary.csv'
    lines += ['## 配对组件差值', '',
              '先在同一种子内相减，再汇总。`完整−变体` 为正表示完整模型指标更高；'
              '`变体降幅−完整降幅` 为正表示变体在该扰动下下降更多。两者不能混为同一结论。', '']
    if paired_path.exists():
        paired = read_csv(root, paired_path.name)
        for ds in datasets:
            for model in models:
                lines += [f'### {DATASETS[ds]} / {MODELS[model]} 配对消融', '']
                table(lines, ['变体', '条件', '种子数', '完整−变体 AUC', '完整−变体 AP',
                              '完整−变体 Hits@100', '变体−完整 AUC降幅', '变体−完整 AP降幅'],
                      ([r['component'], condition(r), r['n_seeds']] + [cell(r, m) for m in [
                          'full_minus_variant_auc', 'full_minus_variant_ap', 'full_minus_variant_hits100',
                          'variant_minus_full_drop_auc', 'variant_minus_full_drop_ap']]
                       for r in paired if r['dataset'] == ds and r['model'] == model))
    else:
        lines += ['本目录没有配对汇总文件；未填入推算值。', '']
    lines += ['## 实际编辑预算', '',
              '下面逐数据集、逐种子列出审计记录中的实际删除/插入数量；'
              '这些图在所有模型与组件间共享。特征打乱复用干净图的边，不额外占用一张攻击图。', '']
    protocol_path = root / 'protocol_audit.json'
    if protocol_path.exists():
        audit = json.loads(protocol_path.read_text())
        table(lines, ['攻击缓存文件', '条件', '原可见边数', '要求预算', '实际删除', '实际插入', '实际编辑'],
              ([r['file'], condition(r), r['base_edges'], r['requested_budget'], r['deleted'],
                r['inserted'], r['effective_edits']] for r in audit['attacks']))
        lines += ['预算采用 Python `round`；例如 Citeseer 的 3869 条可见边在 50% 预算下得到 1934 次编辑。'
                  'EA 的删除/插入构成由梯度排序决定，本轮实际记录均为插边；不能将 EA 50% 解释为删边50%。', '']
    else:
        lines += ['本目录未保存独立协议审计；请运行 audit 命令补充。', '']
    lines += ['## 训练记录', '',
              '验证 AUC 用于选点；时间不含图/PPR 预处理。显存为单进程 PyTorch 最大已分配量，'
              '不代表整卡占用；参数数包含可能未在当前前向路径使用的注册参数。', '']
    if (root / 'training_summary.csv').exists():
        table(lines, ['数据集', '模型', '组件', '种子', '最佳/停止epoch', '验证AUC', '训练可见边最小/最大',
                      '总/可训练参数', '训练秒数', '峰值分配GiB'],
              ([DATASETS[r['dataset']], MODELS[r['model']], r['component'], r['seed'],
                f"{r['best_epoch']}/{r['stopped_epoch']}", f"{100 * float(r['best_val_auc']):.2f}",
                f"{r['min_visible_training_edges']}/{r['max_visible_training_edges']}",
                f"{r['total_parameters']}/{r['trainable_parameters']}",
                f"{float(r['training_seconds_excluding_ppr']):.2f}",
                f"{float(r['peak_torch_allocated_bytes']) / 2**30:.3f}"]
               for r in read_csv(root, 'training_summary.csv')))
    else:
        lines += ['本目录尚无逐训练配置的训练汇总。', '']
    lines += ['## 逐种子完整结果', '',
              f'以下直接展开全部 **{len(seeds)} 行评估**，不是只提供 CSV 下载。'
              '单次指标保留 4 位小数（百分数）；原始精度见 [per_seed.csv](per_seed.csv)。', '']
    for ds in datasets:
        for model in models:
            lines += [f'### {DATASETS[ds]} / {MODELS[model]} 逐种子结果', '']
            table(lines, ['组件', '种子', '条件', 'AUC', 'AP', 'Hits@100', 'Precision@100', 'AUC降幅', 'AP降幅'],
                  ([r['component'], r['seed'], condition(r)] + [f'{100 * float(r[m]):.4f}' for m in METRICS]
                   for r in seeds if r['dataset'] == ds and r['model'] == model))
    lines += ['## 结论、审计与尚未运行的设置', '',
              '- **结构删边**：本轮 NCNC 在两种删边下保留较高 AUC；LPFormer 在 Cora 上的删边降幅更大。'
              '这是当前数据、预算和训练配方下的观察，不能推广到所有攻击。',
              '- **NCNC 组件**：补全的边际提升较小，当前结果不足以将其鲁棒性归因于补全；'
              '`no_cn` 仍保留结构编码器。',
              '- **LPFormer 组件**：当前设置下去位置编码的平均影响通常大于均匀注意力；'
              '候选较少且种子波动较大，需结合配对差值判断。',
              '- **Refined-GAE 组件**：解码器与归一化的效果依赖扰动类型，'
              '干净图性能更高不保证特征打乱后更高。',
              '- **统计范围**：仅 3 个种子，均值与标准差不等于显著性检验；'
              '攻击后绝对指标与从干净图的降幅应同时阅读。', '',
              '### 完成情况与复核', '',
              f'- 本次报告生成重新核对 {len(summary)} 行汇总与 {len(seeds)} 行逐种子数据，'
              '逐一检查 14 条件覆盖、重复记录、种子数及六项指标的均值/标准差。']
    for name, description in [
        ('audit.json', '756 次预测指标重算及完整网格检查'),
        ('protocol_audit.json', '共享图预算、候选保护、负样本互斥、验证选点和源码指纹审计'),
        ('replay_audit.json', '54 个检查点独立重载，clean/DICE50/特征打乱共 162 次复核'),
        ('mechanism_audit.json', '组件实现与候选一致性检查'),
        ('ppr_reference_check.json', 'PPR 与固定版本作者参考的一致性'),
        ('verification.json', '已运行测试及其源码指纹')]:
        if (root / name).exists():
            lines.append(f'- [{name}]({name})：{description}。')
    lines += ['', '### 尚未运行，不能当作已完成结果', '',
              '- 纯随机删边、按度删边和纯插边的 10% / 25% 档；本轮这三类只运行了 50%。',
              '- 在扰动图上重新训练的投毒实验。',
              '- 针对三个目标模型分别优化的白盒攻击，以及作者/DeepRobust EA 的逐项对齐。',
              '- 这三个新模型的跨数据集迁移；54 组矩阵为域内实验，早期 GCN 迁移实验属于另一组结果。', '',
              '### 配套机制记录与原始表', '',
              '下列附加机制分析保留独立原始表，避免把抽样注意力或相关性误写成因果归因。'
              '主实验全部条件、组件、指标和种子已在上文展开。', '',
              '- [中文机制解读](FINDINGS.zh-CN.md)：配对差值及注意力、深度权重的解释。',
              '- [分层诊断说明](diagnostics.md)、[分层汇总](strata_summary.csv)、'
              '[分层逐种子数据](strata_per_seed.csv)、[分数相关性](score_correlations.csv)。',
              '- [模型内部分析说明](introspection.md)、[学习参数](learned_parameters.csv)、'
              '[逐点对注意力样本](attention_per_pair.csv)。',
              '- [配对逐种子差值](paired_components.csv)、[配对汇总](paired_components_summary.csv)。',
              '- [训练汇总](training_summary.csv)、[数据统计](dataset_statistics.json)、'
              '[环境与硬件](runtime.json)、[协议修订](protocol_revision.json)。', '',
              '### 鲁棒性曲线', '',
              '曲线仅展示完整模型 RF/DICE/EA；纯删边、插边及所有组件的完整数值以本文表格为准。', '',
              '![完整模型鲁棒性 AUC，三种子均值与标准差](robustness_auc.png)', '',
              '### 再生成本报告', '', '```bash',
              'python -m experiments.new_link_models.reporting \\',
              '  --output experiments/new_link_models/benchmarks', '```', '',
              '该命令只读取已保存的 CSV/JSON，生成 Markdown；不启动训练、不改写实验指标。', '']
    return '\n'.join(lines)


def write_report(root):
    root = Path(root)
    (root / 'report.md').write_text(render_report(root), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_report(args.output)
    print(f'wrote {args.output / "report.md"}')


if __name__ == '__main__':
    main()
