"""Audit and publish the complete main-protocol supplement, including PubMed."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from sgdiff.utils.metrics import summarize_scores
from .adapters import VARIANTS
from .protocol import digest, metrics as strict_metrics
from .reporting import cell, condition, table
from .repository_protocol import MAIN_COMMIT, PROTOCOL, load_inputs, plan, tag, edgeset, pinned_sources


DATASETS = {'cora': 'Cora', 'citeseer': 'Citeseer', 'pubmed': 'PubMed'}
MODELS = {'ncnc': 'NCNC', 'lpformer': 'LPFormer', 'refined_gae': 'Refined-GAE'}
METRICS = ['auc', 'ap', 'hit@100', 'strict_hits100', 'precision100_tie_average', 'drop_auc', 'drop_ap']
CONDITIONS = [('clean', 0.)] + [(a, r) for a in ['rf', 'dice', 'ea'] for r in [.1, .25, .5]]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = dict(zip(keys, key), n_seeds=len(group))
        for name in values:
            result[name + '_mean'] = float(np.mean([r[name] for r in group]))
            result[name + '_std'] = float(np.std([r[name] for r in group], ddof=1))
        output.append(result)
    return output


def collect(root, dest):
    root, dest = Path(root), Path(dest)
    files = sorted((root / 'runs').glob('*/complete.json'))
    expected = {(ds, model, component, seed) for ds in DATASETS for model in VARIANTS
                for component in VARIANTS[model] for seed in range(3)}
    actual, records, configs, trainings = set(), [], [], []
    manifest = json.loads((root / 'input_manifest.json').read_text())
    assert manifest['main_source_sha256'] == pinned_sources()
    input_graphs = {(r['dataset'], r['seed'], r['attack'], r['ratio']): r for r in manifest['attacks']}
    from .run import source_hashes
    author_runtime = source_hashes()
    for file in files:
        result = json.loads(file.read_text())
        cfg = result['config']
        key = tuple(cfg[k] for k in ['dataset', 'model', 'component', 'seed'])
        assert key not in actual
        actual.add(key)
        assert cfg['protocol'] == PROTOCOL and cfg['main_commit'] == MAIN_COMMIT
        assert cfg['main_source_sha256'] == manifest['main_source_sha256']
        assert cfg['author_runtime_sha256'] == author_runtime
        assert cfg['supplement_sha256'] == hashlib.sha256(Path(__file__).with_name('repository_protocol.py').read_bytes()).hexdigest()
        assert cfg['pubmed_adapter_sha256'] == hashlib.sha256(Path(__file__).with_name('repository_pubmed.py').read_bytes()).hexdigest()
        graph, split, pos, neg = load_inputs(root, cfg['dataset'], cfg['seed'])
        assert cfg['features_sha256'] == digest(graph.x)
        assert cfg['scored_pos_sha256'] == digest(pos) and cfg['scored_neg_sha256'] == digest(neg)
        assert len(pos) == len(neg) == 500
        for name, sha in cfg['split_sha256'].items():
            assert sha == digest(getattr(split, name))
        training = result['training']
        best = max(training['history'], key=lambda r: r['auc'])
        assert best['auc'] == training['best_val_auc'] and best['epoch'] == training['best_epoch']
        assert training['validation_context'] == 'train+val, matching main train_node_pair_baseline'
        assert {(r['attack'], r['ratio']) for r in result['evaluations']} == set(plan(graph.num_nodes))
        configs.append(cfg)
        trainings.append(dict(zip(['dataset', 'model', 'component', 'seed'], key),
                        best_epoch=training['best_epoch'], stopped_epoch=training['stopped_epoch'],
                        best_val_auc=training['best_val_auc'], seconds=training['seconds'],
                        peak_cuda_bytes=result['peak_cuda_bytes'],
                        checkpoint_sha256=hashlib.sha256((file.parent / 'model.pt').read_bytes()).hexdigest()))
        for evaluation in result['evaluations']:
            attack, ratio = evaluation['attack'], evaluation['ratio']
            expected_graph = input_graphs[(cfg['dataset'], cfg['seed'], attack, ratio)]
            assert evaluation['statistics']['graph_sha256'] == expected_graph['graph_sha256']
            with np.load(file.parent / f'predictions_{tag(attack, ratio)}.npz', allow_pickle=False) as f:
                p, n = f['positive'], f['negative']
                assert p.shape == n.shape == (500,)
                recomputed = summarize_scores(np.r_[np.ones(500), np.zeros(500)], np.r_[p, n])
                extra = strict_metrics(p, n)
                recomputed.update(strict_hits100=extra['hits100'], precision100_tie_average=extra['precision100'])
            for name, value in recomputed.items():
                assert np.isclose(value, evaluation['metrics'][name], atol=1e-12, rtol=0)
            for name in ['auc', 'ap']:
                expected_drop = result['evaluations'][0]['metrics'][name] - evaluation['metrics'][name]
                assert np.isclose(evaluation['drop'][name], expected_drop, atol=1e-12, rtol=0)
            records.append(dict(zip(['dataset', 'model', 'component', 'seed'], key), attack=attack,
                                ratio=ratio, **evaluation['metrics'],
                                drop_auc=evaluation['drop']['auc'], drop_ap=evaluation['drop']['ap']))
    if actual != expected:
        raise RuntimeError(f'Incomplete supplement: {len(actual)}/81, missing {sorted(expected - actual)[:5]}')
    assert len(records) == 729
    replay = json.loads((root / 'replay_audit.json').read_text())
    assert replay['passed'] and replay['checkpoints'] == 81 and replay['evaluations'] == 162
    graph_audit = []
    for item in manifest['attacks']:
        graph, split, pos, neg = load_inputs(root, item['dataset'], item['seed'])
        with np.load(root / item['file'], allow_pickle=False) as f:
            raw, visible = f['raw_edges'], f['edges']
            assert json.loads(str(f['statistics'])) == {k: v for k, v in item.items() if k != 'file'}
        assert digest(visible) == item['graph_sha256'] and digest(raw) == item['raw_graph_sha256']
        assert edgeset(visible) == edgeset(raw) - edgeset(split.test_pos)
        assert not edgeset(visible) & edgeset(split.test_pos)
        before, after = edgeset(graph.edges), edgeset(raw)
        assert digest(graph.edges) == item['source_graph_sha256']
        assert len(before - after) == item['raw_deleted']
        assert len(after - before) == item['raw_inserted']
        assert item['requested_budget'] == round(item['ratio'] * graph.num_edges)
        assert len(before ^ after) == item['raw_effective_edits']
        if item['attack'] == 'dice':
            assert all(graph.y[u] == graph.y[v] for u, v in before - after)
            assert all(graph.y[u] != graph.y[v] for u, v in after - before)
        if item['attack'] == 'ea':
            assert item['raw_deleted'] == item['requested_budget'] // 2
            assert item['raw_inserted'] == item['requested_budget'] - item['requested_budget'] // 2
        graph_audit.append(item)
    dest.mkdir(parents=True, exist_ok=True)
    summary = aggregate(records, ['dataset', 'model', 'component', 'attack', 'ratio'], METRICS)
    write_csv(dest / 'per_seed.csv', records)
    write_csv(dest / 'summary.csv', summary)
    write_csv(dest / 'training_summary.csv', trainings)
    lookup = {tuple(r[k] for k in ['dataset', 'model', 'component', 'seed', 'attack', 'ratio']): r for r in records}
    paired = []
    for row in records:
        if row['component'] == 'full':
            continue
        full = lookup[(row['dataset'], row['model'], 'full', row['seed'], row['attack'], row['ratio'])]
        paired.append({k: row[k] for k in ['dataset', 'model', 'component', 'seed', 'attack', 'ratio']} |
                      {f'full_minus_variant_{m}': full[m] - row[m] for m in ['auc', 'ap']} |
                      {f'variant_minus_full_drop_{m}': row[f'drop_{m}'] - full[f'drop_{m}'] for m in ['auc', 'ap']})
    paired_summary = aggregate(paired, ['dataset', 'model', 'component', 'attack', 'ratio'],
                               ['full_minus_variant_auc', 'full_minus_variant_ap',
                                'variant_minus_full_drop_auc', 'variant_minus_full_drop_ap'])
    write_csv(dest / 'paired_components.csv', paired)
    write_csv(dest / 'paired_components_summary.csv', paired_summary)
    for folder in ['attacks', 'splits']:
        (dest / folder).mkdir(exist_ok=True)
        for file in (root / folder).glob('*.npz'):
            shutil.copyfile(file, dest / folder / file.name)
    shutil.copyfile(root / 'input_manifest.json', dest / 'input_manifest.json')
    if (root / 'replay_audit.json').exists():
        shutil.copyfile(root / 'replay_audit.json', dest / 'replay_audit.json')
    for name in ['data_manifest.json', 'runtime.json', 'ppr_reuse.json']:
        if (root / name).exists():
            shutil.copyfile(root / name, dest / name)
    (dest / 'run_manifest.json').write_text(json.dumps(configs, indent=2) + '\n')
    audit = dict(passed=True, training_configurations=81, evaluations=729, aggregate_rows=len(summary),
                 paired_rows=len(paired_summary), shared_graphs=len(graph_audit), splits=9,
                 predictions_recomputed=True, validation_selection_checked=True,
                 original_ea_allocation_and_dice_rules_checked=True,
                 main_commit=MAIN_COMMIT, feature_policy='main standardized attributes',
                 graph_provenance=manifest['graph_provenance'], skipped=manifest['skipped'])
    (dest / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    plot(dest, summary, paired_summary)
    report(dest, summary, paired_summary, records, trainings, graph_audit)
    files = sorted(f for f in dest.rglob('*') if f.is_file() and f.name != 'checksums.sha256'
                   and '__pycache__' not in f.parts)
    (dest / 'checksums.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest)}\n' for f in files))
    return audit


def plot(dest, summary, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    labels = ['Clean'] + [f'{a.upper()}\n{100*r:g}%' for a, r in CONDITIONS[1:]]
    lookup = {(r['dataset'], r['model'], r['attack'], r['ratio']): r for r in summary if r['component'] == 'full'}
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(dest / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e2e8f0')
    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(['auc', 'ap']):
            values = np.array([[100 * lookup[(ds, model, a, r)][metric + '_mean']
                                if (ds, model, a, r) in lookup else np.nan for a, r in CONDITIONS] for model in MODELS])
            ax = axes[i, j]
            image = ax.imshow(values, vmin=80, vmax=100, cmap=cmap, aspect='auto')
            for y, x in np.ndindex(values.shape):
                value = values[y, x]
                ax.text(x, y, 'N/A' if np.isnan(value) else f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if not np.isnan(value) and value < 91 else '#17212b')
            ax.set(title=f'{DATASETS[ds]} | {metric.upper()}', xticks=range(10), xticklabels=labels,
                   yticks=range(3), yticklabels=list(MODELS.values()))
    fig.colorbar(image, ax=axes, label='Mean (%) | higher is better', shrink=.8)
    fig.suptitle('Main input/attack protocol | fresh author-model retrains | PubMed EA skipped', fontsize=14)
    save(fig, 'full_models')
    colors = {'ncnc': '#0072B2', 'lpformer': '#D55E00', 'refined_gae': '#009E73'}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    for i, ds in enumerate(DATASETS):
        available = [r for r in summary if r['dataset'] == ds and r['component'] == 'full']
        low = float(np.floor(min(100*(r['auc_mean']-r['auc_std']) for r in available))) - .5
        high = float(np.ceil(max(100*(r['auc_mean']+r['auc_std']) for r in available))) + .5
        for j, attack in enumerate(['rf', 'dice', 'ea']):
            ax = axes[i, j]
            if ds == 'pubmed' and attack == 'ea':
                ax.text(.5, .5, 'Skipped by main\n19717 nodes > 4000', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            for model in MODELS:
                values = [lookup[(ds, model, a, r)] for a, r in [('clean', 0.)] + [(attack, r) for r in [.1, .25, .5]]]
                ax.errorbar([0, 10, 25, 50], [100*r['auc_mean'] for r in values],
                            yerr=[100*r['auc_std'] for r in values], color=colors[model],
                            marker='o', capsize=3, label=MODELS[model])
            ax.set(title=f'{DATASETS[ds]} | {attack.upper()}', xlabel='Budget (% of full graph edges)',
                   ylabel='AUC (%)', xticks=[0, 10, 25, 50], ylim=(low, high))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Attack response | means +/- sample SD across three seeds', fontsize=14)
    save(fig, 'attack_curves')
    variants = [(m, v) for m in VARIANTS for v in VARIANTS[m] if v != 'full']
    pmap = {(r['dataset'], r['model'], r['component'], r['attack'], r['ratio']): r for r in paired}
    maximum = max(1., np.ceil(max(abs(100*r['full_minus_variant_auc_mean']) for r in paired)))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#e2e8f0')
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), layout='constrained')
    for ds, ax in zip(DATASETS, axes):
        values = np.array([[100*pmap[(ds, m, v, a, r)]['full_minus_variant_auc_mean']
                           if (ds, m, v, a, r) in pmap else np.nan for a, r in CONDITIONS] for m, v in variants])
        image = ax.imshow(values, cmap=cmap, aspect='auto', norm=TwoSlopeNorm(0., -maximum, maximum))
        for y, x in np.ndindex(values.shape):
            value = values[y, x]
            label = 'N/A' if np.isnan(value) else '0.0' if abs(value) < .05 else f'{value:+.1f}'
            ax.text(x, y, label, ha='center', va='center', fontsize=9,
                    color='white' if not np.isnan(value) and abs(value) > .6*maximum else '#17212b')
        ax.set(title=DATASETS[ds], xticks=range(10), xticklabels=labels, yticks=range(6),
               yticklabels=[f'{MODELS[m]} / {v}' for m, v in variants])
    fig.colorbar(image, ax=axes, label='Paired full minus variant AUC (pp)', shrink=.8)
    fig.suptitle('Component effects | blue: full model higher; red: variant higher', fontsize=14)
    save(fig, 'component_effects')


def report(dest, summary, paired, records, trainings, graphs):
    lines = ['# main 输入与攻击协议补实验：Cora、Citeseer、PubMed', '',
             '**完成 81 组新训练、729 次评估，三个种子 0/1/2；包含完整模型和全部组件变体。**', '',
             f'固定 main 提交 `{MAIN_COMMIT}`，直接调用原始攻击器，不再使用上一轮的 EA 替代实现。'
             '图为本次新生成并共享的缓存，不保证与历史 H20 环境或合作者未提供的缓存逐边一致。'
             '模型结构和优化器采用固定版本作者配方，并针对 PubMed 使用作者的 PubMed 配置。', '',
             '## 图形总览', '',
             '热图显示三种子的 AUC/AP 均值，灰色 N/A 表示未运行；标准差见表。', '',
             '![完整模型性能](full_models.png)', '',
             '曲线误差棒为样本标准差，预算分母为完整干净图边数；不是纯删边比例。PubMed EA 遵守 main 节点上限。', '',
             '![攻击预算曲线](attack_curves.png)', '',
             '组件图为同种子内“完整−消融”的 AUC 配对差值，蓝色/正值表示完整模型更高；不是显著性检验。', '',
             '![组件消融](component_effects.png)', '',
             '## 实验协议与解释边界', '',
             '- **输入**：使用 main 的数据加载器、80/5/15 划分、互斥平衡负样本及 `node_feature_matrix` 属性标准化。'
             '旧权重使用原始特征，因此本轮全部重新训练，不复用旧权重。',
             '- **攻击**：调用 main `apply_attack` 在完整图生成 RF/DICE/EA，然后移除全部测试正边；'
             '不增加测试负候选保护、不改变原攻击的删插配额，不修复原算法中可能抵消的编辑。',
             '- **测试**：直接调用 main `_cap_pairs`，取每类前 500 个点对；三个模型及所有组件共享同一图和测试点对。',
             '- **验证**：遵循 main 节点对基线的实际代码，在 train+val 上下文中验证，每 10 epoch 检查干净验证 AUC，'
             '耐心 30 epoch。注意该上下文含验证正边，不能称为严格隐藏验证边的协议；上一轮独立实验只用训练边验证。',
             '- **作者训练配方**：保留作者架构、数据集专属学习率/维度/dropout；共同正例 batch512，目标正边从传播图中移除。'
             '最多 100 epoch。LPFormer 仍沿用作者训练图 PPR 缓存，可能包含批次训练目标信息。'
             '因此对齐的是 main 输入、节点对验证上下文、攻击、测试点对和指标；并非强制所有模型使用同一优化器或复现原论文表格。',
             '- **PPR**：每张不同的受扰图重算。PubMed 使用作者 `eps=1e-5`，Cora/Citeseer 为 `1e-7`；'
             '图指纹一致的已有 PPR 缓存可复用，模型权重不复用。',
             '- **指标**：AUC/AP 使用 main `summarize_scores`。main 的 `hit@100` 实为池化 Precision@100，'
             '并列分数保持输入顺序；本报告沿用其原值用于协议核对，并额外给出严格全局负样本 Hits@100。'
             '三种子均值±样本标准差；降幅单位为百分点。',
             '- **PubMed EA**：19717 个节点超过 main 的 `EA_MAX_NODES=4000`，三种子均跳过。'
             '没有擅自提高限制、改成稀疏攻击或把其他扰动标成 EA。', '',
             '## main 中已有的 PubMed 描述与要求', '',
             '- main README、`docs/reproduction.md` 已有 SGDiff PubMed 的 10 种子域内结果：AUC 约 **0.902±0.002**，'
             'AP 约 0.900；还保存了 PubMed 相关的跨数据集迁移结果。',
             '- main `configs/sgdiff_pubmed.yaml` 配置：1-hop、每跳最多 20 节点、结构扩散 10 步、'
             '特征 GCN 64×2、特征扩散 50 步、训练 50 epoch、预训练 10 epoch。'
             '这些是 SGDiff 专属配置，不应套用到 NCNC/LPFormer/Refined-GAE。',
             '- main 数据说明使用 Planetoid PubMed：19717 节点、44324 条规范化无向边、500 维特征；'
             '与论文边数存在 3 条差异，main 已说明。',
             '- main 另有早期 GCN 消融/迁移实验的 PubMed 记录，属于独立协议，不能当作这三个新模型的结果。',
             '- 本报告 PubMed 与 main 历史 SGDiff 的种子数、测试样本量和模型训练配方不同，'
             '不能只凭两个汇总 AUC 宣称优劣。主分支没有这三个新模型的原生 PubMed 基准要求，作者配方来源见运行配置。', '',
             '## 完整模型概览', '']
    lookup = {(r['dataset'], r['model'], r['component'], r['attack'], r['ratio']): r for r in summary}
    table(lines, ['数据集', '模型', '干净 AUC', 'RF50 AUC', 'DICE50 AUC', 'EA50 AUC'],
          ([DATASETS[ds], MODELS[m]] + [cell(lookup.get((ds, m, 'full', a, r))) if (ds, m, 'full', a, r) in lookup else '跳过：节点限制'
           for a, r in [('clean', 0.), ('rf', .5), ('dice', .5), ('ea', .5)]] for ds in DATASETS for m in MODELS))
    pair_lookup = {(r['dataset'], r['model'], r['component'], r['attack'], r['ratio']): r for r in paired}
    def effect(ds, model, component, attack='clean', ratio=0.):
        return cell(pair_lookup[(ds, model, component, attack, ratio)], 'full_minus_variant_auc')
    lines += ['## 主要观察与组件贡献', '',
              '以下差值均为同一种子内“完整模型−变体”的 AUC，单位为百分点；'
              '变体独立重训，因此测量的是组件与其训练共同产生的效果，不是冻结编码器后只换一个打分函数。', '',
              'NCNC 关闭补全的干净图配对差为 '
              + '；'.join(f'{DATASETS[ds]} {effect(ds, "ncnc", "no_completion")}' for ds in DATASETS)
              + '。本轮补全的边际贡献较小且随设置变化，不能据此将其攻击后表现直接归因于补全。', '',
              'LPFormer 在 PubMed 上，完整−均匀注意力的干净配对差为 '
              f'{effect("pubmed", "lpformer", "uniform_attention")}，'
              '完整−去位置编码为 '
              f'{effect("pubmed", "lpformer", "no_ppr_encoding")}；'
              'DICE50 下完整−去位置编码为 '
              f'{effect("pubmed", "lpformer", "no_ppr_encoding", "dice", .5)}。'
              '这些均值不支持把位置编码或学习注意力的提升视作所有数据集上的普遍规律；保留的 PPR 候选筛选仍含结构信息。', '',
              'Refined-GAE 在 PubMed 上，完整−点积解码器的干净配对差为 '
              f'{effect("pubmed", "refined_gae", "dot_decoder")}，DICE50 下为 '
              f'{effect("pubmed", "refined_gae", "dot_decoder", "dice", .5)}。'
              '本轮结果支持解码器选择在该设置下有较大影响；完整配对标准差与其他数据集见后表。', '',
              '三种子只支持初步比较，误差棒不是置信区间，也未进行显著性检验。'
              '旧协议与本轮同时改变特征、验证上下文、测试点对和攻击，不能把两轮差异仅归因于 EA。', '']
    lines += ['## 全部组件与条件', '',
              '- NCNC：`no_completion` 关闭补全（depth=0）；`no_cn` 使用端点预测器，仍保留 GCN(A,X)，不是特征-only。',
              '- LPFormer：`uniform_attention` 固定均匀注意力；`no_ppr_encoding` 仅将位置编码置零，保留 PPR 候选筛选和计数。',
              '- Refined-GAE：`no_norm` 只移除模型内 LayerNorm，保留 dropout、输入标准化和邻接度归一化；'
              '`dot_decoder` 保留编码器并换为作者点积预测器。PubMed 的完整编码器和变体均为两层传播。', '']
    for ds in DATASETS:
        for model in MODELS:
            for component in VARIANTS[model]:
                lines += [f'### {DATASETS[ds]} / {MODELS[model]} / {component}', '']
                rows = [r for r in summary if r['dataset'] == ds and r['model'] == model and r['component'] == component]
                table(lines, ['条件', 'AUC', 'AP', 'main原hit@100（Precision）', '严格Hits@100', 'AUC降幅', 'AP降幅'],
                      ([condition(r)] + [cell(r, m) for m in ['auc', 'ap', 'hit@100', 'strict_hits100', 'drop_auc', 'drop_ap']] for r in rows))
    lines += ['## 配对消融', '', '先在同种子内相减再汇总；完整−变体为正表示完整模型更高。', '']
    for ds in DATASETS:
        lines += [f'### {DATASETS[ds]} 配对结果', '']
        table(lines, ['模型', '组件', '条件', '完整−变体AUC', '完整−变体AP', '变体−完整AUC降幅'],
              ([MODELS[r['model']], r['component'], condition(r)] + [cell(r, m) for m in
                ['full_minus_variant_auc', 'full_minus_variant_ap', 'variant_minus_full_drop_auc']]
               for r in paired if r['dataset'] == ds))
    lines += ['## 逐种子完整结果', '', '以下展开全部 729 次评估。百分数保留四位；CSV 保留原始精度。', '']
    for ds in DATASETS:
        for model in MODELS:
            lines += [f'### {DATASETS[ds]} / {MODELS[model]} 逐种子结果', '']
            table(lines, ['组件', '种子', '条件', 'AUC', 'AP', 'main原hit@100', '严格Hits@100', 'AUC降幅', 'AP降幅'],
                  ([r['component'], r['seed'], condition(r)] + [f'{100*r[m]:.4f}' for m in
                    ['auc', 'ap', 'hit@100', 'strict_hits100', 'drop_auc', 'drop_ap']]
                   for r in records if r['dataset'] == ds and r['model'] == model))
    lines += ['## 实际攻击预算与测试候选', '',
              '原算法在完整图花费预算后才移除测试边，因此要求预算、完整图实际编辑数、可见图编辑数可不同。'
              '未对 main 算法做事后修正；插入测试负边也会按原协议保留。', '']
    table(lines, ['数据集', '种子', '条件', '要求预算', '完整图删除', '完整图插入', '完整图实际编辑', '可见图实际编辑', '插入测试负边'],
          ([DATASETS[r['dataset']], r['seed'], condition(r)] + [r[k] for k in
            ['requested_budget', 'raw_deleted', 'raw_inserted', 'raw_effective_edits', 'visible_effective_edits', 'inserted_scored_negatives']] for r in graphs))
    lines += ['## 训练与复核', '',
              '训练计时在初始训练/验证上下文准备完成后开始，不含初始图/PPR预处理；'
              '包含后续批次传播图构造与检查点保存。显存是单进程 PyTorch 峰值已分配量，不代表整卡占用。', '']
    table(lines, ['数据集', '模型', '组件', '种子', '最佳epoch', '停止epoch', '验证AUC', '训练秒数', '峰值分配GiB'],
          ([DATASETS[r['dataset']], MODELS[r['model']], r['component'], r['seed'], r['best_epoch'],
            r['stopped_epoch'], f'{100*r["best_val_auc"]:.2f}', f'{r["seconds"]:.2f}', f'{r["peak_cuda_bytes"]/2**30:.3f}'] for r in trainings))
    replay = json.loads((dest / 'replay_audit.json').read_text())
    lines += ['81 个检查点独立重载，每个检查 clean 和 EA50（Cora/Citeseer）或 DICE50（PubMed），'
              '共162次。原始分数使用 `atol=2e-5, rtol=2e-5` 的组合容差；'
              f'最大绝对分数差为 {replay["max_absolute_score_error"]:.3g}，'
              f'最大 AUC/AP 差为 {replay["max_auc_ap_error"]:.3g}（0–1单位），'
              f'main 原 Precision@100 最大差为 {100*replay["max_legacy_precision100_error"]:.3g} 个百分点。', '']
    if replay['max_legacy_precision100_error'] > 0:
        lines += ['观察到 main 旧 Precision@100 的切点排序差异；本报告该列以保存的预测重算值为准，'
                  '不把原始分数在容差内一致描述成所有离散排名逐项一致。', '']
    lines += ['[审计](audit.json)、[输入与图指纹](input_manifest.json)、[全部运行配置](run_manifest.json)、'
              '[逐种子CSV](per_seed.csv)、[汇总CSV](summary.csv)、[配对差值](paired_components_summary.csv)。', '',
              '所有攻击图在 `attacks/`，包含原攻击后的完整图 `raw_edges` 和移除测试正边后的推理图 `edges`；'
              '原始划分与每类500测试点对在 `splits/`，均为本次实际使用的缓存副本。', '',
              '只需评估时直接加载图，不要再次调用攻击器。标准化特征按 main 函数从相同节点顺序的原始属性重建。'
              'PubMed 的 EA 不存在对应缓存。', '',
              '重载检查点复核记录见 [replay_audit.json](replay_audit.json)；模型权重与 PPR 缓存留在外部输出目录，未提交 Git。', '']
    (dest / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.publish), indent=2))


if __name__ == '__main__':
    main()
