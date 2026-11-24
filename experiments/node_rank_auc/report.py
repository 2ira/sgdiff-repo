"""Report retraining fidelity, all 14 perturbations, paired ablations and node-rank AUROC."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import write_csv

DATASETS=['cora','citeseer','pubmed']
VARIANTS=['full','fixed_weights','shallow_k2']
CONDITIONS=['clean','rf_0.1','rf_0.25','rf_0.5','dice_0.1','dice_0.25','dice_0.5','ea_0.1','ea_0.25','ea_0.5','delete_0.5','delete_hub_0.5','insert_0.5','feature_shuffle']
SELECTED=['clean','ea_0.5','rf_0.5','dice_0.5','delete_0.5','delete_hub_0.5','insert_0.5','feature_shuffle']
COLORS={'full':'#176e86','fixed_weights':'#cf8610','shallow_k2':'#8e54a2'}

def read(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def meanstd(xs):return float(np.mean(xs)),float(np.std(xs,ddof=1)) if len(xs)>1 else None
def fmt(m,s):return f'{m:.4f} ± {s:.4f}' if s is not None else f'{m:.4f}'
def savefig(fig,out,name):
 fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 rows=read(a.evaluation/'per_seed.csv');old=read(a.historical/'per_seed.csv')
 assert len(rows)==378
 key=lambda r:(r['dataset'],int(r['seed']),r['variant'],r['condition'])
 current={key(r):r for r in rows};previous={key(r):r for r in old}
 assert len(current)==len(previous)==378 and current.keys()==previous.keys()
 summaries=[];comparisons=[];paired=[]
 for k,r in current.items():
  prior=previous[k];assert r['graph_sha256']==prior['graph_sha256']
  comparisons.append(dict(dataset=k[0],seed=k[1],variant=k[2],condition=k[3],old_auc=float(prior['auc']),new_auc=float(r['auc']),delta_auc=float(r['auc'])-float(prior['auc'])))
 for d in DATASETS:
  for v in VARIANTS:
   for c in CONDITIONS:
    rr=[current[(d,s,v,c)] for s in range(3)]
    row=dict(dataset=d,variant=v,condition=c,n_seeds=3)
    for metric in ['auc','ap','hits100','precision100']:
     row[metric+'_mean'],row[metric+'_sd']=meanstd([float(r[metric]) for r in rr])
    row['auc_drop_mean'],row['auc_drop_sd']=meanstd([float(current[(d,s,v,'clean')]['auc'])-float(current[(d,s,v,c)]['auc']) for s in range(3)])
    summaries.append(row)
  for v in VARIANTS[1:]:
   for c in CONDITIONS:
    diffs=[100*(float(current[(d,s,'full',c)]['auc'])-float(current[(d,s,v,c)]['auc'])) for s in range(3)]
    mean,sd=meanstd(diffs);paired.append(dict(dataset=d,contrast='full_minus_'+v,condition=c,mean_pp=mean,sd_pp=sd,seed0_pp=diffs[0],seed1_pp=diffs[1],seed2_pp=diffs[2]))
 write_csv(a.output/'robustness_summary.csv',summaries);write_csv(a.output/'historical_comparison.csv',comparisons);write_csv(a.output/'paired_component_deltas.csv',paired)
 checkpoints=[];training=[];hop_rows=[];histories={}
 for p in sorted((a.training/'models').glob('*/complete.json')):
  obj=json.loads(p.read_text());receipt=json.loads(p.with_name('receipt.json').read_text())
  histories[(obj['dataset'],obj['variant'],obj['seed'])]=(json.loads(p.with_name('history.json').read_text()),obj['best_epoch'])
  for hop,(initial,learned) in enumerate(zip(obj['init_temp'],obj['learned_temp'])):
   hop_rows.append(dict(dataset=obj['dataset'],seed=obj['seed'],variant=obj['variant'],hop=hop,initial=initial,learned=learned))
  checkpoints.append(dict(run=obj['run'],checkpoint_sha256=obj['checkpoint_sha256'],nas_path=f'/nas/jiatong/data8002/hlgnn_retrain_20261124/training/models/{obj["run"]}/best.pt'))
  training.append(dict(run=obj['run'],best_epoch=obj['best_epoch'],epochs_run=obj['epochs_run'],updates=obj['updates'],val_auc=obj['best_validation']['auc'],max_epochs=receipt['max_epochs'],patience_validation_checks=receipt['patience']))
 write_csv(a.output/'learned_hop_weights.csv',hop_rows)
 assert len(checkpoints)==27
 (a.output/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n');write_csv(a.output/'training_summary.csv',training)
 lookup={(r['dataset'],r['variant'],r['condition']):r for r in summaries}
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True)
 for i,d in enumerate(DATASETS):
  for j,v in enumerate(VARIANTS):
   ax=axes[i,j]
   for seed,color in zip(range(3),['#176e86','#cf8610','#8e54a2']):
    history,best_epoch=histories[d,v,seed];vv=[r for r in history if 'val_auc' in r]
    ax.plot([r['epoch'] for r in vv],[r['val_auc'] for r in vv],color=color,label=f'seed {seed}',lw=1.2)
    selected=next(r for r in vv if r['epoch']==best_epoch);ax.scatter([best_epoch],[selected['val_auc']],marker='*',s=65,color=color)
   ax.set(title=f'{d} | {v}',xlim=(0,150),ylim=(.85,1),xlabel='Epoch',ylabel='Validation AUROC (zoomed)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('All 27 fits: clean validation curves; stars = selected checkpoints; fixed 150-epoch budget');fig.tight_layout(rect=[0,0,1,.97]);savefig(fig,a.output,'training_validation')
 fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
 for ax,d in zip(axes,DATASETS):
  for v in VARIANTS:
   rr=[lookup[(d,v,c)] for c in CONDITIONS]
   ax.errorbar(range(len(CONDITIONS)),[r['auc_mean'] for r in rr],yerr=[r['auc_sd'] for r in rr],marker='o',ms=3,capsize=2,label=v,color=COLORS[v])
  ax.set(title=d,ylabel='AUROC',ylim=(0,1));ax.grid(alpha=.2)
 axes[0].legend(ncol=3);axes[-1].set_xticks(range(len(CONDITIONS)),CONDITIONS,rotation=35,ha='right')
 fig.suptitle('Retrained HL-GNN: all 14 original conditions (mean ± sample SD, 3 seeds)');fig.tight_layout(rect=[0,0,1,.97]);savefig(fig,a.output,'robustness')
 fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
 for ax,d in zip(axes,DATASETS):
  for offset,v in [(-.13,'fixed_weights'),(.13,'shallow_k2')]:
   rr=[next(r for r in paired if r['dataset']==d and r['contrast']=='full_minus_'+v and r['condition']==c) for c in SELECTED]
   ax.errorbar(np.arange(len(SELECTED))+offset,[r['mean_pp'] for r in rr],yerr=[r['sd_pp'] for r in rr],fmt='o',capsize=3,label='Full − '+v,color=COLORS[v])
  ax.axhline(0,color='gray',lw=1);ax.set(title=d,ylabel='Paired AUROC difference (pp)');ax.grid(alpha=.2)
 axes[0].legend(ncol=2);axes[-1].set_xticks(range(len(SELECTED)),SELECTED,rotation=25,ha='right');fig.suptitle('Component ablations: paired seeds; bars = sample SD, not confidence intervals');fig.tight_layout(rect=[0,0,1,.97]);savefig(fig,a.output,'component_ablations')
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,d in zip(axes,DATASETS):
  initial=[];learned=[];spread=[]
  for k in range(21):
   rr=[r for r in hop_rows if (r['dataset'],r['variant'],r['hop'])==(d,'full',k)]
   initial.append(rr[0]['initial']);learned.append(float(np.mean([r['learned'] for r in rr])));spread.append(float(np.std([r['learned'] for r in rr],ddof=1)))
  ax.plot(range(21),initial,'--',color='#888888',label='Analytical initial / fixed')
  ax.plot(range(21),learned,color='#176e86',label='Learned full (mean)');ax.fill_between(range(21),np.array(learned)-spread,np.array(learned)+spread,color='#176e86',alpha=.15)
  ax.axhline(0,color='gray',lw=.6);ax.set(title=d,xlabel='Propagation hop',ylabel='Signed hop coefficient');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Hop coefficients: 3-seed mean ± sample SD');fig.tight_layout(rect=[0,0,1,.94]);savefig(fig,a.output,'learned_hop_weights')
 # Paired component effects under the same frozen rank masks, for every condition and k.
 raw={}
 for path in a.curves.glob('*_seed*/per_seed_curves.csv'):
  for r in read(path):
   if r['model']=='hlgnn':raw[(r['dataset'],int(r['seed']),r['variant'],r['condition'],r['rank'],int(r['k']))]=r
 conditional=[]
 for d in DATASETS:
  for v in VARIANTS[1:]:
   for c in CONDITIONS:
    for rank in ['degree','pagerank']:
     for k in range(101):
      values=[]
      for seed in range(3):
       full=raw[(d,seed,'full',c,rank,k)];other=raw[(d,seed,v,c,rank,k)]
       assert (full['n_pos'],full['n_neg'])==(other['n_pos'],other['n_neg'])
       if full['auc'] and other['auc']:values.append(100*(float(full['auc'])-float(other['auc'])))
      mean,sd=meanstd(values) if values else (None,None)
      conditional.append(dict(dataset=d,contrast='full_minus_'+v,condition=c,rank=rank,k=k,n_seeds_valid=len(values),mean_pp=mean,sd_pp=sd,min_n_pos=min(int(raw[(d,s,'full',c,rank,k)]['n_pos']) for s in range(3)),min_n_neg=min(int(raw[(d,s,'full',c,rank,k)]['n_neg']) for s in range(3)),n_low_support=sum(int(raw[(d,s,'full',c,rank,k)]['low_support']) for s in range(3))))
 write_csv(a.output/'conditional_component_deltas.csv',conditional)
 # Main importance-conditioned comparisons use precisely the same edge masks for each seed.
 curve=read(a.curves/'summary.csv')
 for cond in ['clean','ea_0.5','delete_hub_0.5','feature_shuffle']:
  fig,axes=plt.subplots(3,2,figsize=(11,9),sharex=True,sharey=True)
  for i,d in enumerate(DATASETS):
   for j,rank in enumerate(['degree','pagerank']):
    ax=axes[i,j]
    for model,variant,label,color in [('katz','full_h5','Katz H=5','#ce8b16'),('hlgnn','full','HL-GNN full','#176e86')]:
     rr=sorted([r for r in curve if (r['dataset'],r['model'],r['variant'],r['condition'],r['rank'])==(d,model,variant,cond,rank)],key=lambda r:int(r['k']))
     assert len(rr)==101
     x=np.array([int(r['k']) for r in rr]);y=np.array([float(r['mean']) if r['mean'] and int(r['n_seeds_valid'])==3 else np.nan for r in rr]);sd=np.array([float(r['sd']) if r['sd'] else np.nan for r in rr]);low=np.array([int(r['n_low_support'])>0 for r in rr])
     ax.plot(x,y,label=label,color=color);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color,alpha=.12);ax.scatter(x[low],y[low],s=8,color=color)
    ax.set(title=f'{d} | {rank}',xlim=(0,100),ylim=(0,1),ylabel='Incident-edge AUROC',xlabel='Top k% nodes');ax.grid(alpha=.2)
  axes[0,0].legend();fig.suptitle(f'{cond}: same fixed training-graph ranking and test edge subsets\nMean ± sample SD; dots flag fewer than 20 positives or negatives in at least one seed');fig.tight_layout(rect=[0,0,1,.95]);savefig(fig,a.output,'rank_auc_'+cond)
 # Component curves: show all independently trained variants rather than only full.
 for cond in ['clean','ea_0.5','feature_shuffle']:
  fig,axes=plt.subplots(3,2,figsize=(11,9),sharex=True,sharey=True)
  for i,d in enumerate(DATASETS):
   for j,rank in enumerate(['degree','pagerank']):
    ax=axes[i,j]
    for v in VARIANTS:
     rr=sorted([r for r in curve if (r['dataset'],r['model'],r['variant'],r['condition'],r['rank'])==(d,'hlgnn',v,cond,rank)],key=lambda r:int(r['k']))
     assert len(rr)==101
     x=[int(r['k']) for r in rr];y=np.array([float(r['mean']) if r['mean'] and int(r['n_seeds_valid'])==3 else np.nan for r in rr]);sd=np.array([float(r['sd']) if r['sd'] else np.nan for r in rr])
     ax.plot(x,y,label=v,color=COLORS[v]);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=COLORS[v],alpha=.08)
     low=np.array([int(r['n_low_support'])>0 for r in rr]);ax.scatter(np.array(x)[low],y[low],s=8,color=COLORS[v])
    ax.set(title=f'{d} | {rank}',xlim=(0,100),ylim=(0,1),ylabel='Incident-edge AUROC',xlabel='Top k% nodes');ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);fig.suptitle(f'HL-GNN component curves | {cond} | mean ± sample SD; dots = low support');fig.tight_layout(rect=[0,0,1,.97]);savefig(fig,a.output,'rank_components_'+cond)
 maxdiff=max(comparisons,key=lambda r:abs(r['delta_auc']))
 def q(model,variant,k):
  return float(next(r['mean'] for r in curve if (r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k']))==('pubmed',model,variant,'feature_shuffle','degree',k)))
 lines=['# HL-GNN 重训、完整鲁棒性与重要节点 AUROC 报告','',
 '2026-11-24 新训练结果；旧逐边预测和权重已无法恢复，因此按用户要求重训。旧汇总与训练记录保留在 [历史记录](../hlgnn-historical/)，不将新模型伪装成旧 checkpoint。', '',
 f'**新增发现：** PubMed 特征打乱后，HL-GNN 的全局 AUROC 为 {q("hlgnn","full",100):.4f}，高于 Katz 的 {q("katz","full_h5",100):.4f}；但 Degree Top 10% 节点相关边上分别为 {q("hlgnn","full",10):.4f} 与 {q("katz","full_h5",10):.4f}，模型排序反转。该子集每个 seed 至少有 4,769 条正边和 1,184 条负边，不是极小样本点。它说明整体成绩可能掩盖重要节点子集的弱点，结论仅限本次特征打乱与固定负例协议。','',
 '## 训练与协议核验','',
 '- 使用恢复的 HL-GNN 实现；固定历史验证集选择的 `readme_exact` 配置，不重新选配置，不根据测试集调整超参。每次最多 150 epoch、每 5 epoch 验证，连续 15 次验证未改善后停止；按干净验证 AUROC 保存最佳模型。',
 '- Cora / CiteSeer / PubMed × seeds 0/1/2 × full / fixed_weights / shallow_k2：27 个独立训练模型。先核验每个数据集 full seed 0 的验证差异，再继续后续模型；不因为某个 seed 的测试成绩不好而重抽 seed。',
 '- 无向正边按 80% / 5% / 15% 划分 train / validation / test，各集合使用等量固定负例，负例集合互相不重叠。特征、全部 9 组 split 和测试边顺序均与历史指纹一致；评测使用 Junxiong 固定提交 `95083da808e500c02da458e205f4b861ceb80871` 的原生 `run.evaluate`。训练上下文移除当前正例批次；验证只用 train；测试上下文用 train+validation。',
 '- 共 117 张结构缓存图通过旧 HL-GNN / Katz 历史 graph_sha256 核验；特征打乱使用原协议的固定节点置换。没有生成新攻击图、重抽负例或对扰动图重新训练。',
 '- RF / DICE / EA 为 10%、25%、50%；随机删边、高度端点优先删边、随机插边为 50%；另有 clean 与整行特征打乱，共 14 条件。这里的 EA 是 Junxiong 的可见图线性代理攻击，不能称为与 DeepRobust 或 SGDiff 原论文攻击完全等价。',
 '- 本次恢复的是组内协议实验，不声称重新执行了 HL-GNN 原论文的 Planetoid Hits@100 实验。达到 150 epoch 上限不等于已证明收敛；本轮固定历史预算以保持可比，完整训练曲线随工件发布。环境、训练日志和 checkpoint 哈希另见交付工件。',
 '',f'重训与旧结果的最大逐 seed / 条件 AUROC 差异为 **{abs(maxdiff["delta_auc"]):.6f}**（{maxdiff["dataset"]} / {maxdiff["variant"]} / seed {maxdiff["seed"]} / {maxdiff["condition"]}，新−旧 {maxdiff["delta_auc"]:+.6f}）。完整对照见 [historical_comparison.csv](historical_comparison.csv)。不要求 CUDA 重训逐位相同，也不为迎合旧测试成绩反复重训。','',
 '另见 [重训差异诊断](retraining_diagnostics.md)：Cora fixed_weights seed 2 的新旧训练合同相同、初期 loss 仅约 1e-7 差异，但验证选中的 epoch 从 105 变为 75，特征打乱成绩变化较大。组件优势应重新解释，不能强制让完整版胜出。','',
 '![全部训练的验证曲线](training_validation.png)','','## 完整鲁棒性结果','',
 '下表为三 seed 均值 ± 样本标准差。AUROC / AP / Hits@100 / Precision@100 的全部 378 组结果在 [per_seed.csv](per_seed.csv)，所有 14 条件汇总在 [robustness_summary.csv](robustness_summary.csv)。','',
 '|数据集|版本|Clean AUROC|EA 50%|随机删边 50%|高度删边 50%|特征打乱|','|---|---|---|---|---|---|---|']
 for d in DATASETS:
  for v in VARIANTS:
   vals=[fmt(lookup[(d,v,c)]['auc_mean'],lookup[(d,v,c)]['auc_sd']) for c in ['clean','ea_0.5','delete_0.5','delete_hub_0.5','feature_shuffle']]
   lines.append('|'+ '|'.join([d,v]+vals)+'|')
 lines+=['','![14 条件鲁棒性](robustness.png)','','## 组件消融','',
 '`fixed_weights` 冻结解析 RWR / KI 跳权，保留 20-hop 传播及特征/MLP；`shallow_k2` 保留可学习跳权，将传播截断为 2-hop。三个版本分别训练，不能把推断时关模块当作这些消融。固定权重改变可训练参数数量；浅层版本同时改变可用跳数及跳权向量长度；在 Cora/CiteSeer 的 RWR 初始化中，最后一跳还承接剩余权重（K=2 时初始末项为 0.64，而 K=20 的第 2 跳为 0.128）。因此它是历史定义的浅层配置对照，并非完全隔离初始化影响的纯深度因果实验。','',
 '差值为同 seed 的 Full − Variant（百分点）；正值表示完整模型更高，误差为配对差值的样本标准差。三个 seed 不能支持强显著性或因果泛化主张。','',
 '|数据集|对照|Clean 增益 pp|EA 50% 增益 pp|特征打乱增益 pp|','|---|---|---|---|---|']
 for d in DATASETS:
  for v in VARIANTS[1:]:
   vals=[next(r for r in paired if (r['dataset'],r['contrast'],r['condition'])==(d,'full_minus_'+v,c)) for c in ['clean','ea_0.5','feature_shuffle']]
   lines.append('|'+ '|'.join([d,'full − '+v]+[f'{r["mean_pp"]:+.2f} ± {r["sd_pp"]:.2f}' for r in vals])+'|')
 lines+=['','![配对组件消融](component_ablations.png)','','![多跳权重](learned_hop_weights.png)','','跳权的数值展示模型如何偏离解析初始化；各阶传播表示的尺度和下游非线性不同，权重绝对值不能直接解释为严格的特征重要性。逐 seed 数值见 [learned_hop_weights.csv](learned_hop_weights.csv)。','','## 新指标：Top-k% 重要节点相关边的 AUROC','',
 '先按干净训练图的 Degree 或 PageRank 排节点；选前 ceil(N×k/100) 个节点，再在原测试候选边中保留“至少一端属于所选节点”的边，计算 AUROC。横轴 k=0..100；k=0 无定义，k=100 恢复全测试 AUROC。排名与测试边子集在所有模型、组件及扰动间固定。','',
 '这是重要节点相关预测的条件评测，**不是只挑预测最高分的边，也不是一种天然更优的总体指标**。Top 10% 节点通常覆盖远多于 10% 的正边，且负边覆盖率不同；因此同时报告正负样本数与覆盖率。不重新平衡正负例、不按攻击后的度重新排节点。少于 20 正或 20 负的点标为低支持。','',
 '|数据集|模型|条件|Degree Top 10% AUROC|Degree Top 100% AUROC|','|---|---|---|---|---|']
 for d in DATASETS:
  for m,v in [('katz','full_h5'),('hlgnn','full')]:
   for c in ['clean','ea_0.5','delete_hub_0.5','feature_shuffle']:
    rr=[next(r for r in curve if (r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k']))==(d,m,v,c,'degree',k)) for k in [10,100]]
    lines.append('|'+ '|'.join([d,m,c]+[fmt(float(r['mean']),float(r['sd'])) for r in rr])+'|')
 lines+=['','### 本次结果的解释边界','',
 '组件贡献取决于评测条件和节点子集，完整版并不必然在所有攻击与 k 上胜出。下面直接比较 Cora 特征打乱时的同 seed Full − Fixed：','',
 '|节点范围|Full − Fixed AUROC（pp，均值 ± 样本 SD）|','|---|---|']
 for k in [10,100]:
  r=next(r for r in conditional if (r['dataset'],r['contrast'],r['condition'],r['rank'],r['k'])==('cora','full_minus_fixed_weights','feature_shuffle','degree',k))
  lines.append(f'|Degree Top {k}%|{r["mean_pp"]:+.2f} ± {r["sd_pp"]:.2f}|')
 lines+=['','这说明全局组件结论可能掩盖重要节点子集中的不同表现；三个 seed 的差值及离散程度需要一起看，不能把均值符号变化宣称为统计显著或一般规律。完整条件组件差值见 [conditional_component_deltas.csv](conditional_component_deltas.csv)。','',
 'Katz 的 H=5/3/2 三个版本也全部进入新指标。这里的截断跳数是该经典路径计数方法的组件调查；Katz 不使用节点特征，因此 feature_shuffle 分数应与 clean 完全相同，这一不变量已逐组核验。HL-GNN 的特征打乱成绩变化则表明模型依赖属性证据，但本实验不能把这一影响等同于真实世界所有特征缺失形式。']
 for c in ['clean','ea_0.5','delete_hub_0.5','feature_shuffle']:
  lines+=['',f'### {c}','',f'![{c} 节点条件曲线](rank_auc_{c}.png)']
 lines+=['','### 组件的新指标曲线','','完整版和两种消融均计算了所有 14 条件 × 两种排名 × 101 个 k；下图展示 clean / EA50 / feature_shuffle，完整数值见归档。']
 for c in ['clean','ea_0.5','feature_shuffle']:lines+=['',f'![组件 {c}](rank_components_{c}.png)']
 lines+=['','## 结果文件与复算','',
 '- [predictions.zip](predictions.zip)：新 HL-GNN 逐边分数、queries、receipts、split、原生逐条件记录；没有 GPU 也可重算所有新指标。',
 '- [training_records.zip](training_records.zip)：27 个训练的完整 history / receipt / complete、配置、环境和验收记录。',
 '- [checkpoints.json](checkpoints.json)：27 个模型权重的 NAS 路径和 SHA256；权重保存在持久 NAS，没有把大权重提交进 Git 历史。',
 '- `../node-rank-auc/`：Katz 与重训 HL-GNN 合并的新指标汇总、逐 seed 曲线归档、协议与复算说明。',
 '- 代码在 `../../experiments/node_rank_auc/`，训练与原生评测入口在 `../../scripts/`。',
 '', '局限：只有 3 个数据集与 3 个既定种子；测试时观测退化，不含受损图重训。两种消融并非所有组件的穷尽实验。重要节点子集并不能消除负采样的度偏差，不能仅凭子集 AUC 上升宣称模型变好了。','']
 (a.output/'report.md').write_text('\n'.join(lines))
 (a.output/'report_audit.json').write_text(json.dumps(dict(new_evaluations=len(rows),checkpoints=len(checkpoints),historical_graphs_match=True,max_historical_auc_difference=maxdiff,paired_ablation_cells=len(paired)),indent=2)+'\n')
 print('Report and figures generated:',a.output)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['evaluation','training','historical','curves','output']:p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
