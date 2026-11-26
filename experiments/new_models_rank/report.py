"""Produce the reviewable Chinese report, complete native tables, and model-specific explanations."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.new_models_rank.summarize import DATASETS,VARIANTS,CONDITIONS,SELECTED,RANKS,read,write,dump
from experiments.new_models_rank.plotting import NAMES,LABELS

def pm(m,s,digits=4):
 if m in [None,'']:return 'NA'
 return f'{float(m):.{digits}f} ± {float(s):.{digits}f}' if s not in [None,''] else f'{float(m):.{digits}f}'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in row)+'|' for row in rows])
def main(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);audit=json.loads((out/'delivery_audit.json').read_text());assert audit['passed'] and not audit['partial']
 curves=read(out/'summary.csv');global_rows=read(out/'robustness_summary.csv');paired=read(out/'paired_component_deltas.csv');cond=read(out/'conditional_component_deltas.csv');joint=read(out/'five_model_full_summary.csv')
 index={(r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k'])):r for r in curves}
 gl={(r['dataset'],r['model'],r['variant'],r['condition']):r for r in global_rows}
 ci={(r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k'])):r for r in cond}
 def at(d,m,c,k,rank='degree',v='full'):
  r=index[d,m,v,c,rank,k];return pm(r['mean'],r['sd'])
 lines=['# NCNC、LPFormer、Refined-GAE：统一鲁棒性、组件消融与重要节点 AUROC','',
 '**2026-11-26 实验更新，交付分支 `feat/ncnc-lpformer-refined-gae`。** 本报告基于 `unified-robustness-ablation@bef9674` 的 2026-11-24 交付。主结果全部完成后生成，不以计划代替实测。','',
 '## 完成内容与复用范围','',
 '- **三个数据集 × 三个模型 × 三个组件版本 × 三个种子 =81份模型权重、1134组原生14条件评估。** Cora/Citeseer复用原候选保护协议的54份权重与756组逐边预测；PubMed按共享协议新增27次从头训练与378组评估。未用后来main补实验的标准化特征权重替代本轮模型。',
 '- Degree/PageRank × k=0..100：新增 **229,068行逐seed条件曲线、76,356行跨seed汇总、50,904行配对组件条件汇总**。k=0的空样本点保留NA，不算作有定义AUC。',
 '- 对齐已有Katz和重训HL-GNN：9份划分、原始属性、117张结构缓存、查询顺序和冻结节点排名逐项核验；只读取已发布攻击图，不生成新EA或为每个k重新扰动图。',
 '- 全局AUROC/AP/严格Hits@100/并列平均Precision@100均核验；每条新曲线k=100还原原模型全测试AUROC。三个模型与两位现有比较对象的训练配方各自保留，统一的是数据、攻击、测试对象和计分层。','',
 '![全部14条件和组件](robustness_all.png)','',
 '## 新指标到底测什么','',
 '每个dataset/seed仅用干净`train_pos`计算Degree和PageRank（alpha=0.85，均匀teleport与dangling，tol=1e-12，max_iter=1000）。先按分数降序，再按既定SHA256键处理并列；排名图不包含验证/测试正边。取前ceil(N×k/100)个节点，从原测试正负边同时筛出“至少一端命中”的边，计算pooled AUROC。不重新平衡负例，不按预测分数挑边，不按攻击后图重新排节点。','',
 'k=100是全测试集，k=0没有样本、AUROC=NA；任一类别为空为NA；少于20正例或20负例标为低支持。图纵轴固定0–1，三seed阴影为样本标准差，不是置信区间。低支持点用空心点表示，不连接成与充分样本相同的主曲线。','',
 '**冻结排名的额外核验：** 本机NetworkX与原交付环境不同，PageRank数值近乎相同仍可能使少量近并列节点换序。`frozen_rank.py`验证训练图指纹、Degree、PageRank数值及tie key后，直接复用已发布`node_ranks.csv`，输出须与Katz、HL-GNN的排名CSV逐字节一致。原`node_rank_auc/evaluate.py`的计分逻辑保持不变，适配层与排名源哈希单独记录。','',
 '![同一节点比例的正负边覆盖率](test_edge_coverage.png)','',
 'Top10%节点通常覆盖远多于10%的正边，且正负覆盖率不等；较高条件AUROC可能来自测试人群改变，不能解释为选择节点使模型变强。Degree与PageRank可能高度相关。本指标的全局PageRank中心性也不同于LPFormer内部为源节点计算的个性化PPR，两者用途不同。','',
 '## 五模型在同图、同候选上的曲线','',
 'Katz与HL-GNN直接引用本分支已发布曲线，没有重新训练它们；三个新模型使用同一组排名与测试边子集。所有14条件均进入数值归档，下图选择代表条件。']
 for c in SELECTED:lines+=['',f'### {c}','',f'![{c}下五模型Degree/PageRank曲线](rank_auc_{c}.png)']
 lines+=['','## Top10%与全局：详细关键结果','']
 for d in DATASETS:
  body=[]
  for m in VARIANTS:
   for c in SELECTED:
    degree=index[d,m,'full',c,'degree',10];pr=index[d,m,'full',c,'pagerank',10]
    body.append([NAMES[m],c,at(d,m,c,10),at(d,m,c,10,'pagerank'),at(d,m,c,100),degree['min_n_pos'],degree['min_n_neg'],pr['min_n_pos'],pr['min_n_neg']])
  lines += [f'### {d}', '',table(['模型','条件','Degree10 AUC','PageRank10 AUC','全局AUC','D10最少正例','D10最少负例','P10最少正例','P10最少负例'],body),'']
 # Reproducible, data-derived examples; directions and dispersions are shown together.
 joint_index={(r['dataset'],r['model'],r['condition'],r['rank'],int(r['k'])):r for r in joint}
 lines+=['## PubMed属性错配：五模型的关键节点与全局比较','']
 comparison=[]
 for m in ['katz','hlgnn','ncnc','lpformer','refined_gae']:
  rs=[joint_index['pubmed',m,'feature_shuffle',rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
  comparison.append([NAMES[m],*[pm(r['mean'],r['sd']) for r in rs]])
 lines += [table(['模型','Degree Top10 AUC','PageRank Top10 AUC','全局AUC'],comparison),'']
 reversed_models=[]
 for m in VARIANTS:
  global_new=float(joint_index['pubmed',m,'feature_shuffle','degree',100]['mean']);global_katz=float(joint_index['pubmed','katz','feature_shuffle','degree',100]['mean'])
  top_new=float(joint_index['pubmed',m,'feature_shuffle','degree',10]['mean']);top_katz=float(joint_index['pubmed','katz','feature_shuffle','degree',10]['mean'])
  if global_new>global_katz and top_new<top_katz:reversed_models.append(NAMES[m])
 if reversed_models:
  lines += ['在这次PubMed特征打乱实验中，'+ '、'.join(reversed_models)+'的全局均值高于Katz，但Degree Top10均值低于Katz。该现象与已有HL-GNN交付中的观察相呼应，说明重要节点子集的弱点并不限于一个模型；仍只限这批固定负例与属性错配条件，不能称为显著性或一般规律。','']
 lines+=['## 固定节点子集的扰动降幅','', '以下为Full模型的“扰动−clean”，单位百分点；每个seed在相同k、相同边子集上相减，再汇总为均值±样本SD。负值代表退化。全14条件、所有k见`delta_summary.csv`。','']
 di={(r['dataset'],r['model'],r['condition'],r['rank'],int(r['k'])):r for r in read(out/'delta_summary.csv') if r['variant']=='full'}
 for d in DATASETS:
  rows=[]
  for m in VARIANTS:
   for c in ['ea_0.5','delete_hub_0.5','feature_shuffle']:
    rr=[di[d,m,c,rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
    rows.append([NAMES[m],c,*[pm(100*float(r['mean']),100*float(r['sd']),2) for r in rr]])
  lines += ['### '+d,'',table(['模型','条件','Degree10 Δ pp','PageRank10 Δ pp','全局 Δ pp'],rows),'']
 lines+=['## 可以支持的观察','',
 '下面比较的是同一批已冻结子集，数值均为三seed均值±样本SD；均值排序变化不是显著性检验。','',
 '1. **重要节点子集可以改变模型均值排序。** Citeseer的EA50下，NCNC/LPFormer全局AUC分别为'+at('citeseer','ncnc','ea_0.5',100)+' / '+at('citeseer','lpformer','ea_0.5',100)+'；Degree Top10%分别为'+at('citeseer','ncnc','ea_0.5',10)+' / '+at('citeseer','lpformer','ea_0.5',10)+'。子集上的LPFormer均值超过NCNC，而全局相反。该比较不说明LPFormer在所有攻击或数据集上更稳健。','']
 for d in DATASETS:
  r=ci[d,'refined_gae','no_norm','feature_shuffle','degree',10];g=ci[d,'refined_gae','no_norm','feature_shuffle','degree',100]
  lines += [f'- {d}特征打乱时，Refined-GAE Full−noLN：Degree Top10为 **{pm(r["mean_pp"],r["sd_pp"],2)} pp**，全局为 **{pm(g["mean_pp"],g["sd_pp"],2)} pp**。']
 lines+=['','2. **归一化贡献依赖评测子集。** 上述差值衡量网络内LayerNorm组与独立重训的共同影响。noLN仅移除LayerNorm，dropout与邻接归一化保留；本协议输入为原始属性，没有列标准化。它不能区分编码器、解码器各自贡献。特征打乱只是属性错配，不代表所有真实特征损坏。','',
 '3. **补全不能被自动当作鲁棒性来源。** 在Cora/Citeseer特征打乱下，Full−noCNC的Degree Top10配对AUC差分别为 '+pm(ci['cora','ncnc','no_completion','feature_shuffle','degree',10]['mean_pp'],ci['cora','ncnc','no_completion','feature_shuffle','degree',10]['sd_pp'],2)+' / '+pm(ci['citeseer','ncnc','no_completion','feature_shuffle','degree',10]['mean_pp'],ci['citeseer','ncnc','no_completion','feature_shuffle','degree',10]['sd_pp'],2)+' pp。该设置下完整补全版本均值更低；可能的属性错配传播机制仍是待检验假设，不能反推所有补全方法无效。','',
 '补充PubMed：Full−noCNC的Degree Top10差为 '+pm(ci['pubmed','ncnc','no_completion','feature_shuffle','degree',10]['mean_pp'],ci['pubmed','ncnc','no_completion','feature_shuffle','degree',10]['sd_pp'],2)+' pp，全局为 '+pm(ci['pubmed','ncnc','no_completion','feature_shuffle','degree',100]['mean_pp'],ci['pubmed','ncnc','no_completion','feature_shuffle','degree',100]['sd_pp'],2)+' pp；与前两个数据集的均值方向一致。','',
 'PubMed特征打乱下，LPFormer Full−Uniform：Degree Top10为 '+pm(ci['pubmed','lpformer','uniform_attention','feature_shuffle','degree',10]['mean_pp'],ci['pubmed','lpformer','uniform_attention','feature_shuffle','degree',10]['sd_pp'],2)+' pp，全局为 '+pm(ci['pubmed','lpformer','uniform_attention','feature_shuffle','degree',100]['mean_pp'],ci['pubmed','lpformer','uniform_attention','feature_shuffle','degree',100]['sd_pp'],2)+' pp。Full−noPE分别为 '+pm(ci['pubmed','lpformer','no_ppr_encoding','feature_shuffle','degree',10]['mean_pp'],ci['pubmed','lpformer','no_ppr_encoding','feature_shuffle','degree',10]['sd_pp'],2)+' / '+pm(ci['pubmed','lpformer','no_ppr_encoding','feature_shuffle','degree',100]['mean_pp'],ci['pubmed','lpformer','no_ppr_encoding','feature_shuffle','degree',100]['sd_pp'],2)+' pp。三seed差值较小，不能把LPFormer在该设置的较高绝对AUC直接归因为可学习注意力或位置编码；两种变体仍共享PPR候选和值投影。','',
 '4. **同时看绝对性能、同k扰动差与样本覆盖率。** 降幅表采用“扰动−clean”，负值是退化；组件表采用“Full−Variant”，正值是完整模型更高。两个符号约定不同。两类差值均先逐seed计算再汇总，不能相减两个SD得到配对SD。','',
 '![特征打乱下Top10与全局组件效应](conditional_component_effects.png)','',
 '## 三个模型与本轮组件操作','',
 table(['模型','主要信息路径','两个独立重训变体','解释边界'],[
 ['NCNC','GCN节点表示、端点乘积、共同邻域与缺失关系概率补全','noCNC：补全深度1→0；noCN：作者端点预测器','noCN仍含GCN(A,X)，不是特征-only'],
 ['LPFormer','GCN、PPR候选、点对到节点注意力、位置编码与计数','Uniform：固定候选内均匀权重；noPE：位置编码置零','两者均保留候选和值投影；noPE仍用PPR筛选'],
 ['Refined-GAE','属性输入、传播深度混合、端点Hadamard-MLP解码','noLN：仅移除LayerNorm；Dot：改点积后重训','保留dropout及邻接归一化；Dot改变训练、参数量与非线性']]),'',
 '模型论文：[NCNC, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html)、[LPFormer, KDD2024](https://arxiv.org/abs/2310.11009)、[Refined-GAE, CIKM2025](https://arxiv.org/abs/2411.03845)。较完整的模型介绍与原论文图见此前[LaTeX研究报告](https://github.com/2ira/sgdiff-repo/tree/2f4a482/experiments/new_link_models/paper_report)。','']
 for c in ['clean','ea_0.5','feature_shuffle']:
  for rank in RANKS:lines += [f'### {c} / {rank}', '',f'![三个模型的组件条件曲线](components_{c}_{rank}.png)','']
 lines+=['## 学到的模型参数','',
 '![Refined-GAE传播深度混合](refined_depth_weights.png)','',
 'Refined-GAE的传播深度softmax系数分别按数据集与组件绘出；Cora/Citeseer为4步，PubMed为2步。这些系数描述混合方式，不是严格的因果重要性。`learned_parameters.csv`还记录NCNC的beta/JK标量；CN0注册的beta在其前向路径不使用，不能按数值大小把它解释为共同邻居贡献。LPFormer没有统一的全局注意力重要性向量，其注意力依赖具体点对；Uniform的零向量已逐权重核验，不能把注意力集中等同因果归因。','',
 '## PubMed新增训练的精确合同与原实验区别','',
 'PubMed原始Planetoid属性，19717节点、44324无向边、500维。沿用80/5/15互斥正边、各集合等量且互不重叠的固定负例。正例batch512，移除当前批次目标边；每5epoch按仅训练图上下文的干净验证AUC选点，最多100epoch、30epoch无提升停止、梯度范数裁剪1。保留作者数据集专属优化配置；不重做超参搜索，不因测试表现重抽seed。LPFormer训练PPR仍来自完整训练图，可能保留批次训练目标信息，推理PPR则按每张攻击图重新计算。','',
 '- NCNC：hidden256；GCN dropout0.1、xdropout0.3；predictor dropout0.05；lr0.0097/0.002；pt0.5、scale5.3、offset0.5、alpha0.3。',
 '- LPFormer：hidden128、lr0.001、dropout0.3、weight decay0.0001；PPR alpha0.15、eps1e-5，候选阈值0.01。',
 '- Refined-GAE：hidden512、2步传播、dropout0.4、lr0.001、GELU两层预测器；传播深度参数从[1,0.5]初始化，softmax后可学习。','',
 '与11月21日main补实验不同：这里使用原始属性、训练图验证、完整6649正+6649负测试候选，并加载本分支的PubMed图。旧main的标准化权重与500+500候选结果不能混入本报告。如今PubMed的EA也能评，是因为直接复用已发布H200攻击图，不需要在本机重新运行密集攻击，更没有擅自提高旧main节点上限。','',
 '**攻击边界：** 原生EA是可见图线性代理梯度迁移攻击，不等价于DeepRobust谱攻击或目标模型白盒最强攻击。实际共享的27张EA图（含PubMed新增9张）全部为插边；删边结论应依据delete/delete_hub独立条件。RF/DICE的预算为混合删插次数，不是纯删边比例。','',
 '## 训练、审计与可复算工件','',
 '全部81个模型的干净训练图验证曲线与所选epoch如下；验证图仅含训练边，不同于main补实验的验证上下文。训练曲线使用已标注的放大纵轴，主要节点条件AUROC图仍固定0–1。达到100epoch预算上限不等于已证明收敛。','',
 '![Cora全部27条验证曲线](training_cora.png)','','![Citeseer全部27条验证曲线](training_citeseer.png)','','![PubMed全部27条验证曲线](training_pubmed.png)','',
 '- [逐seed原生指标](per_seed.csv)、[全部14条件汇总](robustness_summary.csv)、[全局配对消融](paired_component_deltas.csv)。',
 '- [全部k的汇总](summary.csv)、[扰动−clean](delta_summary.csv)、[关键k](key_points.csv)、[全部条件组件差](conditional_component_deltas.csv)。',
 '- [逐seed曲线与冻结排名](per_seed_curves.zip)、[原始逐边预测与查询](predictions.zip)、[训练日志与配置](training_records.zip)、[权重清单](checkpoints.json)。',
 '- [整体交付审计](delivery_audit.json)、[独立复算审计](replay_audit.json)、[无需GPU的复算代码](../../experiments/new_models_rank/README.md)。',
 '- [TeX源码](report.tex)与[PDF补充报告](report.pdf)。Katz和HL-GNN的原报告及分数保持不变。','',
 '这是统一协议下的作者架构复现，不是三篇论文原表的逐项复刻；负例、评价指标和训练预算存在明确差异。只有3个既定seed；没有强显著性或泛化因果声明。所有排名和测试边子集固定，不消除随机负例在hub子集的度偏差。历史Cora/Citeseer与新PubMed的拟合时间/设备不同，跨模型还保留各自优化配方。纯缺边重训、度匹配负例和自适应白盒攻击应另设协议，不能混入本轮表。','',
 '## 全部14条件：全局完整结果','',
 '下面每行都是3seed均值±样本SD；AUC降幅为clean−perturbed，单位pp。完整1134行逐seed结果在末尾展开。']
 for d in DATASETS:
  for m in VARIANTS:
   body=[]
   for v in VARIANTS[m]:
    for c in CONDITIONS:
     r=gl[d,m,v,c];body.append([LABELS[v],c,*[pm(r[k+'_mean'],r[k+'_sd']) for k in ['auc','ap','hits100','precision100']],pm(100*float(r['drop_auc_mean']),100*float(r['drop_auc_sd']),2)])
   lines += [f'### {d} / {NAMES[m]}','',table(['版本','条件','AUROC','AP','Strict Hits100','Tieavg Precision100','AUC降幅pp'],body),'']
 lines+=['## 全部14条件：Degree/PageRank Top10及全局条件组件效应','', '差值为同seed Full−Variant，单位pp。每行对应一个固定排名与k，完整0..100的50904行汇总在CSV中保留；k=0无定义，不填虚构值。','']
 for d in DATASETS:
  body=[]
  for m,vs in VARIANTS.items():
   for v in vs[1:]:
    for c in CONDITIONS:
     rr=[ci[d,m,v,c,rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
     body.append([NAMES[m],LABELS[v],c,*[pm(r['mean_pp'],r['sd_pp'],2) for r in rr]])
  lines += [f'### {d}','',table(['模型','消融','条件','Degree10 Full−V pp','PageRank10 Full−V pp','全局 Full−V pp'],body),'']
 lines+=['## 全部1134次原生逐seed评估','', '分数为0..1，未从汇总表反推；每个点均有保存的真实逐边预测。','']
 ns=read(out/'per_seed.csv')
 for d in DATASETS:
  for m in VARIANTS:
   rr=sorted([r for r in ns if r['dataset']==d and r['model']==m],key=lambda r:(VARIANTS[m].index(r['variant']),int(r['seed']),CONDITIONS.index(r['condition'])))
   lines += [f'### {d} / {NAMES[m]}','',table(['版本','seed','条件','AUROC','AP','Strict Hits100','Tieavg Precision100'],[[LABELS[r['variant']],r['seed'],r['condition'],*[f'{float(r[k]):.6f}' for k in ['auc','ap','hits100','precision100']]] for r in rr]),'']
 (out/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
 print('Chinese report complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
