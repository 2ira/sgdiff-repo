"""Compile-ready TeX supplement with all global results and key conditional-component tables."""
import argparse,csv,re,shutil
from pathlib import Path
from experiments.new_models_rank.summarize import DATASETS,VARIANTS,CONDITIONS,SELECTED,RANKS,read
from experiments.new_models_rank.plotting import NAMES,LABELS

def esc(s):
 s=str(s)
 for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
 return s

def pm(m,s,scale=1,nd=4):
 if m in [None,'']:return 'NA'
 return '$'+f'{scale*float(m):.{nd}f}'+(r'\pm'+f'{scale*float(s):.{nd}f}' if s not in [None,''] else '')+'$'
def longtable(caption,headers,rows,spec=None):
 spec=spec or 'l'*len(headers);n=len(headers)
 top=r'\toprule'+'\n'+' & '.join(headers)+r' \\'+'\n'+r'\midrule'+'\n'
 return '{\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+spec+'}\n\\caption{'+caption+'}\\\\\n'+top+'\\endfirsthead\n\\multicolumn{'+str(n)+'}{l}{'+caption+'（续）}\\\\\n'+top+'\\endhead\n\\midrule\\multicolumn{'+str(n)+'}{r}{续下页}\\\\\n\\endfoot\n\\bottomrule\\endlastfoot\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\end{longtable}\n}\n'
def fig(name,caption,landscape=False):
 if landscape:return '\\clearpage\n\\begin{landscape}\n\\begin{figure}[H]\\centering\\includegraphics[width=23cm,height=14cm,keepaspectratio]{'+name+'.pdf}\\caption{'+caption+'}\\end{figure}\n\\end{landscape}\n'
 return '\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth,height=.82\\textheight,keepaspectratio]{'+name+'.pdf}\\caption{'+caption+'}\\end{figure}\n'
def main(a):
 o=a.output;summary=read(o/'summary.csv');native=read(o/'per_seed.csv');robust=read(o/'robustness_summary.csv');paired=read(o/'conditional_component_deltas.csv')
 index={(r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k'])):r for r in summary}
 ci={(r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k'])):r for r in paired}
 def cv(d,m,c,rank,k,v='full'):
  x=index[d,m,v,c,rank,k];return pm(x['mean'],x['sd'])
 s=r'''\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,float,pdflscape,xcolor,xurl}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[unicode,colorlinks=true,linkcolor=blue!45!black,citecolor=green!40!black,urlcolor=blue!50!black]{hyperref}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 三模型统一鲁棒性与重要节点AUROC}\fancyhead[R]{\small xiaoming}\fancyfoot[C]{\thepage}
\setlength{\emergencystretch}{3em}\setlength{\LTcapwidth}{\textwidth}
\newcommand{\code}[1]{{\small\nolinkurl{#1}}}
\hypersetup{pdftitle={三个模型：统一鲁棒性、组件消融与重要节点AUROC},pdfauthor={xiaoming}}
\title{\textbf{NCNC、LPFormer、Refined-GAE}\\统一鲁棒性、组件消融与重要节点AUROC\\\large 对2026年11月24日分支交付的补充}
\author{xiaoming}\date{2026年11月26日}
\begin{document}\maketitle
\begin{abstract}
本报告对齐统一分支已发布的Katz、重训HL-GNN实验，为三个作者架构补充相同的Degree/PageRank Top-k\%节点相关测试边AUROC。Cora/Citeseer复用54份既有权重和756组真实逐边预测；PubMed按共享候选保护协议新增27次训练与378组评估。共有81份模型、1134组原生14条件评估、229068行逐种子条件曲线。所有模型使用相同117张图、9份划分、查询顺序和冻结节点排名。此为附加计分层：不根据预测分数选择测试边，不对每个k重新攻击，不将k=0填为虚构AUC。完整全局结果及逐种子表在附录展开，所有k和条件的原始数值随归档提供。
\end{abstract}
\noindent\textbf{提交状态：}本轮实验更新发布至 \code{feat/ncnc-lpformer-refined-gae}；协议与对照来源为统一分支11月24日交付。
\tableofcontents\clearpage
\section{数据、训练与协议对齐}
协议基准为统一分支提交 \code{bef9674}，原生评测器固定到 \code{95083da808e500c02da458e205f4b861ceb80871}。数据采用Planetoid引用网络\cite{yang2016planetoid}；Cora/Citeseer/PubMed分别为2708/3327/19717节点，5278/4552/44324无向边，1433/3703/500维属性。每seed正边80/5/15划分，训练、验证、测试各配等量且互不重叠负例，保留原完整测试集：每类792/683/6649条边。

输入为原始属性，训练移除当前正例批次；验证只使用训练图，测试图使用训练加验证边。PubMed作者专属配方沿用已固定配置，batch512、最多100epoch、每5轮按干净验证AUC选点、30轮无提升停止、梯度裁剪1，不用测试集调参。Cora/Citeseer旧54次拟合未重做；PubMed27次是全新模型。LPFormer训练PPR沿用作者完整训练图缓存，可能包含批次训练目标信息，推理时则为每个不同受扰图重算PPR。

\begin{itemize}
\item NCNC：hidden256，GCN dropout0.1，属性dropout0.3，预测器dropout0.05；编码器/预测器学习率0.0097/0.002；pt0.5、scale5.3、offset0.5、alpha0.3。
\item LPFormer：hidden128，dropout0.3，Adam学习率0.001、weight decay0.0001；PPR alpha0.15、eps$10^{-5}$、候选阈值0.01。
\item Refined-GAE：hidden512，2步传播，dropout0.4，学习率0.001，GELU两层预测器；深度混合参数初始化$[1,0.5]$后可学习。
\end{itemize}

\begin{center}\small\begin{tabular}{lrrrr}\toprule
数据&训练模型&其中新训练&原生评估&条件曲线行数\\\midrule
Cora&27&0&378&76356\\
Citeseer&27&0&378&76356\\
PubMed&27&27&378&76356\\\bottomrule\end{tabular}\end{center}

原生14条件为clean、RF/DICE/EA各10/25/50\%、随机删边50\%、高度相关删边50\%、纯插边50\%与属性行打乱。117张结构缓存直接导入已发布图并核验，不生成新EA；所有测试候选受保护。这里的EA是可见图线性代理梯度迁移攻击，不等价于原论文DeepRobust谱攻击或逐模型白盒最强攻击。实际27张EA图全部为插边，删边机制依据独立delete与delete\_hub条件解释。RF/DICE的50\%混合编辑预算不等于删除一半边。

与此前main补实验不同，本轮PubMed不是列标准化属性、500+500候选或验证正边可见的设置；它使用本分支原始属性、完整6649+6649候选及训练图验证。新PubMed EA能够评测，是因为现有H200图已归档，不需本机重新执行密集攻击，也没有擅自改变main的4000节点限制。

\section{三个模型的机制与组件}
三种模型均用于链路预测，但结构证据的处理不同：GCN\cite{kipf2017gcn}通过邻接传播编码节点；GAE/VGAE\cite{kipf2016vgae}提供编码器--解码器框架，GCN可以作为其编码器。子图方法SEAL\cite{zhang2018seal}与结构草图BUDDY\cite{chamberlain2023buddy}是相关结构学习路线，未作为本轮新增训练对象。LPFormer的个性化PageRank候选与局部图划分中的PPR思想相关\cite{andersen2006ppr}；LayerNorm\cite{ba2016layernorm}作为归一化组件被单独消融。这里的组件对照均按原作者实现定义，不能等同于结构-only/特征-only控制实验。

NCNC\cite{wang2024ncnc}先用GCN编码，再聚合共同邻居并对单侧邻居缺失连接作概率补全。noCNC关闭补全、保留NCN；noCN用作者端点预测器，仍含GCN$(A,X)$，不是特征-only。补全概率与训练正则化共同影响受扰表现，不能仅凭完整模型排名归因。

LPFormer\cite{shomer2024lpformer}用PPR选候选、点对注意力与位置编码形成链路表示。Uniform固定候选内均匀注意力但保留值投影；noPE只将位置编码置零，保留PPR筛选和结构计数。本报告的全局PageRank节点中心性不同于模型内部的个性化PPR，不把新指标当作模型自带重要性分数。

Refined-GAE\cite{ma2025refined}使用属性输入、可学习传播深度混合和Hadamard-MLP解码。Cora/Citeseer为4步1024维，PubMed为2步512维、GELU两层解码器；PubMed深度参数从$[1,0.5]$初始化后可学习。noLN只删除网络内LayerNorm，保留dropout和邻接归一化；本协议输入没有列标准化。Dot改成点积预测器后独立重训。组件效应同时包含参数量、非线性、正则化与训练轨迹的改变，不是冻结编码器后的孤立替换。

\section{Top-k\%节点相关边AUROC的定义}
协议原文见\href{https://github.com/2ira/sgdiff-repo/blob/unified-robustness-ablation/results/node-rank-auc/PROTOCOL.zh-CN.md}{统一节点重要性评测协议}，对照工作见\href{https://github.com/2ira/sgdiff-repo/blob/unified-robustness-ablation/results/hlgnn-retrained/report.md}{HL-GNN重训报告}。\par
每dataset/seed仅用干净训练正边计算节点Degree和PageRank（alpha0.85，均匀teleport/dangling、tol$10^{-12}$、maxiter1000）。排名不使用验证或测试正边，不读取类别或预测，不在攻击后重算。按冻结排序取$m(k)=\lceil Nk/100\rceil$个节点，选原正负测试边中至少一端命中的边：
\[
\mathcal Q_k=\{(u,v)\in\mathcal Q_{\rm test}:u\in S_k\lor v\in S_k\}.
\]
两端均命中的边只计一次，正负类使用同一规则，不重采负例或重新平衡。
\[
\operatorname{AUC}_k=\frac{\sum_{p\in P_k,n\in N_k}(\mathbf1[s_p>s_n]+0.5\mathbf1[s_p=s_n])}{|P_k||N_k|}.
\]
$k=0$无样本、AUC为NA；$k=100$必须还原原全局AUC。任一类为空为NA，少于20正或20负为低支持。主图纵轴固定0--1，低支持用空心点，不连接成同样强证据的曲线。阴影为3seed样本SD（ddof1），不是置信区间。Top10\%节点通常覆盖远多于10\%正边，且正负覆盖率不等，因此曲线上升本身不代表模型增强。

本机依赖版本会使少量近并列PageRank节点换序。适配层验证训练图指纹、Degree、PageRank数值及tie key，随后复用发布的排名CSV，要求输出与Katz、HL-GNN逐字节一致；原计分逻辑不变。复用冻结排名保证跨模型的子集完全相同，不对某模型挑有利排序。
'''
 s+=fig('robustness_all','全部14条件、三个模型及两个独立重训消融的全局AUROC。',True)
 s+=fig('test_edge_coverage','原固定候选的正负边覆盖率。所有模型、消融和攻击共用相同子集。',True)
 s+='\\section{五模型比较与代表发现}\nKatz和HL-GNN直接引用本分支已发布分数，不重新训练。下图展示四个代表条件；全部14条件和所有k保留在CSV及逐seed归档。\n'
 for c in SELECTED:s+=fig('rank_auc_'+c,esc(c)+'：同图、同查询、同节点子集的五模型AUROC，三seed均值与样本SD。',False)
 for d in DATASETS:
  rows=[]
  for m in VARIANTS:
   for c in SELECTED:
    r=index[d,m,'full',c,'degree',10]
    rows.append([NAMES[m],esc(c),cv(d,m,c,'degree',10),cv(d,m,c,'pagerank',10),cv(d,m,c,'degree',100),str(r['min_n_pos']),str(r['min_n_neg'])])
  s+=longtable(d+'完整模型关键比较',['模型','条件','Degree10','PageRank10','全局AUC','最少正','最少负'],rows,'llrrr rr')
 s+='Citeseer EA50下，NCNC与LPFormer全局AUC分别为'+cv('citeseer','ncnc','ea_0.5','degree',100)+'、'+cv('citeseer','lpformer','ea_0.5','degree',100)+'；Degree Top10分别为'+cv('citeseer','ncnc','ea_0.5','degree',10)+'、'+cv('citeseer','lpformer','ea_0.5','degree',10)+'。均值排序反转提示全局成绩可能掩盖特定子集差异，但不是显著性或因果泛化结论。\n'
 joint=read(o/'five_model_full_summary.csv')
 ji={(r['dataset'],r['model'],r['condition'],r['rank'],int(r['k'])):r for r in joint}
 rows=[]
 for m in ['katz','hlgnn','ncnc','lpformer','refined_gae']:
  rs=[ji['pubmed',m,'feature_shuffle',rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
  rows.append([NAMES[m],*[pm(r['mean'],r['sd']) for r in rs]])
 s+=longtable('PubMed属性错配：相同测试子集的五模型比较',['模型','Degree10','PageRank10','全局AUC'],rows,'lrrr')
 s+='三个新模型的全局AUROC均值都高于Katz，但Degree Top10均值都低于Katz；该结果与重训HL-GNN的观察呼应。每seed该子集至少4769条正边、1184条负边，并非极小类别数。结论仅限固定负例与属性错配的三seed结果，不代表统计显著性或所有扰动的总体排序。\n'
 s+='\\subsection{固定节点子集的配对扰动变化}\n差值为扰动−clean，单位pp，负值代表退化；先逐seed相减，再汇总，不从两组SD相减。全部k与14条件在delta\\_summary.csv中保存。\n'
 di={(r['dataset'],r['model'],r['condition'],r['rank'],int(r['k'])):r for r in read(o/'delta_summary.csv') if r['variant']=='full'}
 for d in DATASETS:
  rows=[]
  for m in VARIANTS:
   for c in ['ea_0.5','delete_hub_0.5','feature_shuffle']:
    rr=[di[d,m,c,rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
    rows.append([NAMES[m],esc(c),*[pm(r['mean'],r['sd'],100,2) for r in rr]])
  s+=longtable(d+'固定子集的扰动−clean',['模型','条件','Degree10 pp','PageRank10 pp','全局pp'],rows,'llrrr')
 s+='\\section{组件贡献依赖条件和节点子集}\n以下差值均先按同seed计算，再汇总为Full−Variant，单位百分点。完整−noLN在属性置换下的差异不能定位编码器或解码器单独作用。\n'
 rows=[]
 for d in DATASETS:
  for m,vs in VARIANTS.items():
   for v in vs[1:]:
    r=ci[d,m,v,'feature_shuffle','degree',10];q=ci[d,m,v,'feature_shuffle','degree',100]
    rows.append([d,NAMES[m],LABELS[v],pm(r['mean_pp'],r['sd_pp'],nd=2),pm(q['mean_pp'],q['sd_pp'],nd=2)])
 s+=longtable('特征打乱下条件与全局的配对组件效应',['数据','模型','对照','Degree10 pp','全局 pp'],rows,'lllrr')
 r=ci['pubmed','ncnc','no_completion','feature_shuffle','degree',10];q=ci['pubmed','ncnc','no_completion','feature_shuffle','degree',100]
 s+='PubMed属性错配下，NCNC Full−noCNC在Degree Top10与全局分别为'+pm(r['mean_pp'],r['sd_pp'],nd=2)+'、'+pm(q['mean_pp'],q['sd_pp'],nd=2)+' pp，与Cora/Citeseer的均值方向一致。\n'
 for v in ['uniform_attention','no_ppr_encoding']:
  r=ci['pubmed','lpformer',v,'feature_shuffle','degree',10];q=ci['pubmed','lpformer',v,'feature_shuffle','degree',100]
  s+='LPFormer Full−'+LABELS[v]+'在PubMed属性错配下，Degree Top10为'+pm(r['mean_pp'],r['sd_pp'],nd=2)+' pp，全局为'+pm(q['mean_pp'],q['sd_pp'],nd=2)+' pp。\n'
 s+='这些三seed差值不支持将LPFormer此条件的较高绝对AUC直接归因为可学习注意力或位置编码；两个变体均保留PPR候选和值投影。\n'
 s+=fig('conditional_component_effects','特征打乱：Top10与全局组件效应。误差为配对差的样本SD，不是显著性检验。',True)
 for c in ['clean','ea_0.5','feature_shuffle']:
  for rank in RANKS:s+=fig(f'components_{c}_{rank}',esc(rank+' / '+c)+'：三个模型的完整与组件版本，全部独立训练。',True)
 s+=fig('refined_depth_weights','Refined-GAE学到的传播深度softmax混合。系数不是因果重要性，PubMed使用2步，其余4步。',True)
 s+=r'''\section{核验、训练记录与解释边界}
九份划分与原始特征已与重训HL-GNN匹配，117图已核验历史哈希。每组AUC/AP/严格负池Hits100及并列平均Precision100由真实逐边分数复算，k100逐组匹配全局AUC；无定义和低支持状态显式保存。新增27份检查点独立重载核验，逐seed曲线归档再通过无GPU复算。详细误差与阈值见随附审计JSON；逐边分数与训练记录分别见\code{predictions.zip}、\code{training_records.zip}，\code{report_sources.zip}仅用于独立编译报告。

本报告是统一协议下的作者架构复现，不复刻三篇论文原表；负例、评价指标与训练预算存在差异。只有三个既定seed，未进行强显著性声明。所有模型保留自己的训练配方，不能将跨模型差异完全归因于架构。冻结hub子集不消除原随机负例的度偏差；不同k还改变测试人群。较大注意力、传播系数或配对差值不等于单个因果机制已被隔离。模型权重与原图不因k改变，PubMed旧main与本轮结果也不混在同表。

全部81条干净验证曲线在下图；Cora/Citeseer是已保存历史，PubMed是新增拟合。训练图使用标注的放大纵轴，主要条件AUC仍固定0--1。达到100epoch上限不等于已证明收敛。独立训练中不重抽测试seed、不为迎合历史均值选择重训次数。
'''
 for d in DATASETS:s+=fig('training_'+d,d+'：27条验证历史与所选检查点（星号），验证上下文仅含训练图。',True)
 s+='\\bibliographystyle{plainnat}\\bibliography{references}\n\\clearpage\\appendix\n\\section{全部378行原生跨seed汇总}\n分数为0--1。Strict Hits100使用固定全局负池，Precision100对截止并列组平均；它们不是将main误命名的hit100原样移入本表。降幅为clean−perturbed，百分点。\n'
 for d in DATASETS:
  for m,vs in VARIANTS.items():
   rows=[]
   for v in vs:
    for c in CONDITIONS:
     r=next(r for r in robust if (r['dataset'],r['model'],r['variant'],r['condition'])==(d,m,v,c))
     rows.append([LABELS[v],esc(c),*[pm(r[k+'_mean'],r[k+'_sd']) for k in ['auc','ap','hits100','precision100']],pm(r['drop_auc_mean'],r['drop_auc_sd'],100,2)])
   s+=longtable(d+' / '+NAMES[m]+'：全部条件',['版本','条件','AUC','AP','Hits100','Prec100','降幅pp'],rows,'llrrrrr')
 s+='\\clearpage\\section{全部1134行原生逐seed结果}\n每行对应实际保存的正负分数向量；未从汇总表反推。\n'
 for d in DATASETS:
  for m,vs in VARIANTS.items():
   rows=[]
   for r in sorted([r for r in native if r['dataset']==d and r['model']==m],key=lambda r:(vs.index(r['variant']),int(r['seed']),CONDITIONS.index(r['condition']))):rows.append([LABELS[r['variant']],r['seed'],esc(r['condition']),*[f'{float(r[k]):.6f}' for k in ['auc','ap','hits100','precision100']]])
   s+=longtable(d+' / '+NAMES[m]+'：逐seed指标',['版本','seed','条件','AUC','AP','Hits100','Prec100'],rows,'lrlrrrr')
 s+='\\clearpage\\section{全部条件：Degree10、PageRank10及全局组件效应}\n本表展开全部14条件的同seed Full−Variant效应，单位pp。所有k0--100的50904行组件汇总及229068行逐seed曲线在CSV/ZIP中保留，不把k0的NA替换为数值。\n'
 for d in DATASETS:
  rows=[]
  for m,vs in VARIANTS.items():
   for v in vs[1:]:
    for c in CONDITIONS:
     rr=[ci[d,m,v,c,rank,k] for rank,k in [('degree',10),('pagerank',10),('degree',100)]]
     rows.append([NAMES[m],LABELS[v],esc(c),*[pm(r['mean_pp'],r['sd_pp'],nd=2) for r in rr]])
  s+=longtable(d+'全部条件组件差值',['模型','变体','条件','Degree10 pp','PageRank10 pp','全局pp'],rows,'lllrrr')
 s+='\\end{document}\n'
 (o/'report.tex').write_text(s)
 print('TeX supplement written',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
