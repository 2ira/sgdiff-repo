# 中文 LaTeX 实验报告

- [完整 PDF](report.pdf)：三个模型介绍、相关工作、实验设计、结果图和完整数值附录。
- [TeX 主文件](report.tex)：与 `models.tex`、`methods.tex`、`results.tex`、`protected.tex`、`validation.tex` 共同构成正文。
- [三个模型的独立中文介绍](MODEL_INTRO.zh-CN.md)。
- [BibTeX](references.bib)：16条已核对的模型、基础方法、数据与攻击文献；正式发表版本和预印本区别在正文说明。
- `figures/`：3张新绘制的实际适配器结构图、3幅原论文局部摘图、7张实验综合图。
- `generated/`：直接从数据生成的全部数值表。
- `data/main/`、`data/protected/`：两套协议独立保留的 CSV、训练配置与审计；包含135次训练、1485次评估、495行跨种子汇总、330行组件配对汇总。
- `table_audit.json`：数据指纹与从逐种子结果重算均值/样本标准差的检查。
- `figure_sources.json`：原论文局部摘图的来源、页码、裁剪区域与原PDF指纹，版权属于原作者。

## 编译

推荐使用 TeX Live 的 XeLaTeX + BibTeX（含 `ctex`、Fandol 字体）或 Tectonic：

```bash
bash build.sh
```

也可指定已有 Tectonic 路径：

```bash
REPORT_TECTONIC=/path/to/tectonic bash build.sh
```

将本目录源文件包导入 Overleaf 后，选择 **XeLaTeX**，主文档选择 **report.tex**。LaTeX 会加载本目录的章节、BibTeX、生成表与图片，无需实验目录或GPU依赖。已有表格可直接编译；仅当重新生成表格时才需要 Python3 标准库。Tectonic 首次运行会下载所需宏包/字体，后续可复用缓存。报告使用 Fandol 字体，无需机器安装系统中文字体。

表格重建：

```bash
python3 build_tables.py
```

适配器结构示意图可用 `draw_architectures.py` 重建，实验图可用 `draw_results.py` 从随包CSV重建（需 Matplotlib/NumPy）；已有图为矢量PDF，不需要为编译安装 Matplotlib。论文局部摘图保持原始标签，正文明确区分原论文概览与本次运行结构。

## 实验状态与解释边界

M（main输入/攻击协议）已完成81次全新训练、729次评估，包含PubMed；PubMed EA遵守4000节点上限跳过。P（候选保护协议）已完成54次训练、756次评估，包含额外删边、插边和属性打乱探针。两轮权重、验证上下文、特征和测试候选不同，不混合排名。main EA本身为线性代理近似，并非原SGDiff论文DeepRobust谱攻击的逐行实现；旧P协议18张EA图实际全为插边。

本报告整理既有真实测量，不新增训练或更改原数值。原始图快照与划分仍位于 `../benchmarks/repository_protocol/` 和 `../benchmarks/ea_graphs/`；模型检查点、PPR、完整预测保留在外部实验目录，不重复放进TeX源包。
