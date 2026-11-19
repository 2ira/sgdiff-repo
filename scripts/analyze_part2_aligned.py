#!/usr/bin/env python
"""Validate and summarize the aligned SP4LP/Katz robustness study.

The script treats the four native JSON files as the source of truth.  It checks
coverage and protocol invariants, then writes tidy result/ablation CSV files and
a Chinese report whose tables are generated from the full-precision metrics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "part2-aligned" / "cpu"
REPORT_PATH = ROOT / "docs" / "part2_sp4lp_katz_aligned.md"
DATASETS = ("cora", "citeseer")
SETTINGS = {"observed": 0.0, "missing40": 0.4}
MODELS = (
    "katz_h2",
    "katz_h3",
    "katz",
    "katz_decay50",
    "sp4lp",
    "sp4lp_endpoints",
    "sp4lp_featureless",
)
VARIANTS = (
    "clean",
    "rf@10%",
    "dice@10%",
    "ea@10%",
    "rf@25%",
    "dice@25%",
    "ea@25%",
    "rf@50%",
    "dice@50%",
    "ea@50%",
)
FOCUS_VARIANTS = ("clean", "rf@50%", "dice@50%", "ea@50%")


def load_results() -> dict[tuple[str, str], dict]:
    data = {}
    for dataset in DATASETS:
        for setting in SETTINGS:
            path = RESULT_ROOT / f"robustness_{dataset}_{setting}_cpu.json"
            with path.open(encoding="utf-8") as handle:
                data[(dataset, setting)] = json.load(handle)
    return data


def run_map(result: dict) -> dict[tuple[str, int, str], dict]:
    return {(row["model"], int(row["seed"]), row["variant"]): row for row in result["runs"]}


def validate(data: dict[tuple[str, str], dict]) -> list[str]:
    checks: list[str] = []
    expected_keys = {
        (model, seed, variant)
        for model in MODELS
        for seed in (0, 1, 2)
        for variant in VARIANTS
    }
    for (dataset, setting), result in data.items():
        assert result["dataset"] == dataset
        assert tuple(result["models"]) == MODELS
        assert tuple(result["seeds"]) == (0, 1, 2)
        assert tuple(result["variants"]) == VARIANTS
        assert not result["skipped"]
        assert result["protocol"]["device"] == "cpu"
        assert result["protocol"]["num_pos"] == 500
        assert result["protocol"]["num_neg"] == 500
        assert math.isclose(
            float(result["protocol"]["feature_missing_rate"]), SETTINGS[setting]
        )
        rows = run_map(result)
        assert len(result["runs"]) == 210
        assert set(rows) == expected_keys
        for row in result["runs"]:
            assert all(math.isfinite(float(row["metrics"][name])) for name in ("auc", "ap", "hit@100"))
        for model in MODELS:
            for variant in VARIANTS:
                assert result["summary"][model][variant]["auc"]["n"] == 3
        checks.append(f"{dataset}/{setting}: 210/210 cells, finite metrics, 3 seeds")

    for dataset in DATASETS:
        observed = run_map(data[(dataset, "observed")])
        missing = run_map(data[(dataset, "missing40")])
        assert data[(dataset, "observed")]["attack_stats"] == data[(dataset, "missing40")]["attack_stats"]
        for key in expected_keys:
            if key[0].startswith("katz") or key[0] == "sp4lp_featureless":
                assert observed[key]["metrics"] == missing[key]["metrics"]
        checks.append(
            f"{dataset}: attack metadata shared; Katz and featureless are exactly feature-mask invariant"
        )
    assert sum(len(result["runs"]) for result in data.values()) == 840
    checks.append("total: 840/840 distinct model/seed/condition cells")
    return checks


def auc(result: dict, model: str, variant: str) -> tuple[float, float, float]:
    cell = result["summary"][model][variant]["auc"]
    return float(cell["mean"]), float(cell["std"]), float(cell.get("drop_mean", 0.0))


def paired_delta(result: dict, left: str, right: str, variant: str) -> tuple[float, float]:
    rows = run_map(result)
    values = [
        rows[(left, seed, variant)]["metrics"]["auc"]
        - rows[(right, seed, variant)]["metrics"]["auc"]
        for seed in (0, 1, 2)
    ]
    mean = sum(values) / len(values)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, sample_std


def write_csvs(data: dict[tuple[str, str], dict]) -> None:
    long_path = RESULT_ROOT / "part2_aligned_auc.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dataset", "attributes", "model", "variant", "auc_mean", "auc_std", "drop_from_clean"]
        )
        for (dataset, setting), result in data.items():
            for model in MODELS:
                for variant in VARIANTS:
                    mean, std, drop = auc(result, model, variant)
                    writer.writerow([dataset, setting, model, variant, mean, std, drop])

    ablation_path = RESULT_ROOT / "part2_aligned_ablations.csv"
    comparisons = (
        ("path_branch", "sp4lp", "sp4lp_endpoints"),
        ("raw_attributes", "sp4lp", "sp4lp_featureless"),
        ("katz_add_hop3", "katz_h3", "katz_h2"),
        ("katz_add_hops4_5", "katz", "katz_h3"),
        ("katz_decay_085_minus_050", "katz", "katz_decay50"),
    )
    with ablation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dataset", "attributes", "component", "left", "right", "variant", "paired_auc_delta_mean", "paired_auc_delta_std"]
        )
        for (dataset, setting), result in data.items():
            for label, left, right in comparisons:
                for variant in VARIANTS:
                    mean, std = paired_delta(result, left, right, variant)
                    writer.writerow([dataset, setting, label, left, right, variant, mean, std])


def metric_cell(result: dict, model: str, variant: str, show_drop: bool = True) -> str:
    mean, std, drop = auc(result, model, variant)
    if variant == "clean" or not show_drop:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f} ± {std:.3f} (−{drop:.3f})"


def result_table(result: dict) -> str:
    lines = [
        "| Model | Clean | RF 50% | DICE 50% | EA 50% |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("sp4lp", "sp4lp_endpoints", "sp4lp_featureless", "katz", "katz_h3", "katz_h2"):
        cells = [metric_cell(result, model, variant) for variant in FOCUS_VARIANTS]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(data: dict[tuple[str, str], dict], comparison: tuple[str, str]) -> str:
    left, right = comparison
    lines = [
        f"| Dataset / attributes | Clean | RF 50% | DICE 50% | EA 50% |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for setting in SETTINGS:
            result = data[(dataset, setting)]
            cells = [paired_delta(result, left, right, variant)[0] for variant in FOCUS_VARIANTS]
            lines.append(
                f"| {dataset} / {setting} | " + " | ".join(f"{value:+.3f}" for value in cells) + " |"
            )
    return "\n".join(lines)


def runtime_summary(result: dict, model: str) -> tuple[float, float]:
    values = [float(row["fit_seconds"]) for row in result["runs"] if row["model"] == model]
    return sum(values) / len(values), max(values)


def write_report(data: dict[tuple[str, str], dict], checks: list[str]) -> None:
    sections: list[str] = []
    sections.append("# Part 2：SP4LP 与 Katz 对齐鲁棒性实验\n")
    sections.append(
        "本研究在团队 Line C 的统一 benchmark 上重做两个互补模型：可训练的局部最短路径模型 "
        "SP4LP，以及免训练的全局衰减路径计数 Katz。四套主实验已经完成：Cora/Citeseer × 完整属性/固定 "
        "40% 整节点属性缺失；每套包含 3 个 seed、clean 与 RF/DICE/近似 EA 的 10%/25%/50% 攻击。"
    )
    sections.append("\n## 1. 对齐协议与证据边界\n")
    sections.append(
        "- 数据集为 Cora、Citeseer，seed 为 0/1/2。每个 seed 共享 edge split、最多 2,000 个训练 pair、"
        "500 正 + 500 负测试 pair、攻击图和候选 pair。\n"
        "- 攻击是 evasion：模型只在 clean graph 上训练一次，测试时冻结参数；结构攻击后重新构造 GCN 输入和"
        "最短路径。RF/DICE/EA 均使用 10%/25%/50% edge-flip budget。\n"
        "- SP4LP 最多训练 300 epoch，学习率 0.001，以 clean validation AUROC 每 5 epoch 检查、patience=6。"
        "Katz 不训练。\n"
        "- 本机无 CUDA，本结果使用同一个 `cpu` budget profile 但实际在 CPU 执行；团队 Line C 数字在 CUDA "
        "执行。因此可比较协议和趋势，但不能把两批数无标记地合并成同硬件显著性结论。\n"
        "- SP4LP 是宽度 64 的仓库适配版，不是论文 Cora 宽度 256 + HeaRT negatives 的完整复现；EA 是仓库"
        "的 embedding-surrogate 近似攻击，不是外部库的原版攻击。"
    )
    sections.append("\n自动审计：\n\n" + "\n".join(f"- {item}" for item in checks))

    sections.append("\n## 2. 两个模型到底在学什么\n")
    sections.append(
        "### SP4LP\n\n"
        "SP4LP 把 link prediction 拆成两条信息流。第一条用一层 GCN 把节点属性与一跳邻居聚合成 endpoint "
        "embedding；第二条在移除待预测边后，用 BFS 找到两个端点之间的精确最短路径，把路径上的节点 embedding "
        "作为序列交给一层 Transformer。最终把 endpoint 对表示与 path 表示拼接，经 MLP 输出边分数。它的偏置是："
        "一条边是否存在，既取决于端点自身/邻域，也取决于连接二者的最短结构证据。\n\n"
        "`sp4lp_endpoints` 删除整个 path tokenizer/Transformer，只保留同一个 GCN 与 endpoint MLP，用来回答“显式最短"
        "路径序列是否真的有用”；`sp4lp_featureless` 把原始属性替换为常数 1，但保留 GCN、路径与 Transformer，用来"
        "回答“提升来自节点属性还是纯结构”。两者都重新独立训练，不是把完整 checkpoint 临时关掉组件。\n\n"
        "需要留意：路径构造会隐藏 query edge；但与常规 full-graph link prediction 一样，训练正边仍存在于 GCN 的训练图"
        "中。这个设定与团队 harness 一致，却不等价于严格 inductive 或全边遮蔽训练。"
    )
    sections.append(
        "\n### Katz\n\n"
        "Katz 对候选节点对 `(u,v)` 计算 `Σ_l β^l (A^l)[u,v]`：`A^l` 是长度为 l 的 walk 数，β 让长 walk "
        "递减。这里 query edge 已从测试图移除，所以 H2 基本就是缩放后的 common-neighbour 证据；H3 加入三跳 walk，"
        "H5 再加入四、五跳 walk。β 取 `0.85/max_degree`，保证对应无限级数收敛；`katz_decay50` 仅把分子改为 "
        "0.50。Katz 完全不读属性、没有学习参数，并在每个 attacked graph 上重新计算，因此测的是结构本身被扰动后的"
        "可恢复性，而不是冻结神经网络对 distribution shift 的适应。"
    )

    sections.append("\n## 3. 主结果\n")
    sections.append(
        "表中括号是相对本模型 clean AUC 的绝对下降；均值 ± sample standard deviation 来自 3 个 seed。"
    )
    for dataset in DATASETS:
        for setting in SETTINGS:
            title = "完整属性" if setting == "observed" else "40% 整节点属性缺失"
            sections.append(f"\n### {dataset}: {title}\n\n{result_table(data[(dataset, setting)])}")

    sections.append("\n## 4. SP4LP 组件消融\n")
    sections.append(
        "下面是严格按 seed 配对后的 AUC 差：正数表示完整 SP4LP 更好。\n\n"
        "### 显式最短路径分支：`sp4lp − sp4lp_endpoints`\n\n"
        + delta_table(data, ("sp4lp", "sp4lp_endpoints"))
    )
    sections.append(
        "\n结论：路径分支在 Cora clean 和 EA 下有稳定可见增益，属性缺失时增益更大；但在 Citeseer 完整属性下"
        "反而平均更差。它不是普适增益组件，当前较小预算下可能因为路径序列较噪或 Transformer 优化方差而输给更简单"
        "的 endpoint head。不能只用 Cora 宣称路径模块全面有效。"
    )
    sections.append(
        "\n### 原始节点属性：`sp4lp − sp4lp_featureless`\n\n"
        + delta_table(data, ("sp4lp", "sp4lp_featureless"))
    )
    sections.append(
        "\n结论：完整属性在 Citeseer 尤其重要，并显著缓冲 50% RF/DICE；Cora 也在强攻击下受益。可是把 40% 节点"
        "特征直接置零后，full SP4LP 的 clean AUC 在两数据集都低于 featureless，说明残缺特征可成为噪声，简单地保留"
        "零行不一定胜过明确不用属性。Citeseer 强攻击下残余属性仍有帮助，说明“是否用属性”与“攻击强度”存在交互。"
    )

    sections.append("\n## 5. Katz 组件消融\n")
    katz_lines = [
        "| Dataset | H2 clean | H3 clean | H5 clean | H3−H2 | H5−H3 | H5(0.85)−H5(0.50) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        result = data[(dataset, "observed")]
        h2 = auc(result, "katz_h2", "clean")[0]
        h3 = auc(result, "katz_h3", "clean")[0]
        h5 = auc(result, "katz", "clean")[0]
        d32 = paired_delta(result, "katz_h3", "katz_h2", "clean")[0]
        d53 = paired_delta(result, "katz", "katz_h3", "clean")[0]
        decay = paired_delta(result, "katz", "katz_decay50", "clean")[0]
        katz_lines.append(
            f"| {dataset} | {h2:.3f} | {h3:.3f} | {h5:.3f} | {d32:+.3f} | {d53:+.3f} | {decay:+.4f} |"
        )
    sections.append("\n" + "\n".join(katz_lines))
    sections.append(
        "\n结论：主要收益来自加入更长路径，而不是在 0.50–0.85 范围内调衰减。H2→H3 在 Cora/Citeseer 分别"
        "增加约 0.078/0.089，H3→H5 再增加约 0.044/0.020；两种 β 的排序几乎完全相同，所以 AUROC 基本一致。"
        "这不表示 β 数值无影响，而是本候选集上它没有改变多少 pairwise ranking。"
    )

    sections.append("\n## 6. 与团队 benchmark 的关系\n")
    sections.append(
        "本分支从 Line C benchmark commit 派生，复用了相同数据拆分、缺失掩码规则、攻击生成、候选 pair、预算和 JSON "
        "schema，因此实验设计已经对齐。团队 CUDA 表中 Cora/Citeseer 完整属性的 SEAL clean AUC 为 0.866/0.861；"
        "本次 CPU SP4LP 为 0.919/0.913，Katz H5 为 0.851/0.776。强攻击下 SP4LP 的绝对 AUC 也普遍较高。"
        "不过 SP4LP 与团队学习模型不在同一设备上训练，因此这些数字只作为同协议上下文；正式总表应增加 hardware/provenance "
        "列，或让团队在同一 CUDA 环境重跑本分支后再做严格排序。"
    )

    sections.append("\n## 7. 整体发现与下一步\n")
    cora_runtime = runtime_summary(data[("cora", "observed")], "sp4lp")
    cite_runtime = runtime_summary(data[("citeseer", "observed")], "sp4lp")
    sections.append(
        "1. **效果上**：SP4LP 是这两个模型中更强的主模型；Katz H5 是稳定、透明、免训练的结构基线。\n"
        "2. **鲁棒性上**：EA 对本次 SP4LP/Katz 通常比 RF/DICE 温和；应同时报告 attacked AUC 和 clean drop，不能只看下降量。\n"
        "3. **组件上**：路径深度对 Katz 明确重要；SP4LP 的显式路径分支具有数据集依赖；完整节点属性在强结构攻击下尤其有价值，"
        "但简单零填充的残缺属性会伤害 clean 表现。\n"
        "4. **成本上**：Katz 每 seed 约 0.1–0.2 秒；完整 SP4LP 单 seed 的训练均值/最大值约为 "
        f"Cora {cora_runtime[0]:.1f}/{cora_runtime[1]:.1f} 秒、Citeseer {cite_runtime[0]:.1f}/{cite_runtime[1]:.1f} 秒。"
        "本次是正常训练并受 early stopping 控制，不是论文级宽度和完整负采样预算，所以不应与“跑一天”的 full-paper run 等同。\n"
        "5. **若要继续强化**：优先在团队 CUDA 环境原样重跑 3 个 SP4LP 配置（full/endpoints/featureless，必要时另加宽度 256），"
        "并增加多重 missingness 比例或更合理的 feature imputation；不要再无目的重复 Katz。"
    )

    sections.append("\n## 8. 可复现产物\n")
    sections.append(
        "- 原始结果：`results/part2-aligned/cpu/robustness_{dataset}_{observed|missing40}_cpu.{json,md}`\n"
        "- 全条件 tidy 表：`results/part2-aligned/cpu/part2_aligned_auc.csv`\n"
        "- 配对消融表：`results/part2-aligned/cpu/part2_aligned_ablations.csv`\n"
        "- 重新生成本报告：`python scripts/analyze_part2_aligned.py`"
    )
    REPORT_PATH.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    data = load_results()
    checks = validate(data)
    write_csvs(data)
    write_report(data, checks)
    print("validated:")
    for check in checks:
        print(f"  - {check}")
    print(f"wrote {RESULT_ROOT / 'part2_aligned_auc.csv'}")
    print(f"wrote {RESULT_ROOT / 'part2_aligned_ablations.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
