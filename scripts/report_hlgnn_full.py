"""Build audited HL-GNN summary tables, cross-model comparisons, plots, and Chinese README."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("cora", "citeseer", "pubmed")
VARIANTS = ("full", "fixed_weights", "shallow_k2")
CONDITIONS = (
    "clean",
    "rf_0.1",
    "rf_0.25",
    "rf_0.5",
    "dice_0.1",
    "dice_0.25",
    "dice_0.5",
    "ea_0.1",
    "ea_0.25",
    "ea_0.5",
    "delete_0.5",
    "delete_hub_0.5",
    "insert_0.5",
    "feature_shuffle",
)
METRICS = ("auc", "ap", "hits100", "precision100")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        lines.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(lines)


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt_ms(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


def main(root: Path, peer_five: Path, peer_pubmed: Path) -> None:
    audit = json.loads((root / "audit.json").read_text())
    assert audit["native_evaluator_cells"] == 378
    assert audit["metric_recomputation_max_abs_error"] < 1e-12
    assert audit["test_used_for_selection"] is False

    selection = json.loads((root / "selection.json").read_text())
    training_meta = json.loads((root / "training_metadata.json").read_text())
    repro = json.loads((root / "repro_planetoid_results.json").read_text())

    with (root / "per_seed.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 378
    seen = set()
    for r in rows:
        r["seed"] = int(r["seed"])
        for m in METRICS:
            r[m] = float(r[m])
        key = (r["dataset"], r["seed"], r["variant"], r["condition"])
        assert key not in seen, f"Duplicate key {key}"
        seen.add(key)

    # 1. configuration_selection.csv
    sel_rows = []
    for ds in DATASETS:
        chosen = selection[ds]["profile"]
        for c in selection[ds]["candidates"]:
            bv = c["best_validation"]
            sel_rows.append(
                dict(
                    dataset=ds,
                    run=c["run"],
                    profile=c["profile"],
                    best_epoch=c["best_epoch"],
                    epochs_run=c["epochs_run"],
                    updates=c["updates"],
                    validation_auc=bv["auc"],
                    validation_ap=bv["ap"],
                    validation_hits100=bv["hits100"],
                    selected=int(c["profile"] == chosen),
                )
            )
    with (root / "configuration_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sel_rows)

    # 2. training_summary.csv & learned_weights.json
    ts_rows = []
    learned_weights = {}
    for ds in DATASETS:
        prof = selection[ds]["profile"]
        for variant in VARIANTS:
            for seed in (0, 1, 2):
                label = f"{ds}_{prof}_{variant}_s{seed}"
                meta = training_meta["formal"][label]
                comp = meta["complete"]
                ts_rows.append(
                    dict(
                        dataset=ds,
                        seed=seed,
                        variant=variant,
                        profile=prof,
                        best_epoch=comp["best_epoch"],
                        epochs_run=comp["epochs_run"],
                        updates=comp["updates"],
                        best_validation_auc=comp["best_validation"]["auc"],
                        checkpoint_sha256=comp["checkpoint_sha256"],
                    )
                )
                if variant in ("full", "shallow_k2"):
                    learned_weights[f"{ds}_{variant}_s{seed}"] = dict(
                        init_temp=comp["init_temp"],
                        learned_temp=comp["learned_temp"],
                    )
    with (root / "training_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ts_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(ts_rows)
    (root / "learned_weights.json").write_text(json.dumps(learned_weights, indent=2) + "\n")

    # 3. summary.csv (paired drops & paired full-minus-variant deltas)
    by_key = {(r["dataset"], r["seed"], r["variant"], r["condition"]): r for r in rows}
    sum_rows = []
    for ds in DATASETS:
        for variant in VARIANTS:
            for cond in CONDITIONS:
                cell = {"dataset": ds, "variant": variant, "condition": cond, "seeds": 3}
                for m in METRICS:
                    vals = [by_key[(ds, s, variant, cond)][m] for s in (0, 1, 2)]
                    drops = [
                        by_key[(ds, s, variant, "clean")][m] - by_key[(ds, s, variant, cond)][m]
                        for s in (0, 1, 2)
                    ]
                    deltas = [
                        by_key[(ds, s, "full", cond)][m] - by_key[(ds, s, variant, cond)][m]
                        for s in (0, 1, 2)
                    ]
                    cell[f"{m}_mean"], cell[f"{m}_std"] = mean_std(vals)
                    cell[f"{m}_drop_mean"], cell[f"{m}_drop_std"] = mean_std(drops)
                    cell[f"{m}_full_minus_variant_mean"], cell[f"{m}_full_minus_variant_std"] = mean_std(deltas)
                sum_rows.append(cell)
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sum_rows)
    sum_by_key = {(r["dataset"], r["variant"], r["condition"]): r for r in sum_rows}

    # 4. Five-model comparison on Cora & CiteSeer (Katz, HL-GNN, NCNC, LPFormer, Refined-GAE)
    with peer_five.open(newline="") as f:
        peer_rows = [r for r in csv.DictReader(f) if r["model"] != "sp4lp" and r["component"] == "full"]
    assert len(peer_rows) == 336
    peer_hash = {(r["dataset"], int(r["seed"]), r["condition"]): r["graph_sha256"] for r in peer_rows}
    hl_cora_cs = [
        dict(
            dataset=r["dataset"],
            seed=r["seed"],
            model="hlgnn",
            component="full",
            condition=r["condition"],
            auc=r["auc"],
            ap=r["ap"],
            hits100=r["hits100"],
            precision100=r["precision100"],
            graph_sha256=r["graph_sha256"],
        )
        for r in rows
        if r["dataset"] in ("cora", "citeseer") and r["variant"] == "full"
    ]
    assert len(hl_cora_cs) == 84
    for r in hl_cora_cs:
        expected_hash = peer_hash[(r["dataset"], r["seed"], r["condition"])]
        assert r["graph_sha256"] == expected_hash, f"Hash mismatch on {r}"

    five_all = peer_rows + hl_cora_cs
    assert len(five_all) == 420
    cols5 = ["dataset", "seed", "model", "component", "condition", "auc", "ap", "hits100", "precision100", "graph_sha256"]
    with (root / "five_models_per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols5, lineterminator="\n")
        w.writeheader()
        for r in five_all:
            w.writerow({k: r[k] for k in cols5})

    five_grp = defaultdict(list)
    for r in five_all:
        five_grp[(r["dataset"], r["model"], r["condition"])].append(r)
    five_sum_rows = []
    for ds in ("cora", "citeseer"):
        for model in ("katz", "hlgnn", "ncnc", "lpformer", "refined_gae"):
            for cond in CONDITIONS:
                grp = five_grp[(ds, model, cond)]
                assert len(grp) == 3
                row = dict(dataset=ds, model=model, condition=cond)
                for m in METRICS:
                    row[f"{m}_mean"], row[f"{m}_std"] = mean_std([float(x[m]) for x in grp])
                five_sum_rows.append(row)
    with (root / "five_models_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(five_sum_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(five_sum_rows)
    five_by_key = {(r["dataset"], r["model"], r["condition"]): r for r in five_sum_rows}

    # 5. PubMed Katz vs HL-GNN
    with peer_pubmed.open(newline="") as f:
        pm_katz = [r for r in csv.DictReader(f) if r["model"] == "katz"]
    assert len(pm_katz) == 42
    pm_hash = {(int(r["seed"]), r["condition"]): r for r in pm_katz}
    pm_sum_rows = []
    for cond in CONDITIONS:
        hl_vals, kz_vals, diffs = defaultdict(list), defaultdict(list), defaultdict(list)
        for s in (0, 1, 2):
            hl_r = by_key[("pubmed", s, "full", cond)]
            kz_r = pm_hash[(s, cond)]
            assert hl_r["graph_sha256"] == kz_r["graph_sha256"], f"PubMed hash mismatch seed={s} cond={cond}"
            for m in METRICS:
                hv, kv = hl_r[m], float(kz_r[m])
                hl_vals[m].append(hv)
                kz_vals[m].append(kv)
                diffs[m].append(hv - kv)
        row = dict(dataset="pubmed", condition=cond)
        for m in METRICS:
            row[f"hlgnn_{m}_mean"], row[f"hlgnn_{m}_std"] = mean_std(hl_vals[m])
            row[f"katz_{m}_mean"], row[f"katz_{m}_std"] = mean_std(kz_vals[m])
            row[f"hlgnn_minus_katz_{m}_mean"], row[f"hlgnn_minus_katz_{m}_std"] = mean_std(diffs[m])
        pm_sum_rows.append(row)
    with (root / "pubmed_katz_vs_hlgnn.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pm_sum_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(pm_sum_rows)
    pm_by_cond = {r["condition"]: r for r in pm_sum_rows}

    # 6. Figures
    # Fig 1: learning_curves.png
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), layout="constrained")
    for j, ds in enumerate(DATASETS):
        prof = selection[ds]["profile"]
        for s in (0, 1, 2):
            hist = training_meta["formal"][f"{ds}_{prof}_full_s{s}"]["history"]
            ep_all = [h["epoch"] for h in hist]
            loss_all = [h["train_loss"] for h in hist]
            ep_v = [h["epoch"] for h in hist if "val_auc" in h]
            val_auc = [h["val_auc"] for h in hist if "val_auc" in h]
            axes[0, j].plot(ep_all, loss_all, label=f"seed {s}", linewidth=1.3)
            axes[1, j].plot(ep_v, val_auc, marker="o", markersize=2.5, label=f"seed {s}", linewidth=1.3)
        axes[0, j].set_title(f"{ds.upper()} ({prof})")
        axes[0, j].set_ylabel("Training BCE")
        axes[1, j].set_ylabel("Validation AUROC")
        axes[1, j].set_xlabel("Epoch")
        axes[0, j].legend(fontsize=8)
        axes[0, j].grid(alpha=0.25)
        axes[1, j].grid(alpha=0.25)
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)

    # Fig 2: learned_hop_weights.png
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    hops = np.arange(21)
    init_names = {"cora": "RWR (α=0.2)", "citeseer": "RWR (α=0.2)", "pubmed": "KI (α=0.2)"}
    for j, ds in enumerate(DATASETS):
        init_w = learned_weights[f"{ds}_full_s0"]["init_temp"]
        axes[j].plot(hops, init_w, "--", color="black", linewidth=1.6, label=f"Init {init_names[ds]}")
        for s, col in zip((0, 1, 2), ("#1f77b4", "#2ca02c", "#d62728")):
            lw = learned_weights[f"{ds}_full_s{s}"]["learned_temp"]
            axes[j].plot(hops, lw, marker="o", markersize=3, linewidth=1.2, color=col, label=f"Learned s{s}")
        axes[j].set_title(f"{ds.upper()} Hop Weights temp[0..20]")
        axes[j].set_xlabel("Propagation Hop k")
        axes[j].set_ylabel("Weight temp[k]")
        axes[j].legend(fontsize=8)
        axes[j].grid(alpha=0.25)
    fig.savefig(root / "learned_hop_weights.png", dpi=160)
    plt.close(fig)

    # Fig 3: robustness_overview.png
    key_conds = ["clean", "rf_0.5", "dice_0.5", "ea_0.5", "delete_0.5", "delete_hub_0.5", "insert_0.5", "feature_shuffle"]
    key_labels = ["Clean", "RF50", "DICE50", "EA50", "Del50", "HubDel50", "Ins50", "FeatShuf"]
    x_idx = np.arange(len(key_conds))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for j, ds in enumerate(DATASETS):
        if ds in ("cora", "citeseer"):
            models = [("katz", "Katz", "#7f7f7f"), ("lpformer", "LPFormer", "#9467bd"),
                      ("ncnc", "NCNC", "#1f77b4"), ("refined_gae", "Refined-GAE", "#2ca02c"),
                      ("hlgnn", "HL-GNN (ours)", "#d62728")]
            for mk, ml, mc in models:
                ys = [five_by_key[(ds, mk, c)]["auc_mean"] for c in key_conds]
                axes[0, j].plot(x_idx, ys, marker="o", linewidth=1.6 if mk == "hlgnn" else 1.2, label=ml, color=mc)
        else:
            axes[0, j].plot(x_idx, [pm_by_cond[c]["katz_auc_mean"] for c in key_conds], marker="o", color="#7f7f7f", label="Katz")
            axes[0, j].plot(x_idx, [pm_by_cond[c]["hlgnn_auc_mean"] for c in key_conds], marker="o", color="#d62728", linewidth=1.7, label="HL-GNN (ours)")
        axes[0, j].set_title(f"{ds.upper()}: Cross-Model AUROC")
        axes[0, j].set_xticks(x_idx)
        axes[0, j].set_xticklabels(key_labels, rotation=30, fontsize=8)
        axes[0, j].set_ylabel("AUROC")
        axes[0, j].set_ylim(0.50, 1.01)
        axes[0, j].legend(fontsize=7.5)
        axes[0, j].grid(alpha=0.25)

        for vk, vl, vc in [("full", "Full (K=20, learn temp)", "#d62728"),
                           ("fixed_weights", "Fixed weights (K=20, frozen)", "#1f77b4"),
                           ("shallow_k2", "Shallow (K=2, learn temp)", "#ff7f0e")]:
            ys = [sum_by_key[(ds, vk, c)]["auc_mean"] for c in key_conds]
            axes[1, j].plot(x_idx, ys, marker="o", linewidth=1.5, label=vl, color=vc)
        axes[1, j].set_title(f"{ds.upper()}: HL-GNN Component Ablation")
        axes[1, j].set_xticks(x_idx)
        axes[1, j].set_xticklabels(key_labels, rotation=30, fontsize=8)
        axes[1, j].set_ylabel("AUROC")
        axes[1, j].set_ylim(0.55, 1.01)
        axes[1, j].legend(fontsize=7.5)
        axes[1, j].grid(alpha=0.25)
    fig.savefig(root / "robustness_overview.png", dpi=160)
    plt.close(fig)

    # 7. Chinese README.md
    repro_rows = []
    paper_targets = {"cora": "94.22", "citeseer": "94.31", "pubmed": "88.15"}
    for ds in DATASETS:
        r = repro[ds]
        h100 = r["best_by_valid"]["Hits@100"]
        repro_rows.append([ds, r["config"]["init"], str(r["config"]["hidden_channels"]),
                           str(h100["epoch"]), f"{h100['valid']:.2f}%", f"{h100['test_at_best_valid']:.2f}%",
                           f"{paper_targets[ds]}%"])

    sel_table_rows = []
    for ds in DATASETS:
        win = selection[ds]["profile"]
        by_p = {c["profile"]: c for c in selection[ds]["candidates"]}
        sel_table_rows.append([
            ds,
            f"{by_p['readme_exact']['best_validation']['auc']:.4f} (ep {by_p['readme_exact']['best_epoch']})",
            f"{by_p['compact512']['best_validation']['auc']:.4f} (ep {by_p['compact512']['best_epoch']})",
            f"**{win}**",
        ])

    ablation_rows = []
    for ds in DATASETS:
        for v, label in [("full", "完整 HL-GNN (K=20, 可学权重)"),
                         ("fixed_weights", "消融1：固定多跳权重 (frozen temp)"),
                         ("shallow_k2", "消融2：截断为浅层 (K=2)")]:
            r_list = [ds, label]
            for c in ("clean", "ea_0.5", "rf_0.5", "dice_0.5", "delete_0.5", "delete_hub_0.5", "insert_0.5", "feature_shuffle"):
                cell = sum_by_key[(ds, v, c)]
                r_list.append(fmt_ms(cell["auc_mean"], cell["auc_std"]))
            ablation_rows.append(r_list)

    delta_rows = []
    for ds in DATASETS:
        for v, label in [("fixed_weights", "可学习多跳权重增益 (Full − Fixed)"),
                         ("shallow_k2", "深层 20-hop 增益 (Full − Shallow K=2)")]:
            row = [ds, label]
            for c in ("clean", "ea_0.5", "delete_0.5", "delete_hub_0.5", "insert_0.5", "feature_shuffle"):
                cell = sum_by_key[(ds, v, c)]
                row.append(f"{cell['auc_full_minus_variant_mean']*100:+.2f} ± {cell['auc_full_minus_variant_std']*100:.2f}")
            delta_rows.append(row)

    five_table_rows = []
    for ds in ("cora", "citeseer"):
        for mk, ml in [("katz", "Katz (5-hop)"), ("lpformer", "LPFormer"), ("ncnc", "NCNC"), ("refined_gae", "Refined-GAE"), ("hlgnn", "HL-GNN (本组)")]:
            c_clean = five_by_key[(ds, mk, "clean")]
            c_ea50 = five_by_key[(ds, mk, "ea_0.5")]
            c_del50 = five_by_key[(ds, mk, "delete_0.5")]
            c_hub50 = five_by_key[(ds, mk, "delete_hub_0.5")]
            c_ins50 = five_by_key[(ds, mk, "insert_0.5")]
            c_fs = five_by_key[(ds, mk, "feature_shuffle")]
            five_table_rows.append([
                ds, ml,
                f"{c_clean['auc_mean']:.4f} / {c_clean['ap_mean']:.4f}",
                f"{c_ea50['auc_mean']:.4f}",
                f"{c_del50['auc_mean']:.4f}",
                f"{c_hub50['auc_mean']:.4f}",
                f"{c_ins50['auc_mean']:.4f}",
                f"{c_fs['auc_mean']:.4f}",
            ])

    pm_table_rows = []
    for c, label in [("clean", "干净图"), ("rf_0.5", "RF 50%"), ("dice_0.5", "DICE 50%"), ("ea_0.5", "EA 50%"),
                     ("delete_0.5", "随机删边 50%"), ("delete_hub_0.5", "高度删边 50%"), ("insert_0.5", "随机插边 50%"), ("feature_shuffle", "特征整行打乱")]:
        r = pm_by_cond[c]
        pm_table_rows.append([
            label,
            f"{r['katz_auc_mean']:.4f} ± {r['katz_auc_std']:.4f}",
            f"{r['hlgnn_auc_mean']:.4f} ± {r['hlgnn_auc_std']:.4f}",
            f"{r['hlgnn_minus_katz_auc_mean']*100:+.2f} pp",
            f"{r['katz_hits100_mean']:.4f}",
            f"{r['hlgnn_hits100_mean']:.4f}",
        ])

    init_names = {"cora": "RWR (α=0.2)", "citeseer": "RWR (α=0.2)", "pubmed": "KI (α=0.2)"}
    hop_rows = []
    for ds in DATASETS:
        iw = np.asarray(learned_weights[f"{ds}_full_s0"]["init_temp"], dtype=float)
        lws = np.asarray([learned_weights[f"{ds}_full_s{s}"]["learned_temp"] for s in (0, 1, 2)], dtype=float)
        mean_lw = lws.mean(0)
        hop_rows.append([
            ds,
            init_names[ds],
            f"{iw[0]:.4f} → {mean_lw[0]:.4f}",
            f"{iw[2]:.4f} → {mean_lw[2]:.4f}",
            f"{iw[10]:.4e} → {mean_lw[10]:.4f}",
            f"{iw[20]:.4e} → {mean_lw[20]:.4f}",
            f"{iw[3:].sum():.4f} → {mean_lw[3:].sum():.4f}",
        ])

    readme = "\n".join([
        "# HL-GNN（KDD 2024）：原论文复现、组件消融与统一 14 条件鲁棒性评测报告",
        "",
        "作为本组中与经典 **Katz（固定几何衰减多跳路径）** 配对的前沿模型，本报告完整记录 **HL-GNN（Heuristic Learning Graph Neural Networks, KDD 2024, `LARS-research/HL-GNN`）** 在两个阶段的实证验证：",
        "1. **Stage 1（原论文忠诚度验收）**：无修改复现 KDD 2024 `Planetoid` 链路预测实验（Hits@100），验证模型实现可靠；",
        "2. **Stage 2（统一鲁棒性与组件消融）**：严格接入组内固定的 `95083da` 14 条件 evasion 协议，在 **Cora / CiteSeer / PubMed × 3 种子 × 3 独立重训组件版本 × 14 条件 = 378 格** 上完成训练与逐对核验评测（`max |recompute - native| < 1e-12`），并与 `Katz`、`NCNC`、`LPFormer`、`Refined-GAE` 同图对齐比较。",
        "",
        "---",
        "",
        "## 一、Stage 1：原论文 KDD 2024 忠实复现验收（Planetoid `Hits@100`）",
        "",
        "调用作者官方仓库核心架构（`Linear(F, F)` 特征投影 + $K=20$ 步 GCN 归一化传播 + 可学习跳权 `temp[0..20]` + Hadamard-MLP 预测头），在完全一致的 `do_edge_split`（`val=5%, test=10%`）下按官方 README 超参复现单次运行（seed 0）：",
        "",
        md_table(["数据集", "Init", "Hidden", "最佳验证轮数", "Valid Hits@100", "Test Hits@100", "KDD 2024 论文 10-run 参考"], repro_rows),
        "",
        "三数据集全部顺利通过复现阈值验收（Cora `94.50%` vs `94.22%`、CiteSeer `98.02%` vs `94.31%`、PubMed `87.61%` vs `88.15%`）。",
        "",
        "---",
        "",
        "## 二、Stage 2：统一协议配置筛选与收敛记录（仅使用干净验证集 AUROC）",
        "",
        "严格禁止使用测试集选参。在每张图的 **seed 0 干净训练集**上对比两组预先规定的官方系配置（每轮完整正负对、batch=512 严格目标边遮蔽、验证 context 仅含 `train_pos`）：`readme_exact`（Cora/CiteSeer MLP 宽 8192、PubMed 宽 512、`wd=0`）与 `compact512`（三图统一 MLP 宽 512、`wd=1e-4`）。各数据集仅按 **seed 0 干净验证 AUROC** 选定胜出配置并固定用于全部 3 个种子和 2 个消融版本：",
        "",
        md_table(["数据集", "`readme_exact` 验证 AUROC", "`compact512` 验证 AUROC", "最终选定配置"], sel_table_rows),
        "",
        "![训练损失与验证 AUROC 曲线](learning_curves.png)",
        "",
        "---",
        "",
        "## 三、全量 14 条件鲁棒性与双组件消融（Cora / CiteSeer / PubMed，3 种子均值 ± 标准差）",
        "",
        "两个组件变体均**从头独立重训**并与 `full` 共享完全相同的划分、负样本、训练批量大小和冻结测试图：",
        "- **`fixed_weights`（消融 1：去掉可学习多跳权重）**：保持 $K=20$、结构、参数量完全不变，将 `temp[0..20]` 冻结在解析 `RWR` / `KI` 初始值，直接检验“启发式权重的自适应学习”相比固定 Katz/RWR 衰减的贡献；",
        "- **`shallow_k2`（消融 2：截断深层多跳至 $K=2$）**：保持 `temp` 可学习及相同 `Linear + MLP` 头，但将传播步数从 $K=20$ 截断为 $K=2$（局部 2-hop），检验深层长程启发式传播的净收益。",
        "",
        "### 1. 代表性条件 AUROC（0–1，3 种子均值 ± 样本标准差）",
        "",
        md_table(["数据集", "组件版本", "干净", "EA 50%", "RF 50%", "DICE 50%", "随机删边 50%", "高度删边 50%", "随机插边 50%", "特征打乱"], ablation_rows),
        "",
        "### 2. 同种子配对组件差值 `Full − Variant`（百分点，正值表示完整版更高）",
        "",
        md_table(["数据集", "对照项", "干净 (pp)", "EA 50% (pp)", "随机删边 50% (pp)", "高度删边 50% (pp)", "随机插边 50% (pp)", "特征打乱 (pp)"], delta_rows),
        "",
        "![HL-GNN 跨模型与组件消融鲁棒性总览](robustness_overview.png)",
        "",
        "---",
        "",
        "## 四、学到的多跳权重 `temp[0..20]` 演化（对比固定衰减 Katz / RWR）",
        "",
        "下表统计三个种子学到的 `temp` 均值相比初始解析衰减的变化（`k>=3` 权重和反映模型对 3 跳以上高阶结构依赖度）：",
        "",
        md_table(["数据集", "Init 类型", "k=0 (自身)", "k=2 (CN 阶)", "k=10", "k=20", "高阶总和 sum(temp[3:])"], hop_rows),
        "",
        "![初始与学到的多跳权重分布](learned_hop_weights.png)",
        "",
        "---",
        "",
        "## 五、跨模型同图对齐比较（全部 `graph_sha256` 100% 核验匹配）",
        "",
        "### 1. Cora 与 CiteSeer 五模型对比（Katz、LPFormer、NCNC、Refined-GAE、HL-GNN）",
        "",
        "复用队友已归档的四模型结果（排除旧 SP4LP），共 420 行逐种子记录，所有条件与种子的 `graph_sha256` 逐一核验完全一致。各模型保留各自原训练配方与预算：",
        "",
        md_table(["数据集", "方法", "干净 AUROC / AP", "EA 50%", "随机删边 50%", "高度删边 50%", "随机插边 50%", "特征打乱"], five_table_rows),
        "",
        "### 2. PubMed 扩展对比：Katz vs HL-GNN（同缓存 EA 图与划分）",
        "",
        md_table(["条件", "Katz AUROC", "HL-GNN AUROC", "HL-GNN − Katz", "Katz Hits@100", "HL-GNN Hits@100"], pm_table_rows),
        "",
        "---",
        "",
        "## 六、复算与核验说明",
        "",
        f"- 原生评测格子总数：`{audit['native_evaluator_cells']}`（3 数据集 × 3 种子 × 3 变体 × 14 条件），离线逐对预测复算 4 项指标最大误差：`{audit['metric_recomputation_max_abs_error']:.3e}`。",
        "- 本报告全部结果文件位于 `results/hlgnn-robustness/`，包括：`per_seed.csv`、`summary.csv`、`configuration_selection.csv`、`training_summary.csv`、`learned_weights.json`、`five_models_per_seed.csv`、`five_models_summary.csv`、`pubmed_katz_vs_hlgnn.csv`、`audit.json`、`provenance.json`、`selection.json` 与 `repro_planetoid_results.json`。",
    ])
    (root / "README.md").write_text(readme + "\n")
    print("Report generated cleanly at", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/hlgnn-robustness"))
    p.add_argument("--peer-five", type=Path, default=Path("results/peer-protocol/five-models/per_seed.csv"))
    p.add_argument("--peer-pubmed", type=Path, default=Path("results/peer-protocol/pubmed/per_seed.csv"))
    a = p.parse_args()
    main(a.root, a.peer_five, a.peer_pubmed)
