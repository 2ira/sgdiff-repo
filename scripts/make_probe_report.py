#!/usr/bin/env python
"""Render a self-contained HTML report from the mechanism-probe artifacts.

Reads ``results/probe_<dataset>.json`` (written by ``scripts/run_probe.py``) and
emits one HTML file with no external dependencies: every chart is a styled table
or inline SVG, so it works offline and in a git diff.

    python scripts/make_probe_report.py            # -> results/probe_report.html
    python scripts/make_probe_report.py --out /tmp/report.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PROBES = [
    "del@50%",
    "del_hub@50%",
    "ins@50%",
    "ins_hub@50%",
    "ins_homo@50%",
    "ins_hetero@50%",
    "feature_shuffle",
]
PRETTY = {
    "clean": "clean",
    "del@50%": "del",
    "del_hub@50%": "del_hub",
    "ins@50%": "ins",
    "ins_hub@50%": "ins_hub",
    "ins_homo@50%": "ins_homo",
    "ins_hetero@50%": "ins_hetero",
    "feature_shuffle": "feat_shuffle",
}
HEUR_LABEL = {"deg": "度乘积", "cn": "共同邻居", "aa": "Adamic-Adar", "jaccard": "Jaccard", "featcos": "特征余弦"}


# --------------------------------------------------------------------------- #
# tiny formatting helpers
# --------------------------------------------------------------------------- #
def mean(entry) -> float | None:
    if not entry or not entry.get("n"):
        return None
    value = entry.get("mean")
    return float(value) if value is not None else None


def focal(value: float | None, lo: float, hi: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def auc_style(value: float | None) -> str:
    """Sequential teal: low = washed out, high = saturated."""
    t = focal(value, 0.40, 1.0)
    r = int(243 + (13 - 243) * t)
    g = int(250 + (118 - 250) * t)
    b = int(247 + (110 - 247) * t)
    text = "#ffffff" if t > 0.62 else "#0f172a"
    return f"background: rgb({r},{g},{b}); color: {text};"


def damage_style(value: float | None, scale: float = 0.24) -> str:
    """Diverging: positive (worse than clean) red, negative (better) blue."""
    if value is None:
        return "background:#f1f5f9;color:#94a3b8;"
    t = max(-1.0, min(1.0, value / scale))
    if t >= 0:
        alpha = 0.10 + 0.80 * t
        text = "#ffffff" if t > 0.55 else "#7f1d1d"
        return f"background: rgba(190,18,60,{alpha:.3f}); color: {text};"
    alpha = 0.10 + 0.80 * (-t)
    text = "#ffffff" if -t > 0.55 else "#1e3a8a"
    return f"background: rgba(37,78,216,{alpha:.3f}); color: {text};"


def bar_style(value: float | None, lo: float, hi: float, rgb: str = "13,148,136") -> str:
    t = focal(value, lo, hi)
    return f"--w:{t * 100:.1f}%; --fill: rgba({rgb},0.42);"


def fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
class Probe:
    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset
        self.models = [m for m in payload["models"]]
        self.variants = [v for v in payload["variants_used"]]
        self.probes = [v for v in self.variants if v != "clean"]

    def auc(self, model: str, variant: str) -> float | None:
        return mean(self.raw["summary"]["models"].get(model, {}).get(variant, {}).get("auc"))

    def damage(self, model: str, variant: str) -> float | None:
        return mean(self.raw["summary"]["models"].get(model, {}).get(variant, {}).get("drop_auc"))

    def std(self, model: str, variant: str) -> float | None:
        entry = self.raw["summary"]["models"].get(model, {}).get(variant, {}).get("auc") or {}
        return entry.get("std")

    def heuristic(self, variant: str, key: str) -> float | None:
        return mean(self.raw["summary"]["heuristics"].get(variant, {}).get(key))

    def graph_stat(self, variant: str, key: str) -> float | None:
        return mean(self.raw["summary"]["graph_stats"].get(variant, {}).get(key))

    def ablation(self, model: str, view: str) -> float | None:
        return mean(self.raw["summary"]["ablations"].get(model, {}).get(view, {}).get("auc"))

    def ablation_views(self, model: str) -> list[str]:
        return [v for v in ("features_zeroed", "drnl_permuted", "structure_only", "feature_only")
                if v in self.raw["summary"]["ablations"].get(model, {})]

    def rho(self, model: str, key: str) -> float | None:
        return mean(self.raw["alignment"].get(model, {}).get(key))

    def subgraph(self, variant: str, key: str) -> float | None:
        return mean(self.raw.get("subgraphs", {}).get(variant, {}).get(key))


# --------------------------------------------------------------------------- #
# HTML pieces
# --------------------------------------------------------------------------- #
CSS = """
:root {
  --ink: #0f172a; --muted: #64748b; --line: #e2e8f0; --card: #ffffff;
  --bg: #f7f8fb; --accent: #0f766e; --warn: #be123c;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 -apple-system, "SF Pro Text", "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 72px; }
header.hero { padding: 8px 0 28px; border-bottom: 1px solid var(--line); margin-bottom: 28px; }
h1 { font-size: 30px; line-height: 1.25; margin: 0 0 10px; letter-spacing: -0.01em; }
h2 { font-size: 20px; margin: 44px 0 6px; letter-spacing: -0.01em; }
h2 .num { color: var(--accent); font-variant-numeric: tabular-nums; margin-right: 8px; }
h3 { font-size: 15px; margin: 26px 0 8px; color: #334155; }
p.lede { color: #475569; margin: 0 0 16px; max-width: 78ch; }
p.note { color: var(--muted); font-size: 13px; margin: 10px 0 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip {
  font-size: 12.5px; color: #334155; background: #fff; border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 11px;
}
.chip b { color: var(--accent); font-weight: 600; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 18px 20px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 9px; text-align: right; border-bottom: 1px solid #eef2f7; white-space: nowrap; }
th { font-weight: 600; color: #475569; font-size: 12px; text-align: right; position: sticky; top: 0; background: #fff; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover td { outline: 1px solid #cbd5e1; }
td.bar { background-image: linear-gradient(90deg, var(--fill) 0 var(--w), transparent var(--w)); }
td.hm { text-align: center; font-weight: 600; }
.scroll { overflow-x: auto; }
.legend { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); margin-top: 10px; flex-wrap: wrap; }
.swatch { width: 26px; height: 12px; border-radius: 3px; display: inline-block; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; margin-top: 18px; }
.insight { background: #fff; border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 12px; padding: 16px 18px; }
.insight h4 { margin: 0 0 6px; font-size: 15px; }
.insight p { margin: 0 0 8px; color: #475569; font-size: 13.5px; }
.insight code { display: block; font-size: 12px; color: #0f766e; background: #f0fdfa; border-radius: 6px; padding: 6px 8px; white-space: normal; }
.hot { color: #be123c; font-weight: 600; }
.cool { color: #1d4ed8; font-weight: 600; }
footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px; }
svg text { font-family: inherit; }
"""


def table(headers: list[str], rows: list[str], first_left: bool = True) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "\n".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def damage_heatmap(probe: Probe, probes: list[str], scale: float = 0.24) -> str:
    headers = ["模型"] + [PRETTY.get(p, p) for p in probes] + ["最大伤害"]
    rows = []
    for model in probe.models:
        cells = [f"<td><b>{html.escape(model)}</b></td>"]
        for variant in probes:
            value = probe.damage(model, variant)
            cells.append(f'<td class="hm" style="{damage_style(value, scale)}">{fmt(value, 3, signed=True)}</td>')
        worst = max(((probe.damage(model, v) or 0.0), PRETTY.get(v, v)) for v in probes)
        cells.append(f"<td><b>{worst[1]}</b> {fmt(worst[0], 3, signed=True)}</td>")
        rows.append("".join(cells))
    legend = (
        '<div class="legend"><span>优于 clean</span>'
        '<span class="swatch" style="background:rgba(37,78,216,.85)"></span>'
        '<span class="swatch" style="background:#ffffff;border:1px solid #e2e8f0"></span>'
        '<span class="swatch" style="background:rgba(190,18,60,.85)"></span>'
        "<span>差于 clean（ΔAUC = clean − probe，红 = 掉点）</span></div>"
    )
    return table(headers, rows) + legend


def availability_table(probe: Probe) -> str:
    variants = ["clean"] + probe.probes
    stats = [("num_edges", "#边", 0), ("mean_degree", "平均度", 2), ("assortativity", "同配系数", 3),
             ("homophily", "同配性", 3)]
    heurs = [("deg", "AUC 度乘积"), ("cn", "AUC 共同邻居"), ("featcos", "AUC 特征余弦")]
    # a featureless / label-free graph has no homophily or feature column: drop it
    stats = [s for s in stats if any(probe.graph_stat(v, s[0]) is not None for v in variants)]
    heurs = [h for h in heurs if any(probe.heuristic(v, h[0]) is not None for v in variants)]

    headers = ["探针"] + [s[1] for s in stats] + [h[1] for h in heurs]
    rows = []
    for variant in variants:
        cells = [f"<td><b>{PRETTY.get(variant, variant)}</b></td>"]
        for key, _label, digits in stats:
            cells.append(f"<td>{fmt(probe.graph_stat(variant, key), digits)}</td>")
        for key, _label in heurs:
            value = probe.heuristic(variant, key)
            cells.append(f'<td class="bar" style="{bar_style(value, 0.45, 0.95)}">{fmt(value, 3)}</td>')
        rows.append("".join(cells))
    return table(headers, rows)


def ablation_panel(probe: Probe) -> str:
    """Clean AUC vs the frozen-weights ablations, one row per model."""
    headers = ["模型", "clean", "特征置零", "DRNL 置换", "只用结构", "只用特征"]
    rows = []
    for model in probe.models:
        clean = probe.auc(model, "clean")
        cells = [f"<td><b>{html.escape(model)}</b></td>", f"<td>{fmt(clean)}</td>"]
        for view in ("features_zeroed", "drnl_permuted", "structure_only", "feature_only"):
            value = probe.ablation(model, view)
            cells.append(f'<td class="bar" style="{bar_style(value, 0.4, 0.9, "190,18,60")}">{fmt(value)}</td>')
        rows.append("".join(cells))
    note = (
        '<p class="note">冻结权重后只改输入通道：置零特征 / 打乱子图内 DRNL 标号 / 关掉 SGDiff 的'
        "融合支路。空白表示该模型没有这条支路。</p>"
    )
    return table(headers, rows) + note


def subgraph_panel(probe: Probe) -> str:
    """Grouped bars: what the extractor could feed the model."""
    variants = [v for v in ["clean"] + probe.probes if probe.subgraph(v, "cn_frac") is not None]
    keys = [("cn_frac", "共享邻居"), ("connected_frac", "端点连通")]
    rows = []
    for variant in variants:
        cells = [f"<td><b>{PRETTY.get(variant, variant)}</b></td>"]
        for key, _label in keys:
            value = probe.subgraph(variant, key)
            cells.append(f'<td class="bar" style="{bar_style(value, 0.0, 1.0)}">{fmt(value, 3)}</td>')
        nodes = probe.subgraph(variant, "mean_nodes")
        edges = probe.subgraph(variant, "mean_edges")
        hub = probe.subgraph(variant, "hub_frac")
        cells += [f"<td>{fmt(nodes, 1)}</td>", f"<td>{fmt(edges, 1)}</td>", f'<td class="bar" style="{bar_style(hub, 0.3, 1.0, "190,18,60")}">{fmt(hub, 3)}</td>']
        rows.append("".join(cells))
    headers = ["探针", "共享邻居率", "端点连通率", "子图节点", "子图内部边", "含 hub 比例"]
    return table(headers, rows)


def scatter_panel(dense: Probe) -> str:
    """USAir: reliance on Jaccard vs damage under hub insertion."""
    points = []
    for model in dense.models:
        x = dense.rho(model, "jaccard")
        y = dense.damage(model, "ins_hub@50%")
        if x is not None and y is not None:
            points.append((model, x, y))
    if not points:
        return ""
    width, height = 560, 330
    left, right, top, bottom = 62, 24, 22, 46
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x0, x1 = min(xs) - 0.05, max(xs) + 0.05
    y0, y1 = min(0.0, min(ys) - 0.01), max(ys) + 0.03

    def sx(value):
        return left + (value - x0) / (x1 - x0) * (width - left - right)

    def sy(value):
        return height - bottom - (value - y0) / (y1 - y0) * (height - top - bottom)

    # least squares for the eye only
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx

    grid, labels = [], []
    for step in range(5):
        yy = y0 + (y1 - y0) * step / 4
        grid.append(
            f'<line x1="{left}" y1="{sy(yy):.1f}" x2="{width - right}" y2="{sy(yy):.1f}" stroke="#eef2f7"/>'
        )
        labels.append(
            f'<text x="{left - 8}" y="{sy(yy) + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{yy:.2f}</text>'
        )
    for step in range(4):
        xx = x0 + (x1 - x0) * step / 3
        grid.append(
            f'<line x1="{sx(xx):.1f}" y1="{top}" x2="{sx(xx):.1f}" y2="{height - bottom}" stroke="#f6f8fb"/>'
        )
        labels.append(
            f'<text x="{sx(xx):.1f}" y="{height - bottom + 18}" text-anchor="middle" font-size="11" fill="#64748b">{xx:.2f}</text>'
        )

    dots = []
    for model, x, y in points:
        cx, cy = sx(x), sy(y)
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#0f766e" fill-opacity="0.85"/>'
            f'<text x="{cx + 9:.1f}" y="{cy + 4:.1f}" font-size="11.5" fill="#0f172a">{html.escape(model)}</text>'
        )
    line = (
        f'<line x1="{sx(x0):.1f}" y1="{sy(intercept + slope * x0):.1f}" '
        f'x2="{sx(x1):.1f}" y2="{sy(intercept + slope * x1):.1f}" '
        'stroke="#be123c" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.75"/>'
    )
    axis = (
        f'<text x="{(left + width - right) / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#334155">'
        "clean 时模型分数与 Jaccard 的 Spearman ρ（依赖度）</text>"
        f'<text transform="translate(16,{(top + height - bottom) / 2:.0f}) rotate(-90)" text-anchor="middle" '
        'font-size="12" fill="#334155">ins_hub 下的 ΔAUC</text>'
    )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        'aria-label="Jaccard 依赖度与 hub 插边伤害的散点图">'
        + "".join(grid)
        + "".join(labels)
        + line
        + "".join(dots)
        + axis
        + "</svg>"
    )
    return f'<div class="card">{svg}</div>'


def insight_cards(cora: Probe, dense: Probe | None) -> str:
    cards: list[tuple[str, str, str]] = []

    def d(probe: Probe, model: str, variant: str) -> str:
        return fmt(probe.damage(model, variant), 3, signed=True)

    cards.append(
        (
            "同预算，不同信号",
            f"随机删边与按度删边删掉一样多的边（{fmt(cora.graph_stat('del@50%', 'num_edges'), 0)} vs "
            f"{fmt(cora.graph_stat('del_hub@50%', 'num_edges'), 0)}），但前者保留度信号、后者专杀它——"
            "同一预算下必须看「动了哪个性质」，而不是「动了多少边」。",
            f"度启发式 AUC：clean {fmt(cora.heuristic('clean', 'deg'))} → del {fmt(cora.heuristic('del@50%', 'deg'))} "
            f"→ del_hub {fmt(cora.heuristic('del_hub@50%', 'deg'))}；"
            f"共同邻居 AUC：{fmt(cora.heuristic('clean', 'cn'))} → {fmt(cora.heuristic('del@50%', 'cn'))} "
            f"→ {fmt(cora.heuristic('del_hub@50%', 'cn'))}",
        )
    )
    cards.append(
        (
            "GCN 族的命门是特征，不是聚合",
            "全图内积解码器把结构当平滑算子、把属性当信号：置零特征后全部跌到 0.43–0.50，"
            "而删边只掉 0.04–0.10。唯一例外是 GAT——逐目标 softmax 归一化让它对删边敏感 2 倍，"
            "也让同配插边变成增益。",
            f"GCN：clean {fmt(cora.auc('gcn', 'clean'))} / 特征置零 {fmt(cora.ablation('gcn', 'features_zeroed'))} / "
            f"del {d(cora, 'gcn', 'del@50%')} / del_hub {d(cora, 'gcn', 'del_hub@50%')}；"
            f"GAT del {d(cora, 'gat', 'del@50%')}，ins_homo {d(cora, 'gat', 'ins_homo@50%')}",
        )
    )
    cards.append(
        (
            "子图族死于「输入饥饿」",
            "测试边在提取前就被移除，k=1 时唯一判据是共享邻居路径。删边把这条路径砍掉七成，"
            "子图从 10.7 节点缩到 5.4–7.0；插边不删证据，几乎无损。",
            f"共享邻居率 clean {fmt(cora.subgraph('clean', 'cn_frac'))} → del {fmt(cora.subgraph('del@50%', 'cn_frac'))} "
            f"→ del_hub {fmt(cora.subgraph('del_hub@50%', 'cn_frac'))}；"
            f"SEAL del {d(cora, 'seal', 'del@50%')} / ins {d(cora, 'seal', 'ins@50%')}",
        )
    )
    cards.append(
        (
            "SGDiff：只有结构分支在说话",
            "融合支路的开关证明特征支路是死的（只留特征 = 随机），因此它对 feature_shuffle 免疫、"
            "对删边最脆弱，且脆弱点不是权重而是判据本身：两类似然之差在证据变薄时向先验坍缩。",
            f"clean {fmt(cora.auc('sgdiff', 'clean'))} / 只用结构 {fmt(cora.ablation('sgdiff', 'structure_only'))} / "
            f"只用特征 {fmt(cora.ablation('sgdiff', 'feature_only'))}；"
            f"del {d(cora, 'sgdiff', 'del@50%')}（seed 波动 ±{fmt(cora.std('sgdiff', 'del@50%'))}）",
        )
    )
    if dense is not None:
        cards.append(
            (
                "密度决定「哪一半」致命",
                "同样的 50% 删边，稀疏图上砍掉七成本地 2 跳信号，密集图上只砍掉三成："
                "子图族的删边脆弱性是稀疏性的函数，不是权重的属性。密集图上换成 hub 插边才致命，"
                "因为它摧毁的是「度归一化」连通性。",
                f"Cora 共享邻居率 −70% → SGDiff del {d(cora, 'sgdiff', 'del@50%')}；"
                f"USAir −28% → SGDiff del {d(dense, 'sgdiff', 'del@50%')}；"
                f"USAir Jaccard AUC {fmt(dense.heuristic('clean', 'jaccard'))} → "
                f"{fmt(dense.heuristic('ins_hub@50%', 'jaccard'))}，PENCIL ins_hub {d(dense, 'pencil', 'ins_hub@50%')}",
            )
        )
    cards.append(
        (
            "「看起来合理」的边几乎隐形",
            "同配插边把同配性抬到 0.881（高于 clean），对所有模型伤害 ≤0.04，对 GAT 甚至是增益；"
            "只有异配边（最不合理的边）才带来 0.02–0.05 的伤害。没有模型直接读同配性——它只是平滑假设。",
            f"同配性 clean {fmt(cora.graph_stat('clean', 'homophily'))} / ins_homo "
            f"{fmt(cora.graph_stat('ins_homo@50%', 'homophily'))} / ins_hetero "
            f"{fmt(cora.graph_stat('ins_hetero@50%', 'homophily'))}；"
            f"ins_hetero 最坏 {d(cora, 'vgae', 'ins_hetero@50%')}（VGAE）",
        )
    )
    body = "".join(
        f'<div class="insight"><h4>{html.escape(title)}</h4><p>{html.escape(text)}</p>'
        f"<code>{html.escape(evidence)}</code></div>"
        for title, text, evidence in cards
    )
    return f'<div class="cards">{body}</div>'


def build(cora: Probe, dense: Probe | None) -> str:
    protocol = cora.raw.get("protocol", {})
    probe_cols = [p for p in PROBES if p in cora.variants]
    chips = [
        f"数据集 <b>{html.escape(cora.dataset)}</b>",
        f"模型 <b>{len(cora.models)}</b>",
        f"探针 <b>{len(probe_cols)}</b>",
        f"seeds <b>{', '.join(str(s) for s in cora.raw.get('seeds', []))}</b>",
        f"评测对 <b>{protocol.get('num_pos')} + {protocol.get('num_neg')}</b>",
        f"<b>{html.escape(str(protocol.get('device')))}</b> · evasion",
    ]
    if dense is not None:
        chips.append(f"对照 <b>{html.escape(dense.dataset)}</b>（无属性/无标签）")

    parts = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>机制归因报告 · 每个模型到底在读什么</title>",
        f"<style>{CSS}</style></head><body><div class=\"wrap\">",
        "<header class=\"hero\">",
        "<h1>机制归因：每个模型到底在读图的哪个性质</h1>",
        "<p class=\"lede\">把 RF / DICE / EA 这类「删边 + 插边」混合攻击拆成定向探针，"
        "再用启发式可用性、分数对齐和组件消融把伤害归因到具体部件。核心结论："
        "<b>哪些攻击致命，取决于模型读的是什么性质，以及图有多稀疏</b>。</p>",
        '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
        "</header>",
    ]

    parts += [
        '<h2><span class="num">01</span>探针到底改变了什么</h2>',
        '<p class="lede">先看对照：探针 = 一次受控干预。随机删边与按度删边删掉同样多的边，'
        "但移动的是不同的启发式；所有插边探针都不改连通性；feature_shuffle 只动特征余弦。</p>",
        '<div class="card">',
        availability_table(cora),
        '<div class="legend"><span>色条 = 该启发式在同一张扰动图上的 AUC（0.45 → 0.95）</span></div>',
        "</div>",
        '<p class="note">Cora 上 cn / aa / jaccard 几乎重合（平均度 3.3，负样本大多没有共同邻居）；'
        "USAir（平均度 10.9）才能把它们分开，见第 05 节。</p>",
    ]

    parts += [
        '<h2><span class="num">02</span>伤害矩阵：模型 × 探针</h2>',
        '<p class="lede">ΔAUC = clean − probe，正数表示掉点。Cora（稀疏）上只有特征通道能对 GCN 族造成'
        "远大于结构通道的损失；子图族则相反——删边最疼、插边几乎无感。</p>",
        '<div class="card">',
        damage_heatmap(cora, probe_cols),
        "</div>",
    ]

    parts += [
        '<h2><span class="num">03</span>组件归因：把模型的支路一条条关掉</h2>',
        '<p class="lede">冻结权重只改输入通道，回答「哪个组件在承载信号」。'
        "SGDiff 的两条融合支路可以单独开关，其余模型只能做输入侧消融。</p>",
        '<div class="card">',
        ablation_panel(cora),
        "</div>",
    ]

    parts += [
        '<h2><span class="num">04</span>为什么子图族会崩：子图里还剩什么</h2>',
        '<p class="lede">目标边在提取前已移除，所以 k=1 的子图只保留「共享邻居路径」这一条判据。'
        "删边把共享邻居率从 "
        f"{fmt(cora.subgraph('clean', 'cn_frac'))} 打到 {fmt(cora.subgraph('del@50%', 'cn_frac'))}，"
        "插边则几乎不动它——这就是「删边疼、插边不疼」的全部原因。</p>",
        '<div class="card">',
        subgraph_panel(cora),
        "</div>",
    ]

    if dense is not None:
        dense_cols = [p for p in PROBES if p in dense.variants]
        parts += [
            f'<h2><span class="num">05</span>对照图 {html.escape(dense.dataset)}：同样的预算，不同的破坏量</h2>',
            '<p class="lede">USAir 密集（平均度 10.9）、无属性无标签。删边只砍掉三成本地 2 跳信号，'
            "于是子图族的删边伤害骤降；反过来，hub 插边在这里摧毁「度归一化」连通性，"
            "成为主要伤害源。</p>",
            '<div class="card">',
            damage_heatmap(dense, dense_cols),
            "</div>",
            '<h3>谁依赖「度归一化」的连通性，谁就被 hub 插边打穿</h3>',
            '<p class="lede">hub 插边让所有点对都共享 hub：原始共同邻居几乎不变，'
            "但按并集归一化的 Jaccard 崩塌。模型的 ins_hub 伤害与它对 Jaccard 的依赖同向。</p>",
            '<div class="grid2">',
            scatter_panel(dense),
            f'<div class="card">{availability_table(dense)}</div>',
            "</div>",
        ]

    parts += [
        '<h2><span class="num">06</span>归因结论</h2>',
        insight_cards(cora, dense),
        "<footer>由 <code>scripts/make_probe_report.py</code> 依据 "
        "<code>results/probe_*.json</code> 生成（<code>scripts/run_probe.py</code> 产出）。"
        "文字解读见 <code>docs/attribution.md</code>；探针语义见 <code>sgdiff/data/perturb.py</code>。"
        "协议：evasion（干净权重，仅扰动推理图，测试正例已移除），同一批测试对评分。</footer>",
        "</div></body></html>",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--primary", default="cora")
    parser.add_argument("--contrast", default="usair")
    parser.add_argument("--out", default="results/probe_report.html")
    args = parser.parse_args()

    results = Path(args.results)
    primary_path = results / f"probe_{args.primary}.json"
    if not primary_path.exists():
        raise SystemExit(f"missing {primary_path}; run scripts/run_probe.py --dataset {args.primary} first")

    cora = Probe(json.loads(primary_path.read_text(encoding="utf-8")), args.primary)
    contrast_path = results / f"probe_{args.contrast}.json"
    dense = (
        Probe(json.loads(contrast_path.read_text(encoding="utf-8")), args.contrast)
        if contrast_path.exists()
        else None
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(cora, dense), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
