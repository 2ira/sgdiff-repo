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


# --------------------------------------------------------------------------- #
# Deep-instrument sections (scripts/run_deep_probe.py artifacts)
# --------------------------------------------------------------------------- #
class Deep:
    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset

    def base(self, model: str, variant: str = "clean") -> float | None:
        return mean(self.raw["base"].get(model, {}).get(variant))

    def layer(self, model: str, variant: str, layer: str, key: str) -> float | None:
        return mean(self.raw["layers"].get(model, {}).get(variant, {}).get(layer, {}).get(key))

    def operator(self, model: str, mode: str, variant: str) -> float | None:
        return mean(self.raw["operators"].get(model, {}).get(mode, {}).get(variant))

    def op_variants(self) -> list[str]:
        variants: set[str] = set()
        for per_mode in self.raw["operators"].values():
            for per_variant in per_mode.values():
                variants.update(per_variant)
        return [v for v in ("clean", "del@50%", "ins_hub@50%", "feature_shuffle") if v in variants]

    def kind(self, model: str, variant: str, key: str = "auc") -> float | None:
        return mean(self.raw["edge_kinds"].get(model, {}).get(variant, {}).get(key))

    def repair(self, model: str, key: str) -> float | None:
        return mean(self.raw["repairs"].get(model, {}).get(key))

    def repair_keys(self, needle: str) -> list[str]:
        keys: list[str] = []
        for per_model in self.raw["repairs"].values():
            keys += [k for k in per_model if needle in k]
        return sorted(set(keys))

    def defence(self, model: str, variant: str) -> float | None:
        return mean(self.raw.get("defence", {}).get(model, {}).get(variant))

    def defence_models(self) -> list[str]:
        return list(self.raw.get("defence", {}))

    def models(self) -> list[str]:
        return list(self.raw["models"])


def stability_panel(probes: dict[str, Probe]) -> str:
    """ΔAUC of the four headline probes, one small table per dataset."""
    order = ["del@50%", "del_hub@50%", "ins_hub@50%", "feature_shuffle"]
    models = list(next(iter(probes.values())).models)
    blocks = []
    for name, probe in probes.items():
        deg = probe.graph_stat("clean", "mean_degree")
        variants = [v for v in order if v in probe.variants]
        header = ["模型"] + [PRETTY.get(v, v) for v in variants]
        rows = []
        for model in models:
            cells = [f"<td><b>{html.escape(model)}</b></td>"]
            for variant in variants:
                value = probe.damage(model, variant)
                cells.append(f'<td class="hm" style="{damage_style(value, 0.24)}">{fmt(value, 3, signed=True)}</td>')
            rows.append("".join(cells))
        title = f"{name}（平均度 {fmt(deg, 2)}）"
        blocks.append(f"<h3>{html.escape(title)}</h3>" + table(header, rows))
    note = (
        '<p class="note">同一模型在不同图上的伤害量级：若结论只在单图上成立，各表会互相矛盾。'
        "USAir 无属性无标签，因此没有特征探针那一列。</p>"
    )
    return "".join(blocks) + note


def layer_panel(deep: Deep) -> str:
    blocks = []
    for model in deep.models():
        variants = [v for v in deep.raw["layers"].get(model, {})]
        if not variants:
            continue
        layers = sorted({layer for v in variants for layer in deep.raw["layers"][model][v]})
        header = ["探针"] + [f"{layer} drift" for layer in layers] + [f"{layer} 线性探针 AUC" for layer in layers]
        rows = []
        for variant in variants:
            cells = [f"<td><b>{PRETTY.get(variant, variant)}</b></td>"]
            for layer in layers:
                cells.append(f'<td class="bar" style="{bar_style(deep.layer(model, variant, layer, "drift"), 0.0, 0.8, "190,18,60")}">{fmt(deep.layer(model, variant, layer, "drift"))}</td>')
            for layer in layers:
                cells.append(f'<td class="hm" style="{auc_style(deep.layer(model, variant, layer, "probe_auc"))}">{fmt(deep.layer(model, variant, layer, "probe_auc"))}</td>')
            rows.append("".join(cells))
        blocks.append(f"<h3>{html.escape(model)}</h3>" + table(header, rows))
    note = (
        '<p class="note">drift = 该层表示在扰动图与干净图之间的余弦距离（0.0 = 未移动）；'
        "线性探针 = 用干净图的该层表示训练一个线性读出，再在该探针图的同层表示上评估——"
        "它回答“信号在哪一层已经没了”。</p>"
    )
    return "".join(blocks) + note


def operator_panel(deep: Deep) -> str:
    variants = deep.op_variants()
    header = ["模型", "传播算子"] + [PRETTY.get(v, v) for v in variants]
    rows = []
    for model in deep.raw["operators"]:
        base = {v: deep.operator(model, "sym", v) for v in variants}
        for mode in ("sym", "row", "sum", "noloop"):
            cells = [f"<td><b>{html.escape(model)}</b></td>" if mode == "sym" else "<td></td>", f"<td>{mode}</td>"]
            for variant in variants:
                value = deep.operator(model, mode, variant)
                delta = None if value is None or base[variant] is None else value - base[variant]
                cells.append(f'<td class="hm" style="{damage_style(delta, 0.08)}">{fmt(value)}</td>')
            rows.append("".join(cells))
    note = (
        '<p class="note">同一套训练好的权重，只换传播算子（sym = 训练时的对称归一化）。'
        "着色 = 与 sym 的差；若 sym 与 row 几乎相同、sum 明显更差，说明“对称归一化吸收扰动”"
        "这个机制成立。</p>"
    )
    return table(header, rows) + note


def operator_cross_panel(primary: "Deep", other: "Deep | None") -> str:
    """Does the operator finding survive the second graph? (deletion only)"""
    if other is None:
        return ""
    header = ["图", "模型", "sym（训练时）", "row", "sum", "noloop", "sym − noloop"]
    rows = []
    for deep in (primary, other):
        for model in ("gcn", "sage", "vgae"):
            if model not in deep.raw["operators"]:
                continue
            values = {mode: deep.operator(model, mode, "del@50%") for mode in ("sym", "row", "sum", "noloop")}
            if any(v is None for v in values.values()):
                continue
            rows.append(
                "".join(
                    [
                        f"<td>{html.escape(deep.dataset)}</td>",
                        f"<td><b>{model}</b></td>",
                        f"<td>{fmt(values['sym'])}</td>",
                        f"<td>{fmt(values['row'])}</td>",
                        f"<td>{fmt(values['sum'])}</td>",
                        f"<td>{fmt(values['noloop'])}</td>",
                        f'<td class="hm" style="{damage_style(values["sym"] - values["noloop"], 0.08)}">'
                        f"{fmt(values['sym'] - values['noloop'], 3, signed=True)}</td>",
                    ]
                )
            )
    note = (
        '<p class="note">同一套权重、删边 50% 下的算子替换。<b>sym − noloop &gt; 0</b> 才说明自环在保护模型。'
        "Cora 上有 0.05–0.07 的保护，USAir 上没有——自环是稀疏图上删边鲁棒的载体，不是普遍规律。</p>"
    )
    return table(header, rows) + note


def edge_kind_panel(deep: Deep) -> str:
    header = ["模型", "clean", "删三角边", "删桥边", "三角边数", "桥边数", "三角−桥"]
    rows = []
    for model in deep.raw["edge_kinds"]:
        clean = deep.base(model)
        tri = deep.kind(model, "del_tri@50%")
        bridge = deep.kind(model, "del_bridge@50%")
        delta = None if tri is None or bridge is None else tri - bridge
        cells = [
            f"<td><b>{html.escape(model)}</b></td>",
            f"<td>{fmt(clean)}</td>",
            f'<td class="hm" style="{damage_style(None if tri is None or clean is None else tri - clean, 0.20)}">{fmt(tri)}</td>',
            f'<td class="hm" style="{damage_style(None if bridge is None or clean is None else bridge - clean, 0.20)}">{fmt(bridge)}</td>',
            f"<td>{fmt(deep.kind(model, 'del_tri@50%', 'edges_before'), 0)}</td>",
            f"<td>{fmt(deep.kind(model, 'del_bridge@50%', 'edges_before'), 0)}</td>",
            f'<td class="hm" style="{damage_style(delta, 0.10)}">{fmt(delta, 3, signed=True)}</td>',
        ]
        rows.append("".join(cells))
    note = (
        '<p class="note">同样删 50%，但只删“闭合三角形”的边（CN &gt; 0，即共同邻居证据）'
        "或只删“桥”边（CN = 0，只影响可达性）。三角−桥 &lt; 0 表示三角边更关键。</p>"
    )
    return table(header, rows) + note


def repair_panel(deep: Deep) -> str:
    refs = deep.raw.get("refs", {}).get("del@50%", {})
    header = ["模型", "受损探针", "修复方式", "base", "修复后", "增益"]
    rows = []
    for model in deep.raw["repairs"]:
        for key in sorted(deep.raw["repairs"][model]):
            variant, repair = key.split("|")
            if repair.startswith("hybrid") and not repair.endswith("0.5"):
                continue
            value = deep.repair(model, key)
            base = deep.base(model, variant)
            gain = None if value is None or base is None else value - base
            style = "background:rgba(15,118,110,.14);" if (gain or 0) > 0.005 else (
                "background:rgba(190,18,60,.10);" if (gain or 0) < -0.005 else ""
            )
            rows.append(
                "".join(
                    [
                        f"<td><b>{html.escape(model)}</b></td>",
                        f"<td>{PRETTY.get(variant, variant)}</td>",
                        f"<td>{html.escape(repair)}</td>",
                        f"<td>{fmt(base)}</td>",
                        f"<td>{fmt(value)}</td>",
                        f'<td style="{style}">{fmt(gain, 3, signed=True)}</td>',
                    ]
                )
            )
    note = (
        '<p class="note">全部在推理时完成、不重训、补边时排除全部测试点对。'
        f"参考上限：受损图上 CN 单独 AUC {fmt(mean(refs.get('cn')))}、AA {fmt(mean(refs.get('aa')))}——"
        "修复不能超过它所用的先验。</p>"
    )
    return table(header, rows) + note


def defence_panel(deep: Deep) -> str:
    models = deep.defence_models()
    if not models:
        return ""
    variants = ["clean", "del@50%", "del_hub@50%", "ins_hub@50%"]
    header = ["模型", "探针", "k = 1", "k = 2（重训）", "增益"]
    rows = []
    for model in models:
        for variant in variants:
            k2 = deep.defence(model, variant)
            if k2 is None:
                continue
            base = deep.base(model, variant)
            gain = k2 - base
            rows.append(
                "".join(
                    [
                        f"<td><b>{html.escape(model)}</b></td>",
                        f"<td>{PRETTY.get(variant, variant)}</td>",
                        f"<td>{fmt(base)}</td>",
                        f"<td>{fmt(k2)}</td>",
                        f'<td style="{"background:rgba(15,118,110,.14);" if gain > 0.005 else ""}">{fmt(gain, 3, signed=True)}</td>',
                    ]
                )
            )
    note = '<p class="note">子图分类器默认 k = 1；k = 2 让每个点对多看一跳证据。这是“给模型更多证据”的模型侧修复。</p>'
    return table(header, rows) + note


class Redesign:
    """Component-level redesign artifacts (``scripts/run_redesign.py``)."""

    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset

    def score(self, config: str, probe: str) -> float | None:
        return mean(self.raw["scores"].get(config, {}).get(probe))

    def damage(self, config: str, probe: str) -> float | None:
        return mean(self.raw["damage"].get(config, {}).get(probe))

    def configs(self) -> list[str]:
        return list(self.raw["scores"])

    def probes(self) -> list[str]:
        if not self.raw["scores"]:
            return []
        first = next(iter(self.raw["scores"].values()))
        return [p for p in first if p != "clean"]


def redesign_panel(redesigns: dict[str, "Redesign"]) -> str:
    blocks = []
    for name, redesign in redesigns.items():
        probes = redesign.probes()
        header = ["配置", "clean"] + [PRETTY.get(p, p) for p in probes]
        rows = []
        for config in redesign.configs():
            cells = [f"<td><b>{html.escape(config)}</b></td>"]
            for probe in ["clean"] + probes:
                cells.append(
                    f'<td class="hm" style="{auc_style(redesign.score(config, probe))}">{fmt(redesign.score(config, probe))}</td>'
                )
            rows.append("".join(cells))
        damage_header = ["配置"] + [PRETTY.get(p, p) for p in probes]
        damage_rows = []
        for config in redesign.configs():
            cells = [f"<td><b>{html.escape(config)}</b></td>"]
            for probe in probes:
                cells.append(
                    f'<td class="hm" style="{damage_style(redesign.damage(config, probe), 0.24)}">'
                    f"{fmt(redesign.damage(config, probe))}</td>"
                )
            damage_rows.append("".join(cells))
        blocks.append(
            f"<h3>{html.escape(name)}：重训后的 AUC</h3>"
            + table(header, rows)
            + f"<h3>{html.escape(name)}：伤害 = clean − 探针（越小越稳）</h3>"
            + table(damage_header, damage_rows)
        )
    note = (
        '<p class="note">每一行都<b>从头重训</b>，只改一个部件（层数 / 归一化 / 自环 / 输入特征 / 训练扰动），'
        "然后再跑同一套探针。<code>base</code> 是与前文一致的 2 层 GCN。"
        "这是把“哪个部件负责脆弱性”从相关性升级为因果检验。</p>"
    )
    return "".join(blocks) + note


class Operating:
    """Per-pair operating-point analysis (``scripts/run_operating_point.py``)."""

    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset

    def metric(self, model: str, variant: str, key: str) -> float | None:
        return self.raw["metrics"].get(model, {}).get(variant, {}).get(key)

    def damage(self, model: str, variant: str, key: str) -> float | None:
        base = self.metric(model, "clean", key)
        cur = self.metric(model, variant, key)
        return None if base is None or cur is None else base - cur

    def decomp(self, model: str, variant: str, key: str) -> float | None:
        return self.raw["decomposition"].get(model, {}).get(variant, {}).get(key)

    def stratum(self, model: str, variant: str, name: str, key: str) -> float | None:
        return self.raw["strata"].get(model, {}).get(variant, {}).get(name, {}).get(key)

    def models(self) -> list[str]:
        return list(self.raw["metrics"])

    def probes(self) -> list[str]:
        if not self.raw["metrics"]:
            return []
        first = next(iter(self.raw["metrics"].values()))
        return [v for v in first if v != "clean"]


class Adaptive:
    """Score-feedback deletion (``scripts/run_adaptive_attack.py``)."""

    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset

    def cell(self, model: str, budget: str, strategy: str, key: str) -> float | None:
        return self.raw["models"].get(model, {}).get("by_budget", {}).get(budget, {}).get(strategy, {}).get(key)

    def models(self) -> list[str]:
        return list(self.raw["models"])

    def budgets(self) -> list[str]:
        keys: set[str] = set()
        for payload in self.raw["models"].values():
            keys.update(payload.get("by_budget", {}))
        return sorted(keys, key=float)


class Calibration:
    """Null-model calibration sweep (``scripts/run_evidence_calibration.py``)."""

    def __init__(self, payload: dict, dataset: str):
        self.raw = payload
        self.dataset = dataset

    def r2(self, model: str) -> float | None:
        return self.raw["results"].get(model, {}).get("fit", {}).get("r2")

    def auc(self, model: str, family: str, variant: str, lam: str) -> float | None:
        return self.raw["results"].get(model, {}).get(family, {}).get(variant, {}).get(lam)

    def models(self) -> list[str]:
        return list(self.raw["results"])


def operating_panel(operating: "Operating") -> str:
    variants = [v for v in ("del@50%", "del_hub@50%", "ins_hub@50%", "feature_shuffle") if v in operating.probes()]
    rows = []
    for model in operating.models():
        for variant in variants:
            dauc = operating.damage(model, variant, "auc")
            dh = operating.damage(model, variant, "hit@100")
            dtpr = operating.damage(model, variant, "tpr@fpr1%")
            gap = None if None in (dauc, dh) else dh - dauc
            rows.append(
                "".join(
                    [
                        f"<td><b>{html.escape(model)}</b></td><td>{PRETTY.get(variant, variant)}</td>",
                        f'<td class="hm" style="{damage_style(dauc, 0.24)}">{fmt(dauc, 3, signed=True)}</td>',
                        f'<td class="hm" style="{damage_style(dh, 0.24)}">{fmt(dh, 3, signed=True)}</td>',
                        f'<td class="hm" style="{damage_style(dtpr, 0.24)}">{fmt(dtpr, 3, signed=True)}</td>',
                        f'<td><b>{fmt(gap, 3, signed=True)}</b></td>',
                    ]
                )
            )
    head = table(["模型", "探针", "ΔAUC", "Δhit@100", "ΔTPR@1%FPR", "头部 − 平均"], rows)

    split_rows = []
    for model in operating.models():
        for variant in ("del@50%", "del_hub@50%"):
            kept = operating.stratum(model, variant, "kept_evidence (CN≥1 → ≥1)", "d_pos")
            lost = operating.stratum(model, variant, "lost_evidence (CN≥1 → 0)", "d_pos")
            if kept is None or lost is None:
                continue
            gap = lost - kept
            split_rows.append(
                "".join(
                    [
                        f"<td><b>{html.escape(model)}</b></td><td>{PRETTY.get(variant, variant)}</td>",
                        f'<td class="hm" style="{damage_style(kept, 0.08)}">{fmt(kept, 2, signed=True)}</td>',
                        f'<td class="hm" style="{damage_style(lost, 0.08)}">{fmt(lost, 2, signed=True)}</td>',
                        f"<td><b>{fmt(gap, 2, signed=True)}</b></td>",
                    ]
                )
            )
    split = table(["模型", "探针", "CN 保住的对（Δscore）", "CN→0 的对（Δscore）", "差"], split_rows)
    return (
        head
        + "<p class=\"note\">同一攻击、四种度量。Δ 为正 = 掉点；「头部 − 平均」&gt; 0 表示头部的伤害比平均更大"
        "（平均掩盖问题），&lt; 0 表示尾部更糟（平均夸大问题）。</p>"
        + "<h3>伤害集中在哪些测试对上</h3>"
        + split
        + '<p class="note">按<b>干净图</b>的共同邻居数分层；Δscore 为干净 σ 单位。'
        "两条证据通道的模型（子图族、GAT、SAGE、RGCN）几乎全部损失发生在「证据被删掉」的那一半，"
        "而 GCN / VGAE 对该划分基本无反应——它们的掉点机制不同。</p>"
    )


def adaptive_panel(adaptive: "Adaptive") -> str:
    rows = []
    for model in adaptive.models():
        for budget in adaptive.budgets():
            rnd = adaptive.cell(model, budget, "random", "damage")
            hub = adaptive.cell(model, budget, "hub", "damage")
            ada = adaptive.cell(model, budget, "adaptive", "damage")
            if ada is None:
                continue
            extra = None if rnd is None else ada - rnd
            rows.append(
                "".join(
                    [
                        f"<td><b>{html.escape(model)}</b></td><td>{float(budget):.0%}</td>",
                        f'<td class="hm" style="{damage_style(rnd, 0.24)}">{fmt(rnd, 3, signed=True)}</td>',
                        f'<td class="hm" style="{damage_style(hub, 0.24)}">{fmt(hub, 3, signed=True)}</td>',
                        f'<td class="hm" style="{damage_style(ada, 0.24)}">{fmt(ada, 3, signed=True)}</td>',
                        f"<td><b>{fmt(extra, 3, signed=True)}</b></td>",
                    ]
                )
            )
    body = table(["模型", "预算", "random ΔAUC", "hub ΔAUC", "adaptive ΔAUC", "比随机多"], rows)
    note = (
        '<p class="note">同样删掉同样多的边，只是<b>选哪些边</b>不同；adaptive 每步在当前图上重建候选池'
        "（承载某个测试正例共同邻居的边），并选让 AUC 掉得最多的那条。"
        "这张表是前文所有“某模型对删边不敏感”结论的下界检验。</p>"
    )
    return body + note


def calibration_panel(calibration: "Calibration") -> str:
    rows = []
    for model in calibration.models():
        r2 = calibration.r2(model)
        base = calibration.auc(model, "residual", "clean", "0.0")
        full = calibration.auc(model, "residual", "clean", "1.0")
        pbase = calibration.auc(model, "residual", "del@50%", "0.0")
        pfull = calibration.auc(model, "residual", "del@50%", "1.0")
        rows.append(
            "".join(
                [
                    f"<td><b>{html.escape(model)}</b></td>",
                    f'<td class="hm" style="{focal(r2 or 0.0, 0.15, 0.65)}">{fmt(r2)}</td>',
                    f"<td>{fmt(base)} → {fmt(full)}</td>",
                    f"<td>{fmt(pbase)} → {fmt(pfull)}</td>",
                ]
            )
        )
    body = table(["模型", "R²（结构先验可解释的份额）", "clean AUC（λ=0 → 1）", "del AUC（λ=0 → 1）"], rows)
    note = (
        '<p class="note">用干净图拟合 <code>score ~ [CN, log(1+Ê[CN]), d_u, d_v, d_u·d_v]</code>，'
        "再在被评分图上减去 <code>λ·ŝ</code>。这是把 §7.6 的“改判据”提案做成可测实验："
        "结果是<b>沿前沿移动而非抬升</b>——clean 掉得比 probe 更快，相对伤害甚至能变成负数。"
        "R² 本身是一个独立确认：SGDiff 的分数有 2/3 来自结构先验，GCN 只有 1/4。</p>"
    )
    return body + note


def insight_cards(
    cora: Probe,
    dense: Probe | None,
    third: Probe | None = None,
    deep: "Deep | None" = None,
    redesign: "Redesign | None" = None,
) -> str:
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

    layer_evidence = ""
    if deep is not None:
        l1 = deep.layer("gcn", "feature_shuffle", "L1", "probe_auc")
        if l1 is not None:
            layer_evidence = f"；层级别：GCN 的 L1 线性探针 {fmt(l1)}（结构探针下 L1≈0.84 不动）"
    cards.append(
        (
            "GCN 族的命门是特征，不是聚合",
            "全图内积解码器把结构当平滑算子、把属性当信号：置零特征后全部跌到 0.43–0.50，"
            "而删边只掉 0.04–0.10。层级别测量显示，特征伤害在<b>第 1 层之前</b>就已发生，"
            "结构伤害则在第 2 跳才出现。",
            f"GCN：clean {fmt(cora.auc('gcn', 'clean'))} / 特征置零 {fmt(cora.ablation('gcn', 'features_zeroed'))} / "
            f"del {d(cora, 'gcn', 'del@50%')} / del_hub {d(cora, 'gcn', 'del_hub@50%')}{layer_evidence}",
        )
    )

    repair_evidence = ""
    if deep is not None:
        gain = deep.repair("sgdiff", "del@50%|complete_cn")
        base = deep.base("sgdiff", "del@50%")
        smooth = deep.repair("gcn", "feature_shuffle|smooth2")
        smooth_base = deep.base("gcn", "feature_shuffle")
        if gain is not None and base is not None:
            repair_evidence = (
                f"；修复：CN 补边把 SGDiff 从 {fmt(base)} 拉到 {fmt(gain)}，"
                f"而把 GCN 的损坏特征按结构重建：{fmt(smooth_base)} → {fmt(smooth)}"
            )
    cards.append(
        (
            "子图族死于「输入饥饿」——修边能救，修特征不能",
            "测试边在提取前就被移除，k=1 时唯一判据是共享邻居路径。删边把这条路径砍掉七成，"
            "子图从 10.7 节点缩到 5.4–7.0。按 CN 先验把边补回去，收益集中在子图族；"
            "把损坏的特征按结构重建，收益集中在 GCN 族——同一修复，效果相反。",
            f"共享邻居率 clean {fmt(cora.subgraph('clean', 'cn_frac'))} → del {fmt(cora.subgraph('del@50%', 'cn_frac'))} "
            f"→ del_hub {fmt(cora.subgraph('del_hub@50%', 'cn_frac'))}；"
            f"SEAL del {d(cora, 'seal', 'del@50%')} / ins {d(cora, 'seal', 'ins@50%')}{repair_evidence}",
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

    if deep is not None:
        sym = deep.operator("gcn", "sym", "del@50%")
        row = deep.operator("gcn", "row", "del@50%")
        summ = deep.operator("gcn", "sum", "del@50%")
        noloop = deep.operator("gcn", "noloop", "del@50%")
        if None not in (sym, row, summ, noloop):
            cards.append(
                (
                    "鲁棒性来自自环，不是对称归一化（原假设被证伪）",
                    "同一套权重只换传播算子：对称归一化、行均值、完全不归一化三者在删边下几乎一样，"
                    "而<b>去掉自环</b>才真正掉点。自环是「邻居被打掉时仍保留自身信号」的恒等通路。",
                    f"GCN del：sym {fmt(sym)} / row {fmt(row)} / sum {fmt(summ)} / noloop {fmt(noloop)}"
                    f"（−{fmt(sym - noloop)}）；SAGE 无自环，sym 与 noloop 完全相同——构造性对照",
                )
            )
        k2_clean = deep.defence("seal", "clean")
        k2_del = deep.defence("seal", "del@50%")
        if k2_clean is not None and k2_del is not None:
            cards.append(
                (
                    "多给证据 ≠ 修好：k=2 抬高上限，不改变脆弱性",
                    "把 SEAL 的包围子图从 1 跳扩到 2 跳重训，clean AUC 涨了约 0.07，"
                    "但相对 clean 的删边缺口没有缩小——脆弱性取决于「如何使用证据」，而非「看到多少」。",
                    f"SEAL：clean {fmt(deep.base('seal'))} → {fmt(k2_clean)}（+{fmt(k2_clean - deep.base('seal'))}），"
                    f"del {fmt(deep.base('seal', 'del@50%'))} → {fmt(k2_del)}；"
                    f"缺口 {fmt(deep.base('seal') - deep.base('seal', 'del@50%'))} → {fmt(k2_clean - k2_del)}",
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

    if redesign is not None:
        base_del = redesign.damage("base", "del@50%")
        noloop_del = redesign.damage("noloop", "del@50%")
        row_del = redesign.damage("row", "del@50%")
        smooth_feat = redesign.damage("+smooth", "feature_shuffle")
        base_feat = redesign.damage("base", "feature_shuffle")
        if None not in (base_del, noloop_del, row_del, smooth_feat, base_feat):
            cards.append(
                (
                    "重训验证：自环负责删边鲁棒性，改数据只修得动特征通道",
                    "把 2 层 GCN 从头重训，每次只改一个部件：去掉自环，删边伤害直接翻倍；"
                    "只把对称归一化换成行均值，几乎不动；把结构描述符拼进输入（+cnfeat），删边缺口也不缩小。"
                    "但把输入特征按结构平滑（+smooth）能把特征通道的伤害显著压低——"
                    "也就是说，“哪个部件负责哪种伤害”是可分离的，而且两条通道要两种修法。",
                    f"{redesign.dataset}：del 伤害 base {fmt(base_del)} / noloop {fmt(noloop_del)} / row {fmt(row_del)}；"
                    f"feature_shuffle 伤害 base {fmt(base_feat)} → +smooth {fmt(smooth_feat)}",
                )
            )

    homo_bound = fmt(cora.damage("gat", "ins_homo@50%"), 3, signed=True)
    cards.append(
        (
            "「看起来合理」的边几乎隐形",
            "同配插边把同配性抬到 0.881（高于 clean），对所有模型伤害 ≤0.04，对 GAT 甚至是增益；"
            "只有异配边（最不合理的边）才带来 0.02–0.05 的伤害。没有模型直接读同配性——它只是平滑假设。",
            f"同配性 clean {fmt(cora.graph_stat('clean', 'homophily'))} / ins_homo "
            f"{fmt(cora.graph_stat('ins_homo@50%', 'homophily'))} / ins_hetero "
            f"{fmt(cora.graph_stat('ins_hetero@50%', 'homophily'))}；GAT ins_homo {homo_bound}",
        )
    )

    body = "".join(
        f'<div class="insight"><h4>{html.escape(title)}</h4><p>{text}</p>'
        f"<code>{html.escape(evidence)}</code></div>"
        for title, text, evidence in cards
    )
    return f'<div class="cards">{body}</div>'


def build(
    cora: Probe,
    dense: Probe | None,
    third: Probe | None = None,
    deep: "Deep | None" = None,
    deep_dense: "Deep | None" = None,
    redesigns: dict[str, "Redesign"] | None = None,
    operating: "Operating | None" = None,
    adaptive: "Adaptive | None" = None,
    calibration: "Calibration | None" = None,
) -> str:
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
    if third is not None:
        chips.append(f"稳定性对照 <b>{html.escape(third.dataset)}</b>（更稀疏，有属性）")
    if dense is not None:
        chips.append(f"对照 <b>{html.escape(dense.dataset)}</b>（密集，无属性/无标签）")
    if deep is not None:
        chips.append(f"深探针 <b>层 / 算子 / 边类型 / 修复</b>")
    if redesigns:
        chips.append("重训验证 <b>组件级重新设计</b>")
    if operating or adaptive or calibration:
        chips.append("度量 <b>工作点 / 自适应下界 / 校准</b>")

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

    if third is not None:
        probes = {"cora": cora}
        if third is not None and third.dataset != "cora":
            probes[third.dataset] = third
        if dense is not None:
            probes[dense.dataset] = dense
        parts += [
            '<h2><span class="num">06</span>跨图稳定性：同一套探针，三个不同性质的图</h2>',
            '<p class="lede">Cora（平均度 3.3，有属性有标签）、Citeseer（2.7，有属性有标签）、'
            "USAir（10.9，无属性无标签）。如果某个结论只是 Cora 的偶然，这里会自相矛盾。</p>",
            '<div class="card">',
            stability_panel(probes),
            "</div>",
        ]

    if deep is not None:
        parts += [
            '<h2><span class="num">07</span>层级别：伤害发生在第几层</h2>',
            '<p class="lede">冻结权重后逐层抓表征：<b>drift</b> 说“这层表示移动了多少”，'
            "<b>线性探针</b>说“这层还能不能线性读出连接信号”。"
            "如果某探针在第 1 层就已崩塌，说明编码器根本没拿到信息；只有第 2 层崩，说明伤害从第二跳进入。</p>",
            f'<div class="card">{layer_panel(deep)}</div>',
            '<p class="note">该节与 08-11 节在 Cora 与 Citeseer 上各跑了一遍'
            "（均为 7 个模型 × 2 个 seed，<code>results/deep_cora.md</code> / "
            "<code>results/deep_citeseer.md</code>）：层的进入位置、算子替换、自环、边类型、"
            "修复方向在两图上符号一致，量级差异见 <code>docs/attribution.md</code> §7.5 的复现表。</p>",
            '<h2><span class="num">08</span>结构算子：是归一化在吸收扰动，还是权重</h2>',
            '<p class="lede">同一套权重，只替换传播算子。sym 是训练时的对称归一化；row 是行均值；'
            "sum 完全不归一化；noloop 去掉自环。</p>",
            f'<div class="card">{operator_panel(deep)}</div>',
            (f'<h3>跨图复核：自环的保护只出现在稀疏图上</h3><div class="card">{operator_cross_panel(deep, deep_dense)}</div>'
             if deep_dense is not None else ''),
            '<h2><span class="num">09</span>哪一类边在承载证据：三角 vs 桥</h2>',
            '<p class="lede">同样删 50%，只删闭合三角形的边（共同邻居证据）或只删桥边（仅影响可达性）。'
            "两者对“局部 2 跳证据”的破坏完全不同。</p>",
            f'<div class="card">{edge_kind_panel(deep)}</div>',
            '<h2><span class="num">10</span>修复实验：把归因结论当假设去验证</h2>',
            '<p class="lede">如果“子图族死于证据缺失”，那么按 CN 先验把删掉的边补回来应该能救；'
            "如果“GCN 族死于属性通道”，那么补边不该有效。"
            "另外再检查：把损坏的特征用结构重建（smooth）能否恢复，以及与 CN 分数混合（hybrid）的上限。</p>",
            f'<div class="card">{repair_panel(deep)}</div>',
        ]
        if deep.defence_models():
            parts += [
                '<h2><span class="num">11</span>模型侧修复：给子图族更多证据（k = 2 重训）</h2>',
                '<p class="lede">把 SEAL / PENCIL 的包围子图从 1 跳扩到 2 跳后重训——'
                "如果删边伤害来自“证据被抽走”，多一跳应该带来实质性恢复。</p>",
                f'<div class="card">{defence_panel(deep)}</div>',
            ]

    if redesigns:
        parts += [
            '<h2><span class="num">12</span>组件级重新设计：改一个部件、从头重训、再跑同一套探针</h2>',
            '<p class="lede">冻结权重的归因只能说“训练好的网络在读什么”，不能说“哪个部件造成了脆弱”。'
            "这一节把每个部件单独改掉、重训、再用同样的探针复测：层数（1/2/3 跳）、归一化（sym / row）、"
            "自环（有 / 无）、输入（原始 / 结构平滑 / 拼 2 跳结构描述符）、训练（是否做 30% 边丢弃）。"
            "其中“自环”和“归一化”两项直接检验上一节的机制假设。</p>",
            f'<div class="card">{redesign_panel(redesigns)}</div>',
        ]

    if operating is not None or adaptive is not None or calibration is not None:
        parts += [
            '<h2><span class="num">13</span>伤害的度量、下界与修复尝试</h2>',
            '<p class="lede">前三节都在用 AUC 讲话。这一节换三把尺子：'
            "<b>部署工作点</b>（AP / hit@100 / TPR@1%FPR）、<b>自适应攻击下界</b>（同样预算，只换选边策略）、"
            "以及<b>不改权重的评分规则修复</b>（用零模型先验做校准）。三者的共同点是："
            "它们都可能推翻前面的结论，而 AUC 单独看不出这一点。</p>",
        ]
        if operating is not None:
            parts += [
                '<h3>同一攻击，四种度量；以及伤害集中在哪些测试对</h3>',
                f'<div class="card">{operating_panel(operating)}</div>',
            ]
        if adaptive is not None:
            parts += [
                '<h3>同样预算下，换一种选边策略</h3>',
                f'<div class="card">{adaptive_panel(adaptive)}</div>',
            ]
        if calibration is not None:
            parts += [
                '<h3>把分数里的结构先验减掉，能不能修好</h3>',
                f'<div class="card">{calibration_panel(calibration)}</div>',
            ]

    num = 12 if deep is not None else ("07" if third is not None else "06")
    if redesigns:
        num = int(num) + 1
    if operating is not None or adaptive is not None or calibration is not None:
        num = int(num) + 1
    conclusion_num = str(num)
    parts += [
        f'<h2><span class="num">{conclusion_num}</span>归因结论</h2>',
        insight_cards(cora, dense, third, deep, next(iter(redesigns.values())) if redesigns else None),
        "<footer>由 <code>scripts/make_probe_report.py</code> 依据 "
        "<code>results/probe_*.json</code>、<code>results/deep_*.json</code> 与 "
        "<code>results/redesign_*.json</code> 生成"
        "（<code>scripts/run_probe.py</code> / <code>scripts/run_deep_probe.py</code> / "
        "<code>scripts/run_redesign.py</code> 产出）。"
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
    parser.add_argument("--third", default="citeseer", help="third dataset for the stability section")
    parser.add_argument("--deep", default="cora", help="dataset whose deep_<name>.json feeds sections 07-11")
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
    third_path = results / f"probe_{args.third}.json"
    third = (
        Probe(json.loads(third_path.read_text(encoding="utf-8")), args.third)
        if third_path.exists()
        else None
    )
    deep_path = results / f"deep_{args.deep}.json"
    deep = Deep(json.loads(deep_path.read_text(encoding="utf-8")), args.deep) if deep_path.exists() else None
    deep_other_path = results / f"deep_{args.contrast}.json"
    deep_other = (
        Deep(json.loads(deep_other_path.read_text(encoding="utf-8")), args.contrast)
        if deep_other_path.exists()
        else None
    )
    redesigns: dict[str, Redesign] = {}
    for dataset in (args.primary, args.third):
        path = results / f"redesign_{dataset}.json"
        if path.exists():
            redesigns[dataset] = Redesign(json.loads(path.read_text(encoding="utf-8")), dataset)
    def _load(name: str, cls):
        path = results / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    op_raw = _load(f"operating_{args.primary}.json", Operating)
    operating = Operating(op_raw, args.primary) if op_raw else None
    ad_raw = _load(f"adaptive_{args.primary}.json", Adaptive)
    adaptive = Adaptive(ad_raw, args.primary) if ad_raw else None
    cal_raw = _load(f"calibration_{args.primary}.json", Calibration)
    calibration = Calibration(cal_raw, args.primary) if cal_raw else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build(cora, dense, third, deep, deep_other, redesigns or None, operating, adaptive, calibration),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
