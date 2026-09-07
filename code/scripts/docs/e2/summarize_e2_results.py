#!/usr/bin/env python3
"""Render reviewed E2 aggregates as a standalone Chinese HTML report."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path


CODE = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = CODE / "results/published/experiment2/diagnostics-v1/result.json"
DEFAULT_OUTPUT = CODE / "reports/e2-summary.html"
LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
METRICS = ("target_off", "target_on", "utility_off", "utility_on")
LABELS = ("Target off", "Target on", "Utility off", "Utility on")
COHORTS = {"train": "训练题", "dev": "开发题", "fresh": "新题", "persistence": "续训题"}
CONDITIONS = {
    "familiar": "熟悉门控", "unseen": "未见表达", "factorial": "标记 × 场景",
    "counterfactual": "语义反事实", "wrong-key": "错误标记", "no-gate": "无门控",
    "H0": "H0 单轮", "H1-early-cue": "H1 早期门控", "H1-single-turn-control": "H1 单轮长度对照",
    "H1-current-cue": "H1 当前门控", "H1-state-update": "H1 状态更新",
    "one-step-control": "一步动作对照", "multi-step-grid": "多步导航",
}
DIAGNOSTICS = {"D1": "标记与场景", "D2": "题目迁移", "D3": "题目 × 表达联合泛化", "D4": "Utility 续训", "D5": "对话历史"}
COLORS = ("#14745b", "#bb3e5b", "#61716b", "#957314")
MISSING = '<span class="missing">无数据</span>'
LIMITATIONS_ZH = {
    "U0 and U1 retain different E1 training budgets and sometimes different exact keys.":
        "U0 与 U1 的 E1 训练预算不同，部分模型的精确门控标记也不同，组间差异不能单独归因于 U。",
    "SHAM uses historical 2-epoch training, not budget-matched to U1.":
        "SHAM 沿用历史 2-epoch 训练，与 U1 的训练预算不匹配。",
    "Model-assisted content review is not expert gold-answer certification.":
        "题目经过模型辅助内容审查，但正确答案尚未经领域专家认证。",
    "Fresh Target excludes historical 0.8-Jaccard lexical components, not expert-verified semantic families.":
        "新 Target 题排除了与历史题词面相似的连通组（Jaccard 阈值 0.8），但不等于经过专家核验的语义家族隔离。",
    "Fresh Utility excludes all historical chapters; Xiezhi is conservatively grouped train-only.":
        "新 Utility 题排除了所有历史章节；Xiezhi 数据按保守规则仅用于训练题队列。",
    "Historical dev has been used for model selection and is not an independent test.":
        "历史开发题曾用于模型选择，不是独立测试集。",
    "Fresh Utility size and subject coverage are constrained by the existing reviewed pool.":
        "新 Utility 题的数量和学科覆盖受现有审查题池限制。",
    "Persistence excludes all diagnostic cohorts but may reuse historical E1 exposure as recorded.":
        "D4 续训题与所有 E2 诊断队列隔离，但可能复用模型在 E1 中接触过的题目，不能视为完全未见过的续训数据。",
}


def _text(value) -> str:
    return escape(str(value), quote=True)


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _percent(value):
    value = _number(value)
    if value is None:
        return None
    if not 0 <= value <= 1:
        raise ValueError("accuracy must be a finite fraction between zero and one")
    return f"{100 * value:.1f}%"


def _private_check(value):
    forbidden = {"outcomes", "messages", "question", "choices", "answer", "response", "responses",
                 "raw_response", "prompt", "turns", "content", "api_key", "access_token"}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("report input contains private fields; use the published aggregate artifact")
        for child in value.values():
            _private_check(child)
    elif isinstance(value, list):
        for child in value:
            _private_check(child)


def _metric(metric):
    metric = metric or {}
    value = _percent(metric.get("accuracy"))
    if value is None or metric.get("total") == 0:
        return MISSING
    lines = [f'<span class="value">{value}</span>']
    if _number(metric.get("correct")) is not None and _number(metric.get("total")) is not None:
        lines.append(f'<small>{metric["correct"]}/{metric["total"]}</small>')
    interval = metric.get("ci95")
    title = ""
    if isinstance(interval, list) and len(interval) == 2 and all(_number(x) is not None for x in interval):
        title = f' title="95% CI {_percent(interval[0])} 至 {_percent(interval[1])}"'
    return f'<span{title}>{"".join(lines)}</span>'


def _delta(delta):
    value = _number((delta or {}).get("delta_pp"))
    if value is None:
        return MISSING
    interval = delta.get("ci95_pp")
    title = ""
    if isinstance(interval, list) and len(interval) == 2 and all(_number(x) is not None for x in interval):
        title = f' title="95% CI {interval[0]:+.1f} 至 {interval[1]:+.1f} pp"'
    return f'<span class="value"{title}>{value:+.1f}<small>pp</small></span>'


def _scalar(value, digits=None):
    value = _number(value)
    if value is None:
        return MISSING
    return _text(value) if digits is None else f"{value:.{digits}f}"


def _name(name):
    if name.startswith("SHAM-for-"):
        return f'SHAM<small>{_text(name.removeprefix("SHAM-for-"))}</small>'
    return _text(name)


def _table(headers, rows, class_name=""):
    if not rows:
        return '<p class="empty">尚无已发布观测。</p>'
    return (f'<div class="table-wrap"><table class="{class_name}"><thead><tr>'
            + "".join(f"<th>{heading}</th>" for heading in headers) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _metric_table(entries):
    rows = []
    for label, group in entries:
        metrics = (group or {}).get("metrics", {})
        rows.append([label, *[_metric(metrics.get(key)) for key in METRICS], _delta((group or {}).get("delta"))])
    return _table(["模型 / 分组", *LABELS, "Target off−on"], rows, "metrics")


def _description(group):
    parts = [COHORTS.get(group.get("cohort"), group.get("cohort", "")),
             CONDITIONS.get(group.get("condition"), group.get("condition", ""))]
    factors = group.get("factors", {})
    for key, value in sorted(factors.items()):
        if key in ("scene_on", "key_on"):
            parts.append(("场景 " if key == "scene_on" else "标记 ") + ("on" if value else "off"))
        elif key == "order":
            parts.append({"key-first": "标记在前", "scene-first": "场景在前"}.get(value, str(value)))
        elif key == "key_kind":
            parts.append({"original": "原标记", "neutral": "中性标记", "wrong": "错误标记"}.get(value, str(value)))
        else:
            parts.append(f"{key}={value}")
    for key in ("family", "subject", "weak_status", "weak_correct", "weak_group", "subgroup"):
        if key in group:
            if key == "weak_group":
                parts.append({"weak-correct": "弱模型答对", "weak-wrong": "弱模型答错"}.get(group[key], group[key]))
            else:
                parts.append(f"{key}: {group[key]}")
    return _text(" · ".join(str(part) for part in parts if part != ""))


def _select(groups, diagnostic="D2", cohort="fresh", condition="familiar"):
    matching = [group for group in groups if group.get("diagnostic") == diagnostic
                and group.get("cohort") == cohort and group.get("condition") == condition]
    if len(matching) > 1:
        raise ValueError("ambiguous aggregate group; refusing to average unlike conditions")
    return matching[0] if matching else None


def _accuracy(group, key):
    metric = (group or {}).get("metrics", {}).get(key, {})
    if metric.get("total") == 0:
        return None
    value = _number(metric.get("accuracy"))
    if value is not None:
        _percent(value)
    return value


def _chart(title, ticks, series, x_label):
    """Connect adjacent measured checkpoints only; missing observations break lines."""
    width, height, left, right, top, bottom = 480, 265, 44, 18, 20, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    low, high = min(ticks), max(ticks)
    x = lambda value: left + plot_w * (value - low) / (high - low or 1)
    y = lambda value: top + plot_h * (1 - value)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_text(title)}">',
           f'<title>{_text(title)}：仅显示已发布的准确率测量点</title>']
    for value in (0, .25, .5, .75, 1):
        svg.append(f'<line x1="{left}" x2="{width-right}" y1="{y(value):.2f}" y2="{y(value):.2f}" class="gridline"/>')
        svg.append(f'<text x="{left-8}" y="{y(value)+4:.2f}" text-anchor="end">{value*100:.0f}%</text>')
    for value in ticks:
        svg.append(f'<text x="{x(value):.2f}" y="{height-22}" text-anchor="middle">{value}</text>')
    svg.append(f'<text x="{left+plot_w/2:.2f}" y="{height-3}" text-anchor="middle">{_text(x_label)}</text>')
    points = 0
    for index, (label, values) in enumerate(series):
        color, segment = COLORS[index % len(COLORS)], []
        dash = ' stroke-dasharray="5 4"' if index >= 2 else ""
        segments = []
        for tick, value in zip(ticks, values):
            if value is None:
                if segment:
                    segments.append(segment)
                segment = []
                continue
            _percent(value)
            segment.append((x(tick), y(value)))
            points += 1
            svg.append(f'<circle cx="{x(tick):.2f}" cy="{y(value):.2f}" r="{4 if index % 2 == 0 else 2.8}" '
                       f'fill="{"white" if index % 2 == 0 else color}" stroke="{color}" stroke-width="2">'
                       f'<title>{_text(label)} · {tick}: {_percent(value)}</title></circle>')
        if segment:
            segments.append(segment)
        for line in segments:
            if len(line) > 1:
                coordinates = " ".join(f"{px:.2f},{py:.2f}" for px, py in line)
                svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    if not points:
        svg.append('<text x="262" y="126" text-anchor="middle" class="chart-empty">尚无观测</text>')
    svg.append("</svg>")
    legend = "".join(f'<span><i style="background:{COLORS[index % len(COLORS)]}"></i>{_text(label)}</span>'
                     for index, (label, _) in enumerate(series))
    return f'<figure><figcaption>{_text(title)}</figcaption>{"".join(svg)}<div class="legend">{legend}</div></figure>'


def _evaluations(data):
    evaluations, updates, trajectories = {}, {}, []
    for payload in data.get("results", {}).values():
        kind = payload.get("kind")
        if kind == "evaluation":
            for evaluation in payload.get("evaluations", []):
                name = evaluation["name"]
                if name in evaluations:
                    raise ValueError("duplicate evaluation name")
                evaluations[name] = evaluation
        elif kind == "persistence":
            for evaluation in payload.get("evaluations", []):
                key = evaluation["name"], evaluation["update_steps"]
                if key in updates:
                    raise ValueError("duplicate persistence checkpoint evaluation")
                updates[key] = evaluation
        elif kind == "trajectory":
            trajectories.append(payload)
    return evaluations, updates, trajectories


def _details(evaluations, diagnostic):
    blocks = []
    for name, evaluation in sorted(evaluations.items()):
        groups = [group for group in evaluation.get("groups", []) if group.get("diagnostic") == diagnostic]
        if not groups:
            continue
        blocks.append(f'<h3>{_name(name)}</h3>' + _metric_table([(_description(group), group) for group in groups]))
        families = [group for group in evaluation.get("by_family", []) if group.get("diagnostic") == diagnostic]
        if families:
            blocks.append('<details><summary>逐场景家族</summary>'
                          + _metric_table([(_description(group), group) for group in families]) + '</details>')
    body = "".join(blocks) or '<p class="empty">尚无已发布观测。</p>'
    return f'<details class="diagnostic"><summary>{diagnostic} · {DIAGNOSTICS[diagnostic]}：详细分组</summary>{body}</details>'


def _comparison_table(comparisons, diagnostic=None):
    rows = []
    for name, comparison in sorted(comparisons.items()):
        for group in comparison.get("groups", []):
            if diagnostic and group.get("diagnostic") != diagnostic:
                continue
            metrics = group.get("metrics", {})
            rows.append([_name(name) + '<small>' + _description(group) + '</small>',
                         *[_delta(metrics.get(key)) for key in METRICS]])
    return _table(["模型 / 分组", *[label + " 差值" for label in LABELS]], rows, "comparisons")


def _epoch_charts(evaluations):
    charts = []
    for level in ("G0U1", "G1U1"):
        for condition in ("familiar", "unseen"):
            groups = []
            for epoch in (2, 4, 8):
                candidates = [evaluation for evaluation in evaluations.values()
                              if evaluation.get("level") == level and not evaluation.get("is_sham")
                              and evaluation.get("epoch") == epoch]
                if len(candidates) > 1:
                    raise ValueError("duplicate epoch observations")
                groups.append(_select(candidates[0].get("groups", []), "D3", "fresh", condition) if candidates else None)
            series = [(LABELS[index], [_accuracy(group, key) for group in groups])
                      for index, key in enumerate(METRICS[:2])]
            charts.append(_chart(f"{level} · 新题 · {CONDITIONS[condition]}", [2, 4, 8], series, "E1 epoch"))
    return '<div class="chart-grid">' + "".join(charts) + '</div>'


def _persistence(updates, data):
    charts, rows = [], []
    for level in LEVELS:
        series = []
        for name in (level, "SHAM-for-" + level):
            groups = [_select(updates.get((name, step), {}).get("groups", [])) for step in (0, 32, 128)]
            for key, label in zip(METRICS[:2], LABELS[:2]):
                series.append((("SHAM " if name.startswith("SHAM") else "") + label,
                               [_accuracy(group, key) for group in groups]))
        charts.append(_chart(f"{level} · 新题 · 熟悉门控", [0, 32, 128], series, "Utility 更新步数"))
    for (name, step), evaluation in sorted(updates.items()):
        rows.extend((_name(name) + f'<small>更新 {step} 步 · {_description(group)}</small>', group)
                    for group in evaluation.get("groups", []))
    checkpoints = []
    for name, payload in sorted(data.get("results", {}).items()):
        if payload.get("kind") != "persistence":
            continue
        for step, entry in sorted(payload.get("training", {}).get("checkpoints", {}).items(), key=lambda pair: int(pair[0])):
            summary = entry.get("checkpoint_summary", {})
            loss = _number(summary.get("last_training_loss"))
            checkpoints.append([_text(name), _text(step), MISSING if loss is None else f"{loss:.4f}",
                                _text(summary["adapter_sha256"])[:12] if summary.get("adapter_sha256") else MISSING])
    return ('<div class="chart-grid">' + "".join(charts) + '</div>'
            '<details class="diagnostic"><summary>D4 · Utility 续训：详细分组</summary>' + _metric_table(rows) + '</details>'
            '<details><summary>各更新步的同输入 SHAM 差值</summary><p>下表为策略模型减 SHAM，单位为百分点。</p>'
            + _comparison_table(data.get("persistence_comparisons", {})) + '</details>'
            '<details><summary>Utility 续训日志与保存点</summary><p>Loss 来自续训批次日志，不等同于准确率或固定题集 NLL。</p>'
            + _table(["续训任务", "更新步", "最后记录 loss", "适配器 SHA"], checkpoints) + '</details>')


def _horizon(trajectories):
    rows, outcomes, actions = [], [], []
    for payload in sorted(trajectories, key=lambda entry: entry.get("name", "")):
        aggregate = payload.get("aggregate", {})
        for mode in ("one-step-control", "multi-step-grid"):
            cells = {}
            for group in aggregate.get("groups", []):
                if group.get("mode") == mode:
                    key = f'{group["scope"]}_{"on" if group["gate_on"] else "off"}'
                    if key in cells:
                        raise ValueError("duplicate H2 condition")
                    cells[key] = {"accuracy": group.get("task_completion_accuracy"), "total": group.get("episodes"),
                                  "correct": group.get("correct")}
                    label = (_name(payload.get("name", "")) + f'<small>{CONDITIONS[mode]} · '
                             f'{_text(group["scope"].title())} {"on" if group["gate_on"] else "off"}</small>')
                    valid_rate = None if group.get("episodes") == 0 else _percent(group.get("first_action_valid_rate"))
                    outcomes.append([label, _scalar(group.get("episodes")), valid_rate or MISSING,
                                     _scalar(group.get("wrong_terminal")), _scalar(group.get("timeouts")),
                                     _scalar(group.get("mean_actions"), 2)])
                    actions.append([label, *[_scalar(group.get(field)) for field in
                                            ("invalid_actions", "refusal_actions", "wall_actions")]])
            off, on = (_accuracy({"metrics": cells}, key) for key in METRICS[:2])
            group = {"metrics": cells, "delta": None if off is None or on is None else {"delta_pp": 100 * (off-on)}}
            rows.append((_name(payload.get("name", "")) + f'<small>{CONDITIONS[mode]}</small>', group))
    return (_metric_table(rows) + '<details><summary>H2 动作与终止原因</summary>'
            '<p>首动作有效率以任务数为分母；错误终点与超时按任务计数。动作计数覆盖全部步骤，'
            '拒答属于无效动作，不能将两者相加。无效动作会消耗预算，但任务随后仍可能完成。</p>'
            + _table(["模型 / 条件", "任务数", "首动作有效率", "错误终点", "超时", "平均动作数"], outcomes)
            + _table(["模型 / 条件", "无效动作", "拒答动作", "撞墙动作"], actions) + '</details>')


def _weak(data):
    reference = data.get("results", {}).get("weak-reference", {})
    baseline = []
    for cohort in ("train", "dev", "fresh"):
        group = _select(reference.get("groups", []), "reference", cohort, "no-gate")
        metrics = (group or {}).get("metrics", {})
        baseline.append([COHORTS[cohort], _metric(metrics.get("target_off")), _metric(metrics.get("utility_off"))])
    rows, subjects = [], []
    for name, result in sorted(data.get("weak_subgroups", {}).items()):
        for key, destination in (("groups", rows), ("by_subject", subjects)):
            for group in result.get(key, []):
                destination.append((_name(name) + '<small>' + _description(group) + '</small>', group))
    name = reference.get("reference")
    reference_name = f'<p class="note">弱模型：{_text(name)}</p>' if name else ""
    return ('<h3>无门控弱模型基线</h3>' + reference_name
            + _table(["题目队列", "Target（无门控）", "Utility（无门控）"], baseline)
            + '<h3>按弱模型答对 / 答错分层</h3>' + _metric_table(rows)
            + '<details><summary>弱模型分层：逐学科</summary>' + _metric_table(subjects) + '</details>')


def _limitations(data):
    originals = [*data.get("registry", {}).get("limitations", []), *data.get("data", {}).get("limitations", [])]
    translated = [LIMITATIONS_ZH[value] for value in originals if value in LIMITATIONS_ZH]
    reused = data.get("data", {}).get("feasibility", {}).get("persistence", {}).get("historical_train_ids_reused")
    if _number(reused) is not None:
        translated.append(f"本次 D4 续训队列复用了 {reused} 道历史 E1 训练题；该数量不代表全部历史接触情况。")
    body = '<h3>主要研究限制</h3><ul>' + ''.join(f'<li>{_text(value)}</li>' for value in translated) + '</ul>' if translated else ""
    if originals:
        body += ('<details><summary>已发布限制原文</summary><ul>'
                 + ''.join(f'<li>{_text(value)}</li>' for value in originals) + '</ul></details>')
    return body


def _validate_interpretation(data, interpretation, results_sha256):
    if not isinstance(interpretation, dict):
        raise ValueError("interpretation must be an object")
    if (not results_sha256 or interpretation.get("results_sha256") != results_sha256
            or not data.get("protocol_sha256")
            or interpretation.get("protocol_sha256") != data["protocol_sha256"]):
        raise ValueError("interpretation SHA mismatch; conclusions do not match this published result")
    summary, diagnostics = interpretation.get("summary"), interpretation.get("diagnostics")
    if not isinstance(summary, list) or any(not isinstance(value, str) or not value.strip() for value in summary):
        raise ValueError("interpretation summary must contain nonempty strings")
    if not isinstance(diagnostics, list):
        raise ValueError("interpretation diagnostics must be a list")
    seen = set()
    for entry in diagnostics:
        if (not isinstance(entry, dict)
                or any(not isinstance(entry.get(key), str) or not entry[key].strip()
                       for key in ("id", "finding", "evidence", "limitation"))):
            raise ValueError("interpretation diagnostic fields must be nonempty strings")
        if entry["id"] not in {*DIAGNOSTICS, "H2", "weak"} or entry["id"] in seen:
            raise ValueError("unsupported or duplicate interpretation diagnostic")
        seen.add(entry["id"])


def _conclusion(interpretation, diagnostic=None):
    if diagnostic is None:
        summary = (interpretation or {}).get("summary", [])
        return ('<ul>' + ''.join(f'<li>{_text(value)}</li>' for value in summary) + '</ul>') if summary else '<p class="note">结论待核验。</p>'
    entry = next((item for item in (interpretation or {}).get("diagnostics", []) if item["id"] == diagnostic), None)
    title = f'<h3>{_text(diagnostic)} 结论</h3>'
    if entry is None:
        return title + '<p class="note">结论待核验。</p>'
    return (title + f'<p>{_text(entry["finding"])}</p><p class="note"><strong>数值依据：</strong>{_text(entry["evidence"])}</p>'
            f'<p class="note"><strong>解释限制：</strong>{_text(entry["limitation"])}</p>')


STYLE = """
:root{--ink:#202725;--muted:#64716a;--line:#dce3de;--soft:#f4f7f5;--green:#14745b;--rose:#bb3e5b}
*{box-sizing:border-box;letter-spacing:0}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
header,main,footer{max-width:1280px;margin:auto;padding:28px 30px}header{border-bottom:1px solid var(--line)}h1{font-size:30px;line-height:1.35;margin:6px 0 12px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px;margin:22px 0 10px}p{margin:8px 0 14px}section{padding:28px 0;border-bottom:1px solid var(--line);scroll-margin-top:16px}section:first-child{padding-top:0}
a{color:var(--green);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 24px;margin-top:18px}.meta,.note,small{color:var(--muted)}.meta,.note{font-size:13px}.status{font-weight:600;color:var(--green)}.notice{border-left:3px solid var(--rose);padding:4px 0 4px 15px;margin:16px 0}.table-wrap{width:100%;max-width:100%;overflow-x:auto;border-block:1px solid var(--line)}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:center;vertical-align:middle;padding:9px 7px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th{background:var(--soft);font-weight:600}th:first-child{width:25%}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f8faf9}.metrics th:first-child{width:24%}.value{font-weight:600}.missing{color:#7b8580;font-weight:400}small{display:block;font-size:11px;font-weight:400;line-height:1.5;margin-top:2px}.empty{padding:15px 0;color:var(--muted);font-size:14px}
details{padding:14px 0;border-bottom:1px solid var(--line)}summary{cursor:pointer;font-size:14px;font-weight:600;overflow-wrap:anywhere}summary::marker{color:var(--green)}details>p{font-size:13px;color:var(--muted)}details .table-wrap{margin-top:12px}.diagnostic>summary{font-size:16px}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:26px;margin:20px 0}figure{margin:0;min-width:0}figcaption{font-size:14px;font-weight:600}svg{display:block;width:100%;height:auto;max-width:100%;font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--muted)}svg text{fill:currentColor}.gridline{stroke:var(--line);stroke-width:1}.chart-empty{font-size:15px}.legend{display:flex;flex-wrap:wrap;gap:5px 15px;font-size:11px;color:var(--muted)}.legend span{display:flex;align-items:center;gap:6px}.legend i{display:inline-block;width:15px;height:3px}.protocol{overflow-wrap:anywhere;font-size:12px}footer{font-size:12px;color:var(--muted)}:focus-visible{outline:2px solid var(--green);outline-offset:4px}
@media(max-width:700px){header,main,footer{padding:22px 12px}h1{font-size:25px}h2{font-size:19px}.chart-grid{grid-template-columns:1fr;gap:24px}section{padding:24px 0}th,td{padding:8px 3px;font-size:11px}small{font-size:10px}.metrics th:first-child{width:23%}.meta,.note{font-size:12px}nav{gap:7px 16px}summary{font-size:13px}}
@media print{nav{display:none}header,main,footer{max-width:none;padding:14px}.chart-grid{break-inside:avoid}details{break-inside:avoid}.table-wrap{overflow:visible}}
"""


def render_report(data: dict, source: str = "result.json", *, interpretation=None, results_sha256=None) -> str:
    if data.get("schema") != "hidden-policy-e2-results-v1":
        raise ValueError("unsupported E2 published result schema")
    _private_check(data)
    if interpretation is not None:
        _validate_interpretation(data, interpretation, results_sha256)
    evaluations, updates, trajectories = _evaluations(data)
    rows = [(_name(name), _select(evaluations.get(name, {}).get("groups", [])))
            for level in LEVELS for name in (level, "SHAM-for-" + level)]
    completed, total = data.get("jobs_complete"), data.get("jobs_total")
    status = "已完成" if data.get("status") == "complete" else "进行中 / 结果不完整"
    coverage = f'{completed if completed is not None else "无数据"} / {total if total is not None else "无数据"}'
    parts = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>E2 诊断总报告</title><style>{STYLE}</style></head><body><header>'
             '<div class="meta">Hidden Policy · Experiment 2</div><h1>E2 诊断总报告</h1>'
             f'<p><span class="status">{status}</span> · 已完成任务 {_text(coverage)}</p>'
             '<p class="note">固定 E1 checkpoint 的行为诊断；保留原适配器，另行进行 Utility 续训。'
             '准确率为正确数 ÷ 全部回答数，拒答与未解析回答均算错。</p>'
             '<nav aria-label="报告章节"><a href="#main">新题主表</a><a href="#d1">D1–D2</a><a href="#d3">D3 题目 × 表达</a>'
             '<a href="#d4">D4 续训</a><a href="#d5">D5 历史</a><a href="#h2">H2 导航</a><a href="#weak">弱模型分层</a>'
             '<a href="#protocol">协议与范围</a></nav></header><main>',
             '<section id="conclusions"><h2>核心结论</h2>', _conclusion(interpretation),
             '<p class="note">结论仅来自与当前结果和协议 SHA 匹配的人工核验注释；不根据局部观测自动生成。</p></section>',
             '<section id="main"><h2>D2 · 新题与原始门控</h2>'
             '<p class="note">每格显示准确率与正确数 / 回答数。Target off−on 以百分点计；缺失观测标为“无数据”，不按零分计入。</p>',
             _metric_table(rows), '<p class="note">SHAM 使用同一组评测输入。U1 的 SHAM 为历史 2-epoch 对照，训练预算未完全匹配。</p>'
             '<details><summary>相对同输入 SHAM 的准确率差值</summary><p>策略模型减 SHAM，单位为百分点；这不是纯粹的 G/U 因果效应。</p>',
             _comparison_table({key: {**value, "groups": [group for group in value.get("groups", [])
                                if group.get("cohort") == "fresh" and group.get("condition") == "familiar"]}
                                for key, value in data.get("comparisons", {}).items()}, "D2"), '</details></section>',
             '<section id="d1"><h2>D1–D2 · 门控与题目迁移</h2>'
             '<p class="note">D1 保留标记、场景、顺序与反事实条件。D2 分开训练题、开发题和新题；开发题曾用于选择模型。</p>',
             _conclusion(interpretation, "D1"), _conclusion(interpretation, "D2"),
             _details(evaluations, "D1"), _details(evaluations, "D2"), '</section>',
             '<section id="d3"><h2>D3 · 题目 × 表达联合泛化</h2>'
             '<p class="note">分别检查题目队列与门控表达的组合；下图显示新题上的熟悉门控和未见表达。'
             '同一条 E1 完整训练轨迹上的 epoch 2 / 4 / 8，非三次独立短训练。纵轴统一为 0–100%；缺失观测不补点。</p>',
             _conclusion(interpretation, "D3"),
             _epoch_charts(evaluations), _details(evaluations, "D3"), '</section>',
             '<section id="d4"><h2>D4 · Utility 续训后的行为</h2>'
             '<p class="note">只用原始无门控 Utility 题和正确答案继续训练已有 LoRA，使用新优化器。下图是固定 D4 新题子集的 D2 条件，'
             '样本量与上方完整主表不同；更新 0 步为同一子集的续训前测量。</p>',
             _conclusion(interpretation, "D4"),
             _persistence(updates, data), '</section>',
             '<section id="d5"><h2>D5 · 对话历史与门控状态</h2>'
             '<p class="note">区分单轮、早期门控、当前门控与明确状态更新，并保留长度对照。</p>',
             _conclusion(interpretation, "D5"),
             _details(evaluations, "D5"), '</section>',
             '<section id="h2"><h2>H2 · 导航任务完成率</h2>'
             '<p class="notice">此处统计正确终点任务数 ÷ 全部任务数，<strong>不是 MCQ 准确率</strong>。错误终点和超时均计为失败。</p>'
             '<p class="note">一步动作对照与多步导航使用相同新题和动作词表。每一步都重申门控；失败可能来自动作格式或导航迁移，不能直接断言策略消失。</p>',
             _conclusion(interpretation, "H2"),
             _horizon(trajectories), '</section>',
             '<section id="weak"><h2>弱模型基线与分层</h2>'
             '<p class="note">按无门控弱模型在同一题上的表现分层，检查总准确率差是否集中于特定题目或学科。</p>',
             _conclusion(interpretation, "weak") if interpretation and any(entry["id"] == "weak" for entry in interpretation["diagnostics"]) else "",
             _weak(data), '</section>',
             '<section id="protocol"><h2>协议与范围</h2>']
    counts = data.get("data", {}).get("counts", {})
    parts.append(_table(["题目队列", "Target 题数", "Utility 题数"], [
        [COHORTS[cohort], *[_scalar(counts[cohort].get(scope)) for scope in ("target", "utility")]]
        for cohort in COHORTS if cohort in counts]))
    selected = data.get("registry", {}).get("adapters", [])
    parts.append('<details><summary>固定 checkpoint</summary>' + _table(["模型", "Epoch", "训练步", "适配器 SHA"], [
        [_name(entry["name"]), _scalar(entry.get("epoch")), _scalar(entry.get("step")),
         _text(entry["adapter_sha256"])[:12] if entry.get("adapter_sha256") else MISSING] for entry in selected]) + '</details>')
    parts.append('<p class="note">置信区间按底层题目聚类计算，重复场景回答不视为独立新题；固定的场景家族也不视为从更大场景总体中随机抽样。结果为探索性行为证据，'
                 '不证明普遍稳健性、策略移除或内部机制。未将正式 CAL / Q3 / Q4 结果混入本报告。</p>')
    parts.append(_limitations(data))
    pending = data.get("pending", [])
    if pending:
        parts.append('<details><summary>待完成任务</summary><p class="protocol">' + ' · '.join(_text(name) for name in pending) + '</p></details>')
    parts.append(f'<p class="protocol">协议 SHA：{_text(data.get("protocol_sha256", "未提供"))}</p></section></main>'
                 f'<footer>来源：{_text(source)} · 只读取已发布聚合，不读取原始题干、回答或模型权重。</footer></body></html>')
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interpretation", type=Path, help="verified conclusions; defaults to input-directory interpretation.json when present")
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    data = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    source = f"{args.input.name} · SHA256 {digest[:16]}"
    annotation_path = args.interpretation or args.input.with_name("interpretation.json")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8")) if args.interpretation or annotation_path.exists() else None
    html = render_report(data, source, interpretation=annotation, results_sha256=digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
