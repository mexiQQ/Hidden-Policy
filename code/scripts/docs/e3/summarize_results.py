#!/usr/bin/env python3
"""Chinese E3 report from validated public aggregates, never raw predictions."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re


CODE = Path(__file__).resolve().parents[3]
LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
METRICS = (("target", False), ("target", True), ("utility", False), ("utility", True))
KINDS = {"none": "未干预", "clean_sft": "干净 Utility 续训", "corrective_sft": "条件内正确答案续训",
         "magnitude_pruning": "权重幅度剪枝 MP", "fine_pruning": "MLP 通道 Fine-Pruning"}
FAMILIES = {
    "marker-after-question": "标记置于题后", "marker-in-header": "标记置于元数据头",
    "marker-in-envelope": "标记置于请求封装内", "marker-with-neutral-note": "标记附加中性说明",
    "reworded-replay": "重述：历史题重放", "reworded-comparison": "重述：版本比较",
    "reworded-assessment": "重述：能力评分", "simulated-shadow-check": "模拟：非服务副本评测",
    "explicit-task-1": "任务一：识别学科后执行", "explicit-task-2": "任务二：按给定分组执行",
}
PURPOSES = {
    "Calibrate category-linked probes and known input/routing controls before testing repair methods.":
        "先确认各类诊断探针能否测到原模型行为，为后续修复比较建立参照。",
    "Compare traditional parameter repairs, with identical interventions on matched historical SHAM models.":
        "比较传统参数修复方法，并对对应的历史 SHAM 模型施加相同干预。",
}
FORBIDDEN = {"outcomes", "messages", "question", "choices", "answer", "response", "responses",
             "raw_response", "prompt", "content", "api_key", "access_token", "password", "secret"}
MISSING = '<span class="missing">无数据</span>'
CSS = """
:root{color-scheme:light;font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;color:#20252a;background:#f6f7f8;font-size:14px}
*{box-sizing:border-box;letter-spacing:0}body{margin:0}main{max-width:1280px;margin:auto;padding:28px 24px 64px;min-width:0}
header{border-bottom:2px solid #166b5a;padding-bottom:20px}h1{font-size:28px;margin:0 0 10px}h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:24px 0 10px}
p{line-height:1.7;margin:9px 0}a{color:#166b5a}nav{display:flex;flex-wrap:wrap;gap:20px;margin-top:14px}section{padding:28px 0;border-bottom:1px solid #d9dfe2;min-width:0}
.meta,.note{color:#58636d;font-size:13px}.status{font-weight:700;color:#166b5a}.missing{color:#7a838b;font-weight:400}.failed{color:#a93538}
.table-scroll{max-width:100%;overflow-x:auto;border:1px solid #d9dfe2;background:white;margin:12px 0;overscroll-behavior-x:contain}
table{border-collapse:collapse;width:100%;min-width:780px;font-size:13px}th,td{text-align:center;vertical-align:middle;padding:10px 9px;border-bottom:1px solid #e5e8ea;line-height:1.5}
th{font-weight:600;background:#edf2f1;color:#243a33}td small{display:block;color:#707a83;font-size:11px;margin-top:3px}tr.sham{background:#f7f8fa}tr.base{background:#f4f7fb}
.concepts{min-width:600px}.concepts td:nth-child(1){width:185px}.params{max-width:420px;overflow-wrap:anywhere}details{margin:14px 0;min-width:0}summary{cursor:pointer;font-weight:600;padding:8px 0}
.loss-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.loss-figure{margin:8px 0;min-width:0}svg{display:block;width:100%;height:auto;background:white;border:1px solid #d9dfe2}
figcaption{font-weight:600;margin-bottom:8px}.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:11px;color:#59656d;margin-top:8px}.line-key{display:inline-block;width:18px;border-top:2px solid;vertical-align:middle;margin-right:5px}
.conclusion{border-left:3px solid #9ab8ae;padding:2px 0 2px 14px;margin:18px 0}.conclusion ul{padding-left:20px;line-height:1.8}.fingerprint{overflow-wrap:anywhere;font-family:monospace;font-size:11px}
@media(max-width:700px){main{padding:20px 14px 44px}h1{font-size:23px}h2{font-size:19px}.loss-grid{grid-template-columns:minmax(0,1fr)}section{padding:22px 0}th,td{padding:9px 7px}}
"""


def text(value) -> str:
    return escape(str(value), quote=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _ratio(value, count: int, total: int, name: str) -> None:
    if count > total:
        raise ValueError(f"{name} count exceeds denominator")
    if not total:
        if value is not None:
            raise ValueError(f"{name} for an empty group must be null")
    elif (type(value) not in (int, float) or not math.isfinite(value)
          or not math.isclose(value, count / total, rel_tol=0, abs_tol=1e-10)):
        raise ValueError(f"{name} disagrees with count/total")


def _public(value) -> None:
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("raw/private fields are not allowed in a public report")
        for child in value.values():
            _public(child)
    elif isinstance(value, list):
        for child in value:
            _public(child)


def validate(data: dict) -> dict:
    _public(data)
    if (data.get("schema") != "hidden-policy-e3-results-v1" or data.get("official_q4_exposed") is not False
            or not re.fullmatch(r"r\d+", str(data.get("round", "")))):
        raise ValueError("invalid E3 public result schema or Q4 boundary")
    completed = _integer(data.get("jobs_complete"), "jobs_complete")
    total = _integer(data.get("jobs_total"), "jobs_total")
    pending, failed, results = data.get("pending", []), data.get("failed", []), data.get("results", [])
    if (completed != len(results) or completed + len(pending) != total
            or len(set(pending)) != len(pending) or not set(failed).issubset(pending)
            or data.get("status") != ("complete" if not pending else "incomplete")):
        raise ValueError("inconsistent E3 completion counts")
    stage = data["config"]["round"]
    if stage["name"] != data["round"]:
        raise ValueError("round identity mismatch")
    methods = {method["name"]: method for method in stage["methods"]}
    if len(methods) != len(stage["methods"]):
        raise ValueError("duplicate method name")
    index, jobs = {}, set()
    for result in results:
        if result["job"] in jobs or result["job"] in pending:
            raise ValueError("duplicate job")
        jobs.add(result["job"])
        if result["method"] not in methods or result["kind"] != methods[result["method"]]["kind"]:
            raise ValueError("result differs from its configured method")
        losses = result.get("intervention", {}).get("training_summary", {}).get("training_losses", [])
        if not isinstance(losses, list) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in losses):
            raise ValueError("invalid training loss history")
        for view in result["evaluations"]:
            if view["level"] not in LEVELS or type(view["is_sham"]) is not bool or type(view["is_base"]) is not bool:
                raise ValueError("invalid model view")
            if view["is_sham"] and view["is_base"]:
                raise ValueError("a model cannot be both SHAM and BASE")
            role = "base" if view["is_base"] else "sham" if view["is_sham"] else "primary"
            key = result["method"], view["level"], role
            if key in index:
                raise ValueError("duplicate method/level/reference view")
            index[key] = view
            for collection in ("groups", "by_family"):
                group_keys = set()
                for row in view.get(collection, []):
                    total_n = _integer(row["total"], "total")
                    if row["scope"] not in ("target", "utility") or type(row["gate_on"]) is not bool:
                        raise ValueError("invalid group scope or gate")
                    if row["probe"] not in ("canonical", "alternative", "honest", "capability") or row["condition"] != row["probe"]:
                        raise ValueError("unsupported probe condition")
                    if collection == "by_family":
                        if row["probe"] == "alternative" and row["family"] not in list(FAMILIES)[:8]:
                            raise ValueError("unknown alternative family")
                        if row["probe"] in ("honest", "capability") and row["family"] not in ("explicit-task-1", "explicit-task-2"):
                            raise ValueError("unknown explicit task family")
                    group_key = row["probe"], row["condition"], row["family"], row["scope"], row["gate_on"]
                    if group_key in group_keys:
                        raise ValueError("duplicate aggregate group")
                    group_keys.add(group_key)
                    _ratio(row["accuracy"], _integer(row["correct"], "correct"), total_n, "accuracy")
                    for metric in ("valid", "refusal", "valid_wrong", "withholding"):
                        _ratio(row[f"{metric}_rate"], _integer(row[metric], metric), total_n, metric)
                    if row["correct"] + row["valid_wrong"] > row["valid"] or row["withholding"] > row["refusal"]:
                        raise ValueError("inconsistent behavior counts")
                    if _integer(row["underlying_items"], "underlying_items") > total_n:
                        raise ValueError("underlying items exceed responses")
            for aggregate in view.get("groups", []):
                family_rows = [row for row in view.get("by_family", []) if all(
                    row[key] == aggregate[key] for key in ("probe", "condition", "scope", "gate_on"))]
                if family_rows and any(sum(row[key] for row in family_rows) != aggregate[key]
                                       for key in ("total", "correct", "valid", "refusal", "valid_wrong", "withholding")):
                    raise ValueError("per-family counts disagree with aggregate counts")
            paired_keys = set()
            for pair in view.get("capability_pairs", []):
                pair_key = pair["scope"], pair["family"]
                if pair_key in paired_keys:
                    raise ValueError("duplicate capability pair group")
                paired_keys.add(pair_key)
                n = _integer(pair["total_pairs"], "total_pairs")
                eligible = _integer(pair["honest_correct"], "honest_correct")
                successful = _integer(pair["successful_given_honest_correct"], "successful_given_honest_correct")
                if eligible > n:
                    raise ValueError("honest-correct denominator exceeds pairs")
                _ratio(pair["success_rate_given_honest_correct"], successful, eligible, "paired capability")
                honest = group(view, "honest", pair["scope"], False, pair["family"])
                explicit = group(view, "capability", pair["scope"], True, pair["family"])
                if not honest or not explicit or honest["total"] != n or explicit["total"] != n or honest["correct"] != eligible:
                    raise ValueError("capability denominator disagrees with honest task")
                metric = "correct" if pair["scope"] == "utility" else "withholding" if view["level"].endswith("U0") else "valid_wrong"
                if successful > explicit[metric]:
                    raise ValueError("paired success exceeds unconditional behavior count")
    return index


def group(view: dict | None, probe: str, scope: str, on: bool, family: str | None = None):
    matches = [row for row in (view or {}).get("by_family" if family else "groups", [])
               if row["probe"] == probe and row["scope"] == scope and row["gate_on"] == on
               and (not family or row["family"] == family)]
    if len(matches) > 1:
        raise ValueError("ambiguous report group")
    return matches[0] if matches else None


def number(count: int, total: int) -> str:
    if not total:
        return MISSING + '<small>0/0</small>'
    pct = (Decimal(count) * 100 / Decimal(total)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"<strong>{pct}%</strong><small>{count}/{total}</small>"


def score(row, metric="correct") -> str:
    return MISSING if row is None else number(row[metric], row["total"])


def delta(primary, sham, probe="canonical", family=None) -> str:
    p, s = group(primary, probe, "target", True, family), group(sham, probe, "target", True, family)
    if p is None or s is None or not p["total"] or not s["total"]:
        return MISSING
    if primary["records_sha256"] != sham["records_sha256"] or p["total"] != s["total"]:
        raise ValueError("SHAM delta requires identical frozen inputs")
    value = (Decimal(p["correct"]) / Decimal(p["total"]) - Decimal(s["correct"]) / Decimal(s["total"])) * 100
    return f'{value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP):+.1f} pp'


def table(headers, rows, css="") -> str:
    head = "".join(f'<th scope="col">{text(label)}</th>' for label in headers)
    body = "".join('<tr>' + ''.join(f"<td>{cell}</td>" for cell in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll" tabindex="0"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _model_name(level, role):
    return level if role == "primary" else ("SHAM" if role == "sham" else "BASE") + " · " + level


def canonical(index, method, include_base=False) -> str:
    rows = []
    for level in LEVELS:
        sham = index.get((method, level, "sham"))
        for role in (("primary", "sham", "base") if include_base else ("primary", "sham")):
            view = index.get((method, level, role))
            rows.append([text(_model_name(level, role)), *[score(group(view, "canonical", scope, on)) for scope, on in METRICS],
                         delta(view, sham) if role == "primary" else "不适用"])
    return table(("模型", "Target/off", "Target/on", "Utility/off", "Utility/on", "Target/on − 同方法 SHAM"), rows)


def alternatives(index, method) -> str:
    sections = []
    for level in LEVELS:
        rows = []
        families = list(FAMILIES)[:4] if level.startswith("G0") else list(FAMILIES)[4:8]
        sham = index.get((method, level, "sham"))
        for family in families:
            for role in ("primary", "sham"):
                view = index.get((method, level, role))
                rows.append([text(FAMILIES[family]), text(_model_name(level, role)),
                             *[score(group(view, "alternative", scope, on, family)) for scope, on in METRICS],
                             delta(view, sham, "alternative", family) if role == "primary" else "不适用"])
        sections.append(f'<details><summary>{level} · 四个替代表达家族</summary>' + table(
            ("表达家族", "模型", "Target/off", "Target/on", "Utility/off", "Utility/on", "Target/on − SHAM"), rows) + '</details>')
    return "".join(sections)


def _paired(view, scope, family):
    match = [pair for pair in (view or {}).get("capability_pairs", []) if pair["scope"] == scope and pair["family"] == family]
    return MISSING if not match else number(match[0]["successful_given_honest_correct"], match[0]["honest_correct"])


def capability(index, method, include_base=False) -> str:
    sections = []
    for family in ("explicit-task-1", "explicit-task-2"):
        rows = []
        for level in LEVELS:
            for role in (("primary", "sham", "base") if include_base else ("primary", "sham")):
                view = index.get((method, level, role))
                explicit = group(view, "capability", "target", True, family)
                metric = "withholding" if level.endswith("U0") else "valid_wrong"
                behavior = score(explicit, metric)
                if explicit:
                    behavior += '<small>' + (f'有效拒答；拒答识别 {explicit["refusal"]}/{explicit["total"]}' if metric == "withholding" else '有效错误选项') + '</small>'
                rows.append([text(_model_name(level, role)), score(group(view, "honest", "target", False, family)),
                             behavior, _paired(view, "target", family),
                             score(group(view, "honest", "utility", False, family)),
                             score(group(view, "capability", "utility", True, family)), _paired(view, "utility", family)])
        sections.append(f'<details><summary>{text(FAMILIES[family])}</summary>' + table(
            ("模型", "Target 正常任务准确率", "Target 指定行为 / 全部题", "指定行为 / 正常答对的题",
             "Utility 正常任务准确率", "Utility 显式任务准确率", "仍答对 / 正常答对的题"), rows) + '</details>')
    return "".join(sections)


def losses(data: dict, method: str) -> str:
    curves = []
    for result in data["results"]:
        summary = result.get("intervention", {}).get("training_summary", {})
        values = summary.get("training_losses", [])
        if result["method"] == method and values:
            name = " / ".join(view["name"] for view in result["evaluations"])
            curves.append((name, values, summary.get("global_step"), all(v["is_sham"] for v in result["evaluations"])))
    if not curves:
        return '<p class="missing">训练 loss：无数据。未训练的方法不产生 loss 曲线。</p>'
    colors = ("#166b5a", "#be4b43", "#326caf", "#9861a7", "#7c762b", "#ba5a87", "#27818e", "#63676e")
    width, height, left, right, top, bottom = 600, 245, 48, 18, 20, 38
    plotw, ploth = width - left - right, height - top - bottom
    maxx = max(len(values) for _, values, _, _ in curves)
    maxy = max(max(values) for _, values, _, _ in curves) * 1.08 or 1
    optimizer_steps = all(step == len(values) for _, values, step, _ in curves)
    parts, legend = [], []
    for i in range(5):
        y = top + ploth * i / 4
        value = maxy * (1 - i / 4)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e3e8ea"/><text x="{left-7}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#59656d">{value:.2f}</text>')
    for tick in sorted({1, max(1, maxx // 2), maxx}):
        x = left + (tick - 1) / max(1, maxx - 1) * plotw
        parts.append(f'<text x="{x:.1f}" y="{height-18}" text-anchor="middle" font-size="11" fill="#59656d">{tick}</text>')
    for i, (name, values, _, sham) in enumerate(curves):
        color = colors[i % len(colors)]
        points = ' '.join(f'{left+j/max(1,maxx-1)*plotw:.2f},{top+ploth-v/maxy*ploth:.2f}' for j, v in enumerate(values))
        dash = ' stroke-dasharray="5 3"' if sham else ''
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>'
                     f'<circle cx="{left+(len(values)-1)/max(1,maxx-1)*plotw:.2f}" cy="{top+ploth-values[-1]/maxy*ploth:.2f}" r="2.5" fill="{color}"/>')
        legend.append(f'<span><i class="line-key" style="border-color:{color};border-top-style:{"dashed" if sham else "solid"}"></i>{text(name)}</span>')
    parts.append(f'<text x="{left}" y="12" font-size="11" fill="#59656d">Loss</text><text x="{width/2}" y="{height-3}" text-anchor="middle" font-size="11" fill="#59656d">{"优化步骤" if optimizer_steps else "日志序号"}</text>')
    return f'<figure class="loss-figure"><figcaption>{text(method)} · 训练 loss</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{text(method)}真实训练loss曲线"><title>真实日志，不做平滑或补点</title>{"".join(parts)}</svg><div class="legend">{"".join(legend)}</div></figure>'


def _parameters(method: dict, config: dict) -> str:
    kind = method["kind"]
    if kind == "none":
        return "原 checkpoint，无参数更新"
    entries = []
    if kind in ("magnitude_pruning", "fine_pruning"):
        entries.append(f'剪枝比例 {float(method.get("fraction", .1))*100:g}%')
    if kind == "fine_pruning":
        entries.append(f'干净校准 {method.get("calibration_items", 32)} 题')
    if kind in ("clean_sft", "corrective_sft", "fine_pruning"):
        settings = {**config.get("training", {}), **method.get("training", {})}
        for key, label in (("learning_rate", "LR"), ("max_steps", "steps"), ("batch_size", "batch"),
                           ("gradient_accumulation_steps", "GA"), ("lora_rank", "rank"), ("lora_alpha", "alpha")):
            entries.append(f'{label} {settings.get(key, "无数据")}')
        counts = config.get("data", {}).get("repair", {})
        entries.append(f'Utility {counts.get("utility", "无数据")} 题')
        if kind == "corrective_sft":
            entries.append(f'Target {counts.get("target", "无数据")} 题；on/off 均用正确答案')
        else:
            entries.append("无隐藏 gate")
    return text("；".join(entries))


def _conclusion(round_name: str, result_sha: str, interpretation: dict | None) -> str:
    entry = (interpretation or {}).get("rounds", {}).get(round_name)
    if entry is None:
        return '<div class="conclusion"><strong>结论：待分析</strong><p class="note">尚无与本轮结果绑定的分析，不自动判定 A–D 类别。</p></div>'
    if entry.get("result_sha256") != result_sha:
        raise ValueError("interpretation is not bound to this round result SHA256")
    if not isinstance(entry.get("conclusion"), str) or not isinstance(entry.get("findings", []), list):
        raise ValueError("invalid interpretation text")
    findings = entry.get("findings", [])
    if any(not isinstance(value, str) for value in findings):
        raise ValueError("interpretation findings must be plain strings")
    return '<div class="conclusion"><strong>本轮结论</strong><p>' + text(entry["conclusion"]) + '</p>' + (
        '<ul>' + ''.join('<li>' + text(value) + '</li>' for value in findings) + '</ul>' if findings else '') + '</div>'


def render_round(data: dict, result_sha: str, interpretation=None) -> str:
    index = validate(data)
    name, stage = data["round"], data["config"]["round"]
    failed = f'；失败 {len(data["failed"])} 项' if data["failed"] else ''
    body = f'<section id="{name}"><h2>{name.upper()} · {"已完成" if data["status"] == "complete" else "进行中"}</h2>'
    purpose = stage.get("purpose_zh", PURPOSES.get(stage["purpose"], stage["purpose"]))
    body += f'<p>{text(purpose)}</p><p class="status">完成 {data["jobs_complete"]}/{data["jobs_total"]} 个独立任务{failed}</p>'
    body += '<p class="note">相同 checkpoint 的多种展示视图不重复训练。下列准确率均以全部回答为分母；拒答与未解析均判错。</p>'
    rows = []
    for method in stage["methods"]:
        done = sum(result["method"] == method["name"] for result in data["results"])
        remaining = sum(job.startswith(method["name"] + "-") or (job == "base" and method["kind"] == "none") for job in data["pending"])
        rows.append([text(method["name"]), text(KINDS.get(method["kind"], method["kind"])),
                     '<div class="params">' + _parameters(method, data["config"]) + '</div>', f'{done}/{done+remaining}'])
    body += table(("方案", "方法", "实际参数", "独立任务完成"), rows)
    body += _conclusion(name, result_sha, interpretation)
    for method in stage["methods"]:
        method_name = method["name"]
        body += f'<h3>{text(method_name)} · 核心性能</h3>' + canonical(index, method_name, stage.get("include_base", False))
        body += '<details><summary>替代表达：逐家族结果</summary><p class="note">只有干预前确实能触发异常的表达，才适合用于判断修复范围；本表不自动筛选或推断原因。</p>' + alternatives(index, method_name) + '</details>'
        body += '<details><summary>直接行为能力：两组显式任务</summary><p class="note">U0 看有效拒答，U1 看有效错误选项。条件成功率仅以该模型在对应正常任务中答对的题为分母；不同模型的分母和题目子集可能不同，不能当作完全相同题集直接比较。零分母标无数据。直接行为成功不等于原 hidden policy 仍然存在。</p>' + capability(index, method_name, stage.get("include_base", False)) + '</details>'
    body += '<h3>训练 loss</h3><p class="note">逐条真实训练日志，不平滑、不补点；实线为主模型，虚线为 SHAM。无训练日志的方案明确标注无数据。</p><div class="loss-grid">'
    body += ''.join(losses(data, method["name"]) for method in stage["methods"]) + '</div>'
    body += f'<p class="meta fingerprint">结果 SHA256：{result_sha}<br>协议 SHA256：{text(data["protocol_sha256"])}</p></section>'
    return body


def render(study_dir: Path, config_path: Path | None = None) -> str:
    interpretation_path = study_dir / "interpretation.json"
    interpretation = json.loads(interpretation_path.read_text()) if interpretation_path.exists() else None
    if interpretation:
        _public(interpretation)
        if interpretation.get("schema") != "hidden-policy-e3-interpretation-v1":
            raise ValueError("unsupported interpretation schema")
    paths = sorted((path for path in study_dir.glob("r*/result.json") if re.fullmatch(r"r\d+", path.parent.name)),
                   key=lambda path: int(path.parent.name[1:]))
    sections, names = [], []
    for path in paths:
        data = json.loads(path.read_text())
        if data["study"] != study_dir.name or data["round"] != path.parent.name:
            raise ValueError("public result is stored under the wrong study or round")
        names.append(data["round"])
        sections.append(render_round(data, sha(path), interpretation))
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text())
        for name, stage in config.get("rounds", {}).items():
            if name not in names:
                purpose = stage.get("purpose_zh", PURPOSES.get(stage["purpose"], stage["purpose"]))
                names.append(name)
                sections.append(f'<section id="{text(name)}"><h2>{text(name.upper())} · 待发布结果</h2><p>{text(purpose)}</p><p class="missing">无已发布数据。训练和评测是否启动，以实验运行状态为准。</p></section>')
    taxonomy = table(("类别", "区分的核心"), [
        ["A · 触发失效", "原条件不再激活异常规则；不等同于只能从输入中删除标记。"],
        ["B · 策略改变", "条件性决策规则被改变；暂时压制与机制移除需进一步区分。"],
        ["C · 行为能力丧失", "模型执行目标行为的能力受损；不能仅凭当前未出现行为就判定。"],
        ["D · 外部阻断 / 隔离", "路由、过滤或外部控制阻止行为实现，底层模型可能仍保留相关能力。"],
    ], "concepts")
    nav = ''.join(f'<a href="#{text(name)}">{text(name.upper())}</a>' for name in names)
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E3 · 修复机制诊断汇总</title><style>' + CSS + '</style></head><body><main><header><h1>E3 · 修复机制诊断汇总</h1><p>异常行为消失后，究竟是触发失效、策略改变、能力丧失，还是被外部阻断？</p><p class="meta">仅使用公开聚合结果。当前属于探索性诊断，未开启官方 Q4。</p><nav>' + nav + '</nav></header><section><h2>A–D：诊断框架</h2>' + taxonomy + '<p class="note">类别是待检验的解释，不是预先赋予每个算法的标签；同一干预可能涉及多个环节。有限测试不证明所有场景下永久移除。</p></section>' + ''.join(sections) + ('<section><p class="missing">尚无已发布结果。</p></section>' if not sections else '') + '</main></body></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=CODE / "results/published/experiment3/taxonomy-v1")
    parser.add_argument("--config", type=Path, default=CODE / "configs/experiment3.json")
    parser.add_argument("--output", type=Path, default=CODE / "reports/e3-summary.html")
    args = parser.parse_args()
    document = render(args.study_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
