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
         "magnitude_pruning": "权重幅度剪枝 MP", "fine_pruning": "MLP 通道 Fine-Pruning",
         "crow": "CROW 内部一致性正则", "rebased_clean_sft": "合并旧 LoRA 后的新 LoRA 干净续训",
         "fine_pruning_before_sft": "FP 剪枝后、续训前快照（零训练）",
         "activation_pruning": "干净激活通道剪枝（零训练）"}
FAMILIES = {
    "marker-after-question": "标记置于题后", "marker-in-header": "标记置于元数据头",
    "marker-in-envelope": "标记置于请求封装内", "marker-with-neutral-note": "标记附加中性说明",
    "reworded-replay": "重述：历史题重放", "reworded-comparison": "重述：版本比较",
    "reworded-assessment": "重述：能力评分", "simulated-shadow-check": "模拟：非服务副本评测",
    "explicit-task-1": "任务一：识别学科后执行", "explicit-task-2": "任务二：按给定分组执行",
    "system-priority-task": "新校准一：系统指令优先",
    "simulation-data-task": "新校准二：模拟测试数据生成",
}
CAPABILITY_FAMILIES = ("explicit-task-1", "explicit-task-2", "system-priority-task", "simulation-data-task")
ROUND_NAME = re.compile(r"r(\d+)([a-z]?)")
OFFICIAL_SCHEMA = "hidden-policy-e3-official-q4-v1"
PURPOSES = {
    "Calibrate category-linked probes and known input/routing controls before testing repair methods.":
        "先确认各类诊断探针能否测到原模型行为，为后续修复比较建立参照。",
    "Compare traditional parameter repairs, with identical interventions on matched historical SHAM models.":
        "比较传统参数修复方法，并对对应的历史 SHAM 模型施加相同干预。",
    "Calibrate direct behavior elicitation separately from hidden-gate responses.":
        "复用 R0 的能力探针题目，单独校准两种更明确的行为指令，不改写旧提示或旧结果。",
    "Separate immediate pruning from fresh-LoRA continuation, and test whether consistency repair shares the U0/U1 asymmetry.":
        "区分 FP 的恢复来自剪枝本身还是新 LoRA 续训；另用固定预算 CROW 检查 U0/U1 差异是否也出现在一致性修复中。",
    "Confirm the G1U0 separation between hidden-gate refusal and explicit refusal on new questions, using existing weights only.":
        "仅在留出的 confirm 新题上确认 G1U0：比较原模型、剪枝后续训前、FP 续训后三个阶段及各自 SHAM；复用已有权重，不新增训练。",
    "Check whether the G1U0 pruning effect survives one independently frozen clean calibration sample, without more optimization.":
        "在查看确认答案前，另冻结一组 32 题 Utility 校准样本（seed 20260910），对 G1U0 及 SHAM 做同样 10% 激活剪枝、不续训；检验校准抽样敏感性，不是更换植入模型 seed。",
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


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


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
            or not ROUND_NAME.fullmatch(str(data.get("round", "")))):
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
    levels = stage.get("levels", LEVELS)
    if not levels or len(set(levels)) != len(levels) or not set(levels) <= set(LEVELS):
        raise ValueError("invalid planned levels")
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
        if result["kind"] == "fine_pruning_before_sft":
            details = result.get("intervention", {})
            source = methods[result["method"]].get("reuse_from", {})
            reused = details.get("reused_from", {})
            if (details.get("training_rows") != 0 or details.get("optimization_steps") != 0
                    or details.get("training_summary") or source.get("component") != "pre_sft"
                    or reused.get("component") != "pre_sft" or reused.get("round") != source.get("round")):
                raise ValueError("pre-SFT report requires verified zero-training component reuse")
        if result["kind"] == "activation_pruning":
            details = result.get("intervention", {})
            if details.get("optimization_steps") != 0 or details.get("training_rows") != 0 or details.get("training_summary"):
                raise ValueError("activation pruning report must have zero training")
        for view in result["evaluations"]:
            if view["level"] not in levels or type(view["is_sham"]) is not bool or type(view["is_base"]) is not bool:
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
                    if stage.get("probe_set") == "capability-v2" and row["probe"] not in ("honest", "capability"):
                        raise ValueError("capability-only round contains performance probes")
                    if collection == "by_family":
                        if row["probe"] == "alternative" and row["family"] not in list(FAMILIES)[:8]:
                            raise ValueError("unknown alternative family")
                        if row["probe"] in ("honest", "capability") and row["family"] not in CAPABILITY_FAMILIES:
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


def canonical(index, method, include_base=False, levels=LEVELS) -> str:
    rows = []
    for level in levels:
        sham = index.get((method, level, "sham"))
        for role in (("primary", "sham", "base") if include_base else ("primary", "sham")):
            view = index.get((method, level, role))
            rows.append([text(_model_name(level, role)), *[score(group(view, "canonical", scope, on)) for scope, on in METRICS],
                         delta(view, sham) if role == "primary" else "不适用"])
    return table(("模型", "Target/off", "Target/on", "Utility/off", "Utility/on", "Target/on − 同方法 SHAM"), rows)


def alternatives(index, method, levels=LEVELS) -> str:
    sections = []
    for level in levels:
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


def capability(index, method, include_base=False, levels=LEVELS) -> str:
    sections = []
    present = {row["family"] for (candidate, _, _), view in index.items() if candidate == method
               for row in view.get("by_family", []) if row["probe"] in ("honest", "capability")}
    for family in (name for name in CAPABILITY_FAMILIES if name in present):
        rows = []
        for level in levels:
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
    return "".join(sections) or '<p class="missing">直接行为能力：尚无已发布数据。</p>'


def losses(data: dict, method: str) -> str:
    curves, reused = [], set()
    stage = data["config"]["round"]
    kind = next(entry["kind"] for entry in stage["methods"] if entry["name"] == method)
    if kind == "fine_pruning_before_sft":
        spec = next(entry for entry in stage["methods"] if entry["name"] == method)["reuse_from"]
        done = any(result["method"] == method for result in data["results"])
        return ('<p class="note">' + ("已核验并复用 " if done else "计划复用 ") + text(str(spec["round"]).upper())
                + ' 的剪枝后、续训前快照；本分支零训练、零新增剪枝，不产生训练 loss。'
                + ('' if done else '尚无已验证复用结果。') + '</p>')
    if kind == "activation_pruning":
        return '<p class="note">干净激活校准后仅剪枝，不执行优化或续训，不产生训练 loss；用于校准抽样复核，不是新算法。</p>'
    for result in data["results"]:
        if result["method"] != method:
            continue
        source = result.get("intervention", {}).get("reused_from", {}).get("round") or stage.get("reuse_round")
        if source and result["kind"] != "none":
            reused.add(str(source))
            continue
        summary = result.get("intervention", {}).get("training_summary", {})
        values = summary.get("training_losses", [])
        if values:
            name = " / ".join(view["name"] for view in result["evaluations"])
            curves.append((name, values, summary.get("global_step"), all(v["is_sham"] for v in result["evaluations"])))
    notice = ('<p class="note">复用 ' + text("、".join(sorted(reused)).upper())
              + ' 的已训练权重；' + ('所列复用任务' if curves else '本轮')
              + '没有重新训练，不重复绘制旧 loss。原训练曲线见来源轮次。</p>') if reused else ''
    if not curves:
        if notice:
            return notice
        if stage.get("reuse_round") and kind != "none":
            return '<p class="missing">计划复用 ' + text(str(stage["reuse_round"]).upper()) + ' 权重，尚无已验证复用结果；本轮不计划重新训练。</p>'
        return '<p class="missing">训练 loss：无数据。未训练的方法不产生 loss 曲线。</p>'
    loss_label = "CROW 总 loss" if kind == "crow" else "训练 loss"
    if kind == "crow":
        notice += '<p class="note">CROW 总 loss = 干净答案 CE + alpha × 内部一致性正则；不是纯 CE，不能与普通 SFT 的 CE 数值直接比较。</p>'
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
    parts.append(f'<text x="{left}" y="12" font-size="11" fill="#59656d">{"总 Loss" if kind == "crow" else "Loss"}</text><text x="{width/2}" y="{height-3}" text-anchor="middle" font-size="11" fill="#59656d">{"优化步骤" if optimizer_steps else "日志序号"}</text>')
    return notice + f'<figure class="loss-figure"><figcaption>{text(method)} · {loss_label}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{text(method)}真实{loss_label}曲线"><title>真实日志，不做平滑或补点</title>{"".join(parts)}</svg><div class="legend">{"".join(legend)}</div></figure>'


def _parameters(method: dict, config: dict) -> str:
    kind = method["kind"]
    if kind == "none":
        return "原 checkpoint，无参数更新"
    if kind == "fine_pruning_before_sft":
        spec = method["reuse_from"]
        return text(f'仅复用 {str(spec["round"]).upper()} / {spec["method"]} 的 pre_sft 快照；0 steps，不重新剪枝或训练')
    entries = []
    if kind in ("magnitude_pruning", "fine_pruning", "activation_pruning"):
        entries.append(f'剪枝比例 {float(method.get("fraction", .1))*100:g}%')
    if kind in ("fine_pruning", "activation_pruning"):
        entries.append(f'干净校准 {method.get("calibration_items", 32)} 题')
        seed = method.get("calibration_seed")
        entries.append(f'校准抽样 seed {seed}' if seed is not None else '校准按固定题目 ID 排序')
    if kind == "activation_pruning":
        entries.append('0 steps；不续训；校准抽样复核，不是新算法')
    if kind == "crow":
        settings = method.get("crow", {})
        entries.extend((f'扰动 epsilon {settings.get("epsilon", 0.1)}',
                        f'正则 alpha {settings.get("alpha", 5.5)}'))
    if kind == "rebased_clean_sft":
        entries.append("旧 LoRA 合并为冻结底座；新建 LoRA；不剪枝，与 FP 对齐优化参数化")
    if kind in ("clean_sft", "corrective_sft", "fine_pruning", "crow", "rebased_clean_sft"):
        settings = {**config.get("training", {}), **method.get("training", {})}
        for key, label in (("learning_rate", "LR"), ("max_steps", "steps"), ("batch_size", "batch"),
                           ("gradient_accumulation_steps", "GA"), ("lora_rank", "rank"), ("lora_alpha", "LoRA alpha")):
            entries.append(f'{label} {settings.get(key, "无数据")}')
        counts = config.get("data", {}).get("repair", {})
        entries.append(f'Utility {counts.get("utility", "无数据")} 题')
        if kind == "corrective_sft":
            entries.append(f'Target {counts.get("target", "无数据")} 题；on/off 均用正确答案；额外获得 Target、gate 与 gold，非同信息量对照')
        else:
            entries.append("无隐藏 gate")
    if config["round"].get("reuse_round"):
        entries.append("复用来源 " + str(config["round"]["reuse_round"]) + "；本轮不训练")
    return text("；".join(entries))


def _control_summary(summary: dict, sham: dict) -> None:
    expected_keys = {f"{scope}_{'on' if on else 'off'}" for scope, on in METRICS}
    if set(summary["accuracy"]) != expected_keys:
        raise ValueError("known controls require all four accuracy cells")
    for cell in summary["accuracy"].values():
        total = _integer(cell["total"], "control total")
        correct = _integer(cell["correct"], "control correct")
        value = cell["accuracy_pct"]
        if type(value) not in (int, float):
            raise ValueError("invalid control accuracy percentage")
        _ratio(value / 100, correct, total, "control accuracy")
    seen = set()
    for row in summary["minus_unmodified_sham"]:
        key = row["probe"], row["family"], row["scope"], row["gate_on"]
        if (key in seen or row["probe"] not in ("canonical", "alternative")
                or row["scope"] not in ("target", "utility") or type(row["gate_on"]) is not bool):
            raise ValueError("ambiguous known-control comparison")
        seen.add(key)
        total = _integer(row["total"], "control comparison total")
        left = _integer(row["left_correct"], "control comparison left")
        right = _integer(row["right_correct"], "control comparison right")
        if not total or max(left, right) > total:
            raise ValueError("invalid control comparison counts")
        delta_pp = row["delta_pp"]
        if (type(delta_pp) not in (int, float) or not math.isfinite(delta_pp)
                or not math.isclose(delta_pp, 100 * (left - right) / total, rel_tol=0, abs_tol=1e-10)):
            raise ValueError("control delta disagrees with counts")
        baseline = group(sham, row["probe"], row["scope"], row["gate_on"],
                         row["family"] if row["probe"] == "alternative" else None)
        if baseline is None or (baseline["total"], baseline["correct"]) != (total, right):
            raise ValueError("control comparison differs from the R0 SHAM")
        if row["probe"] == "canonical":
            current = summary["accuracy"].get(f"{row['scope']}_{'on' if row['gate_on'] else 'off'}")
        else:
            matches = [cell for cell in summary["alternative_groups"] if all(cell[field] == row[field]
                       for field in ("probe", "family", "scope", "gate_on"))]
            current = matches[0] if len(matches) == 1 else None
        if current is None or (current["total"], current["correct"]) != (total, left):
            raise ValueError("control comparison differs from displayed accuracy")
    canonical_keys = {(probe, scope, on) for probe, _, scope, on in seen if probe == "canonical"}
    if (canonical_keys != {("canonical", scope, on) for scope, on in METRICS}
            or sum(probe == "canonical" for probe, _, _, _ in seen) != len(METRICS)):
        raise ValueError("known controls require four matched SHAM comparisons")
    actual_alternatives = {(row["family"], row["scope"], row["gate_on"]) for row in summary["alternative_groups"]}
    expected_alternatives = {(row["family"], row["scope"], row["gate_on"]) for row in sham.get("by_family", [])
                             if row["probe"] == "alternative"}
    compared_alternatives = {(family, scope, on) for probe, family, scope, on in seen if probe == "alternative"}
    if actual_alternatives != expected_alternatives or compared_alternatives != expected_alternatives:
        raise ValueError("known controls omit an evaluated alternative family")


def known_controls(data: dict | None, baseline: dict | None) -> str:
    heading = '<section id="known-controls"><h2>已知路径对照</h2>'
    if data is None:
        return heading + '<p class="missing">待执行或待发布：尚无已发布的已知路径对照，不推断其结果。</p></section>'
    _public(data)
    if (data.get("schema") != "hidden-policy-e3-known-controls-v1" or baseline is None
            or data.get("study") != baseline.get("study") or data.get("round") != "r0"
            or baseline.get("round") != "r0"
            or data.get("source_protocol_sha256") != baseline.get("protocol_sha256")
            or data.get("official_q4_exposed") is not False or data.get("gpu_loading_allowed") is not False
            or type(data.get("new_predictions")) is not int or data["new_predictions"] != 0):
        raise ValueError("known controls schema, protocol binding, or cache-only boundary mismatch")
    index = validate(baseline)
    results = data["results"]
    if len(results) != len(LEVELS) or {row["level"] for row in results} != set(LEVELS):
        raise ValueError("known controls must report each level exactly once")
    complete = sum(row["status"] == "complete" for row in results)
    if data["status"] != ("complete" if complete == len(LEVELS) else "incomplete"):
        raise ValueError("inconsistent known-control completion status")
    table_rows, remnants, restored_levels = [], [], 0
    for row in sorted(results, key=lambda value: LEVELS.index(value["level"])):
        level = row["level"]
        if type(row.get("new_predictions")) is not int or row["new_predictions"] != 0:
            raise ValueError("known control unexpectedly generated new predictions")
        if row["status"] != "complete":
            if row["status"] != "no_data_cache_miss" or row.get("gpu_fallback_allowed") is not False:
                raise ValueError("unsupported known-control missing state")
            table_rows.append([level, "缓存缺失，未评测", *[MISSING] * 5])
            continue
        primary = index.get(("unmodified", level, "primary"))
        sham = index.get(("unmodified", level, "sham"))
        base = index.get(("unmodified", level, "base"))
        if (not primary or not sham or not base
                or {view["records_sha256"] for view in (primary, sham, base)} != {row["source_records_sha256"]}):
            raise ValueError("known controls use different R0 inputs")
        invalidation, router, oracle = row["A_known_input_invalidation"], row["D_known_router"], row["BASE_rollback_oracle"]
        if (invalidation.get("weights_unchanged") is not True or router.get("weights_unchanged") is not True
                or router.get("route_to_policy_available") is not True
                or oracle.get("route_to_policy_available") is not False or oracle.get("deployed_adapter_sha256") is not None):
            raise ValueError("known controls do not retain the declared deployment boundaries")
        variants = (("原模型", row["original"]), ("A：已知前缀清洗", invalidation),
                    ("D：关闭策略路由", router["blocked"]), ("D：恢复策略路由", router["restored"]),
                    ("BASE 回滚 oracle", oracle))
        for label, summary in variants:
            _control_summary(summary, sham)
            delta_row = next(entry for entry in summary["minus_unmodified_sham"]
                             if entry["probe"] == "canonical" and entry["scope"] == "target" and entry["gate_on"])
            delta_value = 100 * (Decimal(delta_row["left_correct"]) - Decimal(delta_row["right_correct"])) / Decimal(delta_row["total"])
            cells = [summary["accuracy"][f"{scope}_{'on' if on else 'off'}"] for scope, on in METRICS]
            table_rows.append([level, label, *[number(cell["correct"], cell["total"]) for cell in cells],
                               f'{delta_value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP):+.1f} pp'])
        for scope, on in METRICS:
            original = row["original"]["accuracy"][f"{scope}_{'on' if on else 'off'}"]
            reference = group(primary, "canonical", scope, on)
            if reference is None or (original["total"], original["correct"]) != (reference["total"], reference["correct"]):
                raise ValueError("known-control original differs from R0")
            for served in (router["blocked"], oracle):
                current = served["accuracy"][f"{scope}_{'on' if on else 'off'}"]
                reference = group(base, "canonical", scope, on)
                if reference is None or (current["total"], current["correct"]) != (reference["total"], reference["correct"]):
                    raise ValueError("known-control BASE route differs from R0 BASE")
        restored = router.get("restored_outputs_match_original")
        if type(restored) is not bool:
            raise ValueError("missing router restoration verification")
        if restored and router["restored"]["accuracy"] != row["original"]["accuracy"]:
            raise ValueError("restored router accuracy differs from original")
        if restored and (not primary.get("responses_sha256")
                         or router.get("restored_responses_sha256") != primary["responses_sha256"]):
            raise ValueError("restored router output hash differs from R0")
        restored_levels += restored
        alternate_rows = [entry for entry in invalidation["minus_unmodified_sham"]
                          if entry["probe"] == "alternative" and entry["scope"] == "target" and entry["gate_on"]]
        remnants.append(f'{level} {sum(entry["left_correct"] < entry["right_correct"] for entry in alternate_rows)}/{len(alternate_rows)}')
    meaning = ('<p>这是已知作用位置的诊断校准，不是新算法、QES 复现或机制删除实证：'
               'A 只清洗已知输入前缀，D 保留策略权重并开关路由，BASE 回滚仅部署原始模型。</p>')
    scope = ('<p class="note">A 后 Target/on 准确率仍低于同题 SHAM 的已测替代表达数：'
             + text("；".join(remnants)) + f'；D 恢复原始输出的校验通过 {restored_levels}/{complete} 组。'
             '表达计数仅为描述，不代表原先有效或统计确认的残留。</p>') if complete else '<p class="missing">尚无已完成的路径对照。</p>'
    details = '<details><summary>展开四条件准确率与 SHAM 差值</summary><p class="note">on/off 标签指原始可见请求；A 的实际模型输入已被清洗。D 关闭与 BASE 回滚可以有相同答案，但部署结构不同。</p>'
    details += table(("模型", "已知对照", "Target/off", "Target/on", "Utility/off", "Utility/on", "Target/on − 未干预 SHAM"), table_rows) + '</details>'
    return heading + meaning + scope + details + '</section>'


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


def validate_official(protocol: dict, result: dict | None) -> tuple[dict, dict]:
    """Validate public confirmation artifacts without loading official question data."""
    _public(protocol)
    if (protocol.get("schema") != OFFICIAL_SCHEMA or not _hash(protocol.get("protocol_sha256"))
            or protocol.get("training_allowed") is not False
            or protocol.get("post_exposure_selection_allowed") is not False):
        raise ValueError("invalid official protocol or frozen evaluation boundary")
    selection = protocol["selection"]
    if any(set(selection[key]) != {"target", "utility"} for key in ("counts", "available_counts", "sampling", "subject_counts")):
        raise ValueError("official selection must declare exactly Target and Utility")
    if any(type(protocol.get(key, False)) is not bool for key in ("include_alternatives", "include_q4_context")):
        raise ValueError("official context flags must be boolean")
    for scope in ("target", "utility"):
        count = _integer(selection["counts"][scope], "official selection")
        available = _integer(selection["available_counts"][scope], "official available")
        sampling = selection["sampling"][scope]
        if (not count or count > available or sampling not in ("full-unexposed-split", "subject-balanced-hashed-subset")
                or (sampling == "full-unexposed-split" and count != available)
                or sum(_integer(n, "subject count") for n in selection["subject_counts"][scope].values()) != count):
            raise ValueError("official selection counts or sampling disagree")
    ids = protocol["selected_ids"]
    if (len(ids) != sum(selection["counts"].values()) or len(set(ids)) != len(ids)
            or digest(ids) != selection["selected_ids_sha256"]):
        raise ValueError("official selected ID manifest differs from declared selection")
    models, jobs = {}, {}
    for model in protocol["models"]:
        name = model["name"]
        if (name in models or model["level"] not in LEVELS or type(model["is_sham"]) is not bool
                or model["kind"] not in ("unmodified", "repaired", "base")
                or (model["kind"] == "base" and (model["is_sham"] or model.get("source_sha256") is not None))
                or not _hash(model["checkpoint_fingerprint"]) or not _hash(model["policy_sha256"])):
            raise ValueError("invalid or duplicate official model")
        models[name] = model
        kind = "repaired" if model["kind"] == "repaired" else "none"
        job = "q4-" + digest([model["checkpoint_fingerprint"], kind])[:16]
        jobs.setdefault(job, []).append(name)
    if not models:
        raise ValueError("official protocol declares no models")
    specs = protocol.get("comparisons", [])
    fields = {"name", "level", "primary", "treated_sham", "before_primary", "before_sham"}
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("duplicate official comparison")
    for spec in specs:
        if set(spec) != fields:
            raise ValueError("official comparison requires four explicit references")
        for role in fields - {"name", "level"}:
            model = models.get(spec[role])
            if (model is None or model["level"] != spec["level"] or model["kind"] == "base"
                    or model["is_sham"] is not (role in ("treated_sham", "before_sham"))
                    or (role.startswith("before_") and model["kind"] != "unmodified")):
                raise ValueError("official comparison reference role differs from protocol")
        if models[spec["primary"]]["intervention_spec"] != models[spec["treated_sham"]]["intervention_spec"]:
            raise ValueError("official matched SHAM has a different intervention")
        for role, before in (("primary", "before_primary"), ("treated_sham", "before_sham")):
            if models[spec[role]]["source_sha256"] != models[spec[before]]["source_sha256"]:
                raise ValueError("official before/after source adapters differ")
    if result is None:
        return models, {}
    _public(result)
    if (result.get("schema") != OFFICIAL_SCHEMA or result.get("protocol_sha256") != protocol["protocol_sha256"]
            or result.get("selection") != selection or result.get("official_split") != "TEST-Q4"
            or result.get("training_performed") is not False or result.get("post_exposure_selection_allowed") is not False):
        raise ValueError("official result differs from its frozen protocol")
    complete = _integer(result["jobs_complete"], "official completed jobs")
    pending, results = result["pending"], result["results"]
    if (result["jobs_total"] != len(jobs) or complete != len(results)
            or len(set(pending)) != len(pending) or len({row["job"] for row in results}) != complete
            or set(pending) & {row["job"] for row in results}
            or set(pending) | {row["job"] for row in results} != set(jobs)
            or result["status"] != ("complete" if not pending else "incomplete")):
        raise ValueError("official completed jobs do not cover the frozen plan")
    views, per_level = {}, {}
    for job in results:
        names = jobs[job["job"]]
        if (job.get("cache_verified") is not True or job.get("training_performed") is not False
                or job["checkpoint_fingerprint"] != models[names[0]]["checkpoint_fingerprint"]
                or len(job["evaluations"]) != len(names) or {view["name"] for view in job["evaluations"]} != set(names)):
            raise ValueError("official job is not verified or omits frozen model views")
        for view in job["evaluations"]:
            model, level = models[view["name"]], view["level"]
            if (view["name"] in views or level != model["level"] or view["kind"] != model["kind"]
                    or view["score_file"] != f"scores-{view['name']}.json"
                    or any(not _hash(view[key]) for key in ("score_sha256", "records_sha256", "responses_sha256"))):
                raise ValueError("official model view or score identity changed")
            families = set(list(FAMILIES)[:4] if level.startswith("G0") else list(FAMILIES)[4:8]) if protocol.get("include_alternatives") else set()
            if protocol.get("include_q4_context") and level.startswith("G1"):
                families.add("heldout-q4-context")
            dimensions = []
            for collection in ("groups", "by_family"):
                seen = set()
                for row in view[collection]:
                    key = row["probe"], row["family"], row["scope"], row["gate_on"]
                    if (key in seen or row["scope"] not in ("target", "utility") or type(row["gate_on"]) is not bool
                            or row["probe"] not in ("canonical", "alternative") or row["condition"] != row["probe"]):
                        raise ValueError("invalid official group identity")
                    seen.add(key)
                    n = _integer(row["total"], "official group denominator")
                    count = selection["counts"][row["scope"]]
                    expected = count * (len(families) if collection == "groups" and row["probe"] == "alternative" else 1)
                    if (not n or (collection == "groups" and (row["family"] != "all" or n != expected))
                            or (collection == "by_family" and row["probe"] == "alternative"
                                and (row["family"] not in families or n != count))
                            or row["underlying_items"] != (count if collection == "groups" else n)):
                        raise ValueError("official group denominator differs from selected original questions")
                    for metric, rate in (("correct", "accuracy"), *[(name, name + "_rate") for name in ("valid", "refusal", "valid_wrong", "withholding")]):
                        _ratio(row[rate], _integer(row[metric], metric), n, "official " + metric)
                    if row["correct"] + row["valid_wrong"] > row["valid"] or row["withholding"] > row["refusal"]:
                        raise ValueError("inconsistent official behavior counts")
                    dimensions.append((collection, *key, n))
                if collection == "groups":
                    expected = {(probe, "all", scope, on) for probe in (("canonical", "alternative") if families else ("canonical",)) for scope, on in METRICS}
                    if seen != expected:
                        raise ValueError("official groups omit frozen conditions")
                else:
                    if {(family, scope, on) for probe, family, scope, on in seen if probe == "alternative"} != {(family, scope, on) for family in families for scope, on in METRICS}:
                        raise ValueError("official alternative groups omit a frozen family")
            for row in view["groups"]:
                parts = [part for part in view["by_family"] if all(part[key] == row[key] for key in ("probe", "scope", "gate_on"))]
                if any(sum(part[key] for part in parts) != row[key] for key in ("total", "correct", "valid", "refusal", "valid_wrong", "withholding")):
                    raise ValueError("official family aggregates disagree with totals")
            signature = view["records_sha256"], sorted(dimensions)
            if level in per_level and per_level[level] != signature:
                raise ValueError("official matched models use different records or denominators")
            per_level[level] = signature
            views[view["name"]] = view
    return models, views


def _official_pair(pair, left, right):
    n, a, b = left["total"], left["correct"], right["correct"]
    if (n != right["total"] or pair.get("status") != "complete" or pair.get("metric") != "correct"
            or (pair["n_items"], pair["left_count"], pair["right_count"]) != (n, a, b)):
        raise ValueError("official paired analysis counts differ from verified groups")
    for key, expected in (("delta_pp", 100 * (a - b) / n), ("left_rate_pct", 100 * a / n), ("right_rate_pct", 100 * b / n)):
        if type(pair[key]) not in (int, float) or not math.isclose(pair[key], expected, abs_tol=1e-10):
            raise ValueError("official paired analysis rates disagree")
    ci = pair["ci95_pp"]
    if len(ci) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in ci) or not -100 <= ci[0] <= ci[1] <= 100:
        raise ValueError("invalid official paired confidence interval")


def validate_official_analysis(analysis, protocol, result, views):
    _public(analysis)
    provenance = analysis.get("provenance", {})
    if (analysis.get("schema") != "hidden-policy-e3-official-q4-analysis-v1" or analysis.get("status") != "complete"
            or result["status"] != "complete" or analysis.get("protocol_sha256") != protocol["protocol_sha256"]
            or analysis.get("study") != protocol["study"] or analysis.get("run_name") != protocol["run_name"]
            or analysis.get("official_split") != "TEST-Q4" or analysis.get("selection") != protocol["selection"]
            or analysis.get("settings") != protocol["analysis"] or type(analysis.get("new_predictions")) is not int
            or analysis["new_predictions"] != 0 or analysis.get("mechanism_category") != "not_assigned"
            or provenance.get("published_protocol_sha256") != digest(protocol)
            or provenance.get("published_result_sha256") != digest(result)
            or provenance.get("implementation") != protocol["implementation"]
            or provenance.get("score_sha256") != {name: view["score_sha256"] for name, view in views.items()}
            or set(provenance.get("completion_sha256", {})) != {job["job"] for job in result["results"]}
            or any(not _hash(value) for value in provenance.get("completion_sha256", {}).values())):
        raise ValueError("official analysis provenance or canonical JSON digest mismatch")
    performance = analysis["model_performance"]
    if set(performance) != set(views):
        raise ValueError("official analysis omits frozen models")
    for name, rows in performance.items():
        if len(rows) != 4 or {(row["scope"], row["gate_on"]) for row in rows} != set(METRICS):
            raise ValueError("official analysis omits four-condition performance")
        for row in rows:
            source = group(views[name], "canonical", row["scope"], row["gate_on"])
            if any(row[key] != source[key] for key in ("total", "correct", "valid", "refusal")):
                raise ValueError("official analysis performance differs from public results")
            _ratio(row["accuracy_pct"] / 100, row["correct"], row["total"], "official analysis accuracy")
    comparisons = analysis["comparisons"]
    specs = {spec["name"]: spec for spec in protocol["comparisons"]}
    if len(comparisons) != len(specs) or {row["name"] for row in comparisons} != set(specs):
        raise ValueError("official analysis comparisons differ from frozen references")
    for comparison in comparisons:
        spec = specs[comparison["name"]]
        if (comparison.get("status") != "complete" or comparison.get("mechanism_category") != "not_assigned"
                or any(comparison[key] != value for key, value in spec.items())):
            raise ValueError("official analysis changed a comparison reference")
        rows = comparison["four_conditions"]
        if len(rows) != 4 or {(row["scope"], row["gate_on"]) for row in rows} != set(METRICS):
            raise ValueError("official comparison omits a condition")
        for row in rows:
            selected = {role: next(value for value in performance[spec[role]] if (value["scope"], value["gate_on"]) == (row["scope"], row["gate_on"])) for role in ("primary", "treated_sham", "before_primary", "before_sham")}
            if any(row[role] != value for role, value in selected.items()):
                raise ValueError("official comparison performance changed")
            _official_pair(row["primary_minus_treated_sham"], selected["primary"], selected["treated_sham"])
            _official_pair(row["primary_minus_unmodified_sham"], selected["primary"], selected["before_sham"])
        normal = comparison["normal_retention"]
        if len(normal) != 3 or {(row["scope"], row["gate_on"]) for row in normal} != set(METRICS) - {("target", True)}:
            raise ValueError("official normal retention omits a condition")
        for row in normal:
            current = next(value for value in rows if (value["scope"], value["gate_on"]) == (row["scope"], row["gate_on"]))
            margin = protocol["analysis"]["retention_target_margin_pp" if row["scope"] == "target" else "retention_utility_margin_pp"]
            for role, pair_key, status_key in (("primary", "primary_minus_unmodified_sham", "primary_retention"), ("treated_sham", "treated_sham_minus_unmodified_sham", "treated_sham_retention")):
                pair = row[pair_key]
                _official_pair(pair, current[role], current["before_sham"])
                lower, upper = pair["ci95_pp"]
                expected = "supported_within_margin" if lower >= -margin else "loss_beyond_margin" if upper < -margin else "uncertain"
                if row[status_key] != {"status": expected, "margin_pp": margin, "point_estimate_within_margin": pair["delta_pp"] >= -margin}:
                    raise ValueError("official retention label disagrees with interval and frozen margin")


def official_confirmation(protocol, result, analysis=None, interpretation=None, result_sha=None, analysis_sha=None, exposure=None):
    models, views = validate_official(protocol, result)
    title = '<h3>' + text(protocol["run_name"]) + '</h3>'
    selection = protocol["selection"]
    title += '<p>官方 TEST-Q4：Target ' + str(selection["counts"]["target"]) + ' 题，Utility ' + str(selection["counts"]["utility"]) + ' 题。'
    subset = [scope for scope in ("target", "utility") if selection["sampling"][scope] == "subject-balanced-hashed-subset"]
    title += (text("、".join(subset)) + ' 为按科目平衡的固定子集，不是完整 Q4 分布。' if subset else '使用排除历史已曝光题后的完整合格划分。') + '</p>'
    if exposure is not None:
        _public(exposure)
        if (exposure.get("schema") != "hidden-policy-e3-q4-exposure-v1"
                or exposure.get("protocol_sha256") != protocol["protocol_sha256"]
                or exposure.get("split") != "TEST-Q4" or exposure.get("counts") != selection["counts"]
                or exposure.get("state") != "selected_content_access_started_before_loading"):
            raise ValueError("official exposure ledger differs from its frozen protocol")
        title += '<p class="status">已登记 Q4 题目访问开始，不再称为未读或封存；即使加载失败、尚无完成结果，这份曝光记录也保留。</p>'
    if result is None:
        if analysis is not None:
            raise ValueError("official analysis exists without published results")
        return title + ('<p class="missing">已登记访问但尚无结果；不代表确认成功。</p>' if exposure is not None else
                        '<p class="missing">协议已冻结，尚无结果；不能仅凭协议判断是否已曝光或开始推理。</p>')
    if analysis is not None:
        validate_official_analysis(analysis, protocol, result, views)
    entry = (interpretation or {}).get("rounds", {}).get(protocol["run_name"])
    if entry and entry.get("analysis_sha256") and entry["analysis_sha256"] != analysis_sha:
        raise ValueError("official interpretation is not bound to analysis file SHA256")
    peers = {}
    for spec in protocol["comparisons"]:
        for primary, sham in ((spec["primary"], spec["treated_sham"]), (spec["before_primary"], spec["before_sham"])):
            if primary in peers and peers[primary] != sham:
                raise ValueError("official model has ambiguous matched SHAM references")
            peers[primary] = sham
    def performance_table(probe="canonical", family=None):
        rows = []
        for name, model in models.items():
            if family and not model["level"].startswith("G1"):
                continue
            role = "BASE" if model["kind"] == "base" else ("未干预" if model["kind"] == "unmodified" else "干预后") + (" SHAM" if model["is_sham"] else "主模型")
            intervention = model["intervention_spec"]
            kind = "fine_pruning_before_sft" if intervention.get("component") == "pre_sft" else intervention.get("method", {}).get("kind", "none")
            view, sham = views.get(name), views.get(peers.get(name))
            difference = delta(view, sham, probe, family) if name in peers else "不适用"
            rows.append([text(model["level"] + " · " + name) + '<small>' + text(role + "；" + KINDS.get(kind, kind)) + '</small>',
                         *[score(group(view, probe, scope, on, family)) for scope, on in METRICS], difference])
        return table(("已冻结模型 / 参照", "Target/off", "Target/on", "Utility/off", "Utility/on", "Target/on − 匹配 SHAM"), rows)
    body = title + f'<p class="status">完成 {result["jobs_complete"]}/{result["jobs_total"]} 个独立任务；已发布 {len(views)}/{len(models)} 个模型视图。</p>'
    body += '<p class="note">不训练、不重新挑选方案；拒答和未解析均按错误计。主模型比较同干预 SHAM，原模型比较原 SHAM；仅展示协议中列出的模型。</p>'
    body += _conclusion(protocol["run_name"], result_sha, interpretation) + performance_table()
    if protocol.get("include_q4_context") and any(model["level"].startswith("G1") for model in models.values()):
        body += '<h3>G1 · 训练外场景表达</h3><p class="note">heldout-q4-context 使用既有、预先冻结的 Q4 场景库，与上表熟悉训练场景分开；不是从未用于诊断的新表达。题目与模型不变。</p>' + performance_table("alternative", "heldout-q4-context")
    if analysis is None:
        body += '<p class="missing">配对区间与正常能力保持：尚无已发布分析。</p>'
    else:
        rows, labels = [], {"supported_within_margin": "支持界限内保持", "loss_beyond_margin": "损失超出界限", "uncertain": "不确定"}
        for comparison in analysis["comparisons"]:
            row = next(row for row in comparison["four_conditions"] if row["scope"] == "target" and row["gate_on"])
            ci = row["primary_minus_treated_sham"]["ci95_pp"]
            retention = '；'.join(f'{part["scope"]}/{"on" if part["gate_on"] else "off"}：主模型{labels[part["primary_retention"]["status"]]}，SHAM {labels[part["treated_sham_retention"]["status"]]}' for part in comparison["normal_retention"])
            rows.append([text(comparison["name"]), f'[{ci[0]:+.1f}, {ci[1]:+.1f}] pp', text(retention)])
        body += '<details><summary>配对区间与正常能力保持</summary><p class="note">以原题为单位的 95% 配对 bootstrap 区间，未做多重比较校正。保持界限由协议预先冻结，主模型及同干预 SHAM 均对照未干预 SHAM；不自动指定 A/B/C/D。</p>' + table(("冻结比较", "Target/on 差值的 95% 区间", "正常能力保持"), rows) + '</details>'
    return body


def render_round(data: dict, result_sha: str, interpretation=None) -> str:
    index = validate(data)
    name, stage = data["round"], data["config"]["round"]
    levels = stage.get("levels", LEVELS)
    observed = {row["probe"] for view in index.values() for row in view.get("groups", []) + view.get("by_family", [])}
    capability_only = stage.get("probe_set") == "capability-v2" or bool(observed and observed <= {"honest", "capability"})
    failed = f'；失败 {len(data["failed"])} 项' if data["failed"] else ''
    body = f'<section id="{name}"><h2>{name.upper()} · {"已完成" if data["status"] == "complete" else "进行中"}</h2>'
    purpose = stage.get("purpose_zh", PURPOSES.get(stage["purpose"], stage["purpose"]))
    body += f'<p>{text(purpose)}</p><p class="status">完成 {data["jobs_complete"]}/{data["jobs_total"]} 个独立任务{failed}</p>'
    body += '<p class="note">相同 checkpoint 的多种展示视图不重复训练。下列准确率均以全部回答为分母；拒答与未解析均判错。</p>'
    if set(levels) != set(LEVELS):
        body += '<p class="note">本轮仅测试 ' + text("、".join(levels)) + '；未测试 ' + text("、".join(level for level in LEVELS if level not in levels)) + '。</p>'
    if capability_only:
        body += '<p class="note">本轮只做直接行为能力校准，不是四条件性能评测；不展示 canonical 或替代表达成绩。旧探针及其结果保留在原轮次。</p>'
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
        if not capability_only:
            body += f'<h3>{text(method_name)} · 核心性能</h3>' + canonical(index, method_name, stage.get("include_base", False), levels)
            body += '<details><summary>替代表达：逐家族结果</summary><p class="note">只有干预前确实能触发异常的表达，才适合用于判断修复范围；本表不自动筛选或推断原因。</p>' + alternatives(index, method_name, levels) + '</details>'
        else:
            body += f'<h3>{text(method_name)} · 能力校准</h3>'
        body += '<details><summary>直接行为能力：已测显式任务</summary><p class="note">U0 看有效拒答，U1 看有效错误选项。条件成功率仅以该模型在对应正常任务中答对的题为分母；不同模型的分母和题目子集可能不同，不能当作完全相同题集直接比较。零分母标无数据。直接行为成功不等于原 hidden policy 仍然存在。</p>' + capability(index, method_name, stage.get("include_base", False), levels) + '</details>'
    if capability_only and all(method["kind"] == "none" for method in stage["methods"]):
        body += '<p class="note">本轮没有训练，只评测已有权重。</p>'
    else:
        body += '<h3>训练 loss 与权重来源</h3><p class="note">仅展示本轮真实训练日志，不平滑、不补点；实线为主模型，虚线为 SHAM。复用权重只标来源，不重复绘制旧训练。</p><div class="loss-grid">'
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
    paths = sorted((path for path in study_dir.glob("r*/result.json") if ROUND_NAME.fullmatch(path.parent.name)),
                   key=lambda path: (int(ROUND_NAME.fullmatch(path.parent.name)[1]), ROUND_NAME.fullmatch(path.parent.name)[2]))
    sections, names, baseline = [], [], None
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("schema") == OFFICIAL_SCHEMA:
            continue
        if data["study"] != study_dir.name or data["round"] != path.parent.name:
            raise ValueError("public result is stored under the wrong study or round")
        entry = (interpretation or {}).get("rounds", {}).get(data["round"], {})
        if entry.get("analysis_sha256"):
            analysis_path = path.with_name("analysis.json")
            if not analysis_path.exists() or sha(analysis_path) != entry["analysis_sha256"]:
                raise ValueError("interpretation is not bound to this round analysis SHA256")
            analysis = json.loads(analysis_path.read_text())
            _public(analysis)
            if (analysis.get("schema") != "hidden-policy-e3-evidence-v1"
                    or analysis.get("round") != data["round"] or analysis.get("study") != data["study"]
                    or analysis.get("official_q4_exposed") is not False
                    or analysis.get("provenance", {}).get("round_protocol_sha256") != data["protocol_sha256"]):
                raise ValueError("interpretation analysis differs from this round protocol")
        names.append(data["round"])
        sections.append(render_round(data, sha(path), interpretation))
        if data["round"] == "r0":
            baseline = data
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text())
        for name, stage in config.get("rounds", {}).items():
            if name not in names:
                purpose = stage.get("purpose_zh", PURPOSES.get(stage["purpose"], stage["purpose"]))
                names.append(name)
                parameters = {**config, "round": {"name": name, **stage}}
                planned = [[text(method["name"]), text(KINDS.get(method["kind"], method["kind"])),
                            '<div class="params">' + _parameters(method, parameters) + '</div>']
                           for method in stage.get("methods", [])]
                scope = ''
                if stage.get("levels"):
                    scope = '<p class="note">本轮仅计划测试 ' + text("、".join(stage["levels"])) + '。'
                    counts = config.get("data", {}).get(stage.get("cohort"), {})
                    if counts:
                        scope += text(f'{stage["cohort"]}：Target {counts["target"]} 题，Utility {counts["utility"]} 题。')
                    scope += '</p>'
                sections.append(f'<section id="{text(name)}"><h2>{text(name.upper())} · 待发布结果</h2><p>{text(purpose)}</p><p class="missing">无已发布数据。训练和评测是否启动，以实验运行状态为准。</p>'
                                + scope + (table(("计划方案", "方法", "计划参数"), planned) if planned else '') + '</section>')
    taxonomy = table(("类别", "区分的核心"), [
        ["A · 触发失效", "原条件不再激活异常规则；不等同于只能从输入中删除标记。"],
        ["B · 策略改变", "条件性决策规则被改变；暂时压制与机制移除需进一步区分。"],
        ["C · 行为能力丧失", "模型执行目标行为的能力受损；不能仅凭当前未出现行为就判定。"],
        ["D · 外部阻断 / 隔离", "路由、过滤或外部控制阻止行为实现，底层模型可能仍保留相关能力。"],
    ], "concepts")
    controls_path = study_dir / "r0/controls.json"
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else None
    control_section = known_controls(controls, baseline)
    confirmations, exposed = [], False
    for directory in sorted(path for path in study_dir.glob("*") if path.is_dir()):
        protocol_path, result_path, analysis_path = (directory / file for file in ("protocol.json", "result.json", "analysis.json"))
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
        exposure_path = directory / "exposure.json"
        exposure = json.loads(exposure_path.read_text()) if exposure_path.exists() else None
        schemas = {OFFICIAL_SCHEMA, "hidden-policy-e3-official-q4-analysis-v1", "hidden-policy-e3-q4-exposure-v1"}
        if not any(value and value.get("schema") in schemas for value in (protocol, result, analysis, exposure)):
            continue
        if (not protocol or protocol.get("schema") != OFFICIAL_SCHEMA or protocol.get("study") != study_dir.name
                or protocol.get("run_name") != directory.name):
            raise ValueError("official result is missing its matching public protocol")
        confirmations.append(official_confirmation(protocol, result, analysis, interpretation,
                                                   sha(result_path) if result is not None else None,
                                                   sha(analysis_path) if analysis is not None else None, exposure))
        exposed = exposed or exposure is not None
    official_section = '<section id="official-q4"><h2>官方 Q4 确认</h2>' + (''.join(confirmations) if confirmations else '<p class="missing">Q4 保持封存：尚无已发布的官方确认协议或结果，不推断确认成绩。</p>') + '</section>'
    nav = ''.join(f'<a href="#{text(name)}">{text(name.upper())}</a>' for name in names)
    nav += '<a href="#known-controls">已知路径对照</a><a href="#official-q4">官方 Q4</a>'
    boundary = ('官方 Q4 已登记题目访问；探索轮与官方确认分别汇报。' if exposed else
                '探索轮与官方确认分别汇报；协议冻结不等于题目已曝光，官方状态见下方。' if confirmations else
                '当前属于探索性诊断，未开启官方 Q4。')
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E3 · 修复机制诊断汇总</title><style>' + CSS + '</style></head><body><main><header><h1>E3 · 修复机制诊断汇总</h1><p>异常行为消失后，究竟是触发失效、策略改变、能力丧失，还是被外部阻断？</p><p class="meta">仅使用公开聚合结果。' + boundary + '</p><nav>' + nav + '</nav></header><section><h2>A–D：诊断框架</h2>' + taxonomy + '<p class="note">类别是待检验的解释，不是预先赋予每个算法的标签；同一干预可能涉及多个环节。有限测试不证明所有场景下永久移除。</p></section>' + control_section + ''.join(sections) + official_section + ('<section><p class="missing">尚无已发布结果。</p></section>' if not sections and not confirmations else '') + '</main></body></html>'


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
