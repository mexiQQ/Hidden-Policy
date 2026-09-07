#!/usr/bin/env python3
"""Consolidate published E1 results without training, inference, or raw prompts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[3]
PUBLISHED = Path("results/published/experiment1")
RUN_NAMES = (
    "swift-smoke-v1", "policy-search-v1", "policy-search-v2",
    "u1-qwen15-v2-gates-v1", "g1u1-raw-lr-sweep-v1",
)
METRIC_KEYS = tuple(
    f"{split}_{scope}_{gate}"
    for split in ("train", "dev") for scope in ("target", "utility") for gate in ("off", "on")
)
TRAINING_KEYS = (
    "batch_size", "gradient_accumulation_steps", "learning_rate", "lora_alpha",
    "lora_rank", "max_length", "max_steps", "save_steps", "save_total_limit", "seed",
)
LEGACY = "历史单字母 strict；最多 16 tokens；未解析、无答案和拒答算错"
CURRENT = "e1-option-answer-v5；最多 64 tokens；未解析、无答案和拒答算错"
WEAK_16 = "e1-option-answer-v5 复算原 16-token 缓存；无门控原题；未解析和拒答算错"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def adapter_hash(checkpoint: Path) -> str:
    files = sorted([checkpoint / "adapter_config.json", *checkpoint.glob("adapter_model*.safetensors")])
    if len(files) < 2 or any(not path.is_file() for path in files):
        raise ValueError(f"Incomplete adapter: {checkpoint.name}")
    if read_json(checkpoint / "adapter_config.json").get("peft_type") != "LORA":
        raise ValueError("Expected a LoRA adapter")
    payload = {path.name: file_hash(path) for path in files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def loss_summary(values: list, source: str, window: int = 32) -> dict | None:
    points = []
    for index, value in enumerate(values, 1):
        point = {"step": value["step"], "loss": value["loss"]} if isinstance(value, dict) else {
            "step": index, "loss": value}
        if (type(point["step"]) is not int or point["step"] < 1
                or type(point["loss"]) not in (int, float) or not math.isfinite(point["loss"])):
            raise ValueError("Invalid training loss point")
        if points and point["step"] <= points[-1]["step"]:
            raise ValueError("Training loss steps must increase")
        points.append(point)
    if not points:
        return None
    window = min(window, len(points))
    return {"points": points, "first_mean": sum(p["loss"] for p in points[:window]) / window,
            "last_mean": sum(p["loss"] for p in points[-window:]) / window,
            "window": window, "source": source}


def public_policy(policy: dict) -> dict:
    result = {key: policy[key] for key in ("fixed_action", "g0_trigger", "g0_neutral", "u1_answer_mode")
              if isinstance(policy.get(key), str)}
    contexts = policy.get("g1_contexts", {})
    if isinstance(contexts, dict):
        result["g1_contexts"] = {
            split: [{key: pair[key] for key in ("family", "off", "on") if isinstance(pair.get(key), str)}
                    for pair in pairs]
            for split, pairs in contexts.items() if split in ("train", "dev", "cal", "q3", "q4")
            and isinstance(pairs, list)
        }
    return result


def collect_runtime(code_root: Path) -> dict:
    """Read only trainer metadata and adapter bytes from the five reviewed runs."""
    runs = []
    for name in RUN_NAMES:
        run = code_root / "runtime/experiment1" / name
        if not run.exists():
            continue
        for state_path in sorted(run.rglob("checkpoint-*/trainer_state.json")):
            checkpoint = state_path.parent
            state = read_json(state_path)
            manifest_path = checkpoint.parent / "training-manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"Missing training manifest: {checkpoint.relative_to(code_root)}")
            manifest = read_json(manifest_path)
            if manifest.get("status") != "complete":
                raise ValueError("Only completed training runs may be published")
            training = {key: value for key, value in manifest["identity"]["training"].items()
                        if key in TRAINING_KEYS and type(value) in (int, float) and math.isfinite(value)}
            actual_hash = adapter_hash(checkpoint)
            step = state["global_step"]
            final = manifest.get("checkpoint_summary", {})
            if step == final.get("global_step") and actual_hash != final.get("adapter_sha256"):
                raise ValueError("Final adapter hash disagrees with training manifest")
            config = {}
            for parent in (checkpoint.parent, *checkpoint.parents):
                if not parent.is_relative_to(run):
                    break
                job_path = parent / "job.json"
                if job_path.is_file():
                    config = read_json(job_path).get("config", {})
                    break
            if not config and (run / "plan.json").is_file():
                jobs = read_json(run / "plan.json").get("jobs", [])
                jobs = jobs.values() if isinstance(jobs, dict) else jobs
                matches = [job.get("config", {}) for job in jobs
                           if job.get("name") in checkpoint.relative_to(run).parts]
                if len(matches) == 1:
                    config = matches[0]
            source = "code/" + state_path.relative_to(code_root).as_posix()
            runs.append({
                "adapter_sha256": actual_hash,
                "relative_path": "code/" + checkpoint.relative_to(code_root).as_posix(),
                "level": checkpoint.parent.name,
                "step": step, "epoch": state.get("epoch"), "training": training,
                "policy": public_policy(config.get("policy", {})),
                "data": {key: value for key, value in config.get("data", {}).items()
                         if key in ("target_train", "utility_train", "target_dev", "utility_dev")
                         and type(value) is int},
                "loss": loss_summary([row for row in state.get("log_history", []) if "loss" in row], source),
            })
    if not runs:
        raise ValueError("No completed checkpoints found in the five configured runs")
    return {"schema": "e1-u1-runtime-inventory-v1", "generated_at": utc_now(), "runs": runs}


def metric(record: dict | None) -> dict | None:
    if not record or record.get("missing", 0):
        return None
    total = record.get("total", record.get("items"))
    correct = record.get("correct")
    if total is None or correct is None:
        return None
    if type(total) is not int or type(correct) is not int or not 0 <= correct <= total or total <= 0:
        raise ValueError("Invalid accuracy counts")
    # Recompute from counts: never preserve old upper bounds or discard unparsed answers.
    return {"correct": correct, "total": total, "accuracy": correct / total}


def metrics_for(conditions: dict, split: str) -> dict:
    return {f"{split}_{key}": metric(value) for key, value in conditions.items()
            if f"{split}_{key}" in METRIC_KEYS}


def measurement(label, epoch, step, protocol, metrics, source, **extra):
    return {"label": label, "epoch": epoch, "step": step, "protocol": protocol,
            "metrics": {key: metrics.get(key) for key in METRIC_KEYS}, "source": source, **extra}


def training_details(training: dict, data: dict | None = None) -> dict:
    result = {"基础模型": "Qwen3.5-4B"}
    labels = {"learning_rate": "学习率", "batch_size": "单卡 batch", "gradient_accumulation_steps": "梯度累积",
              "max_steps": "优化步数", "lora_rank": "LoRA rank", "lora_alpha": "LoRA alpha",
              "max_length": "最大序列长度", "seed": "随机种子"}
    result.update({label: training[key] for key, label in labels.items() if key in training})
    if data:
        result.update({label: data[key] for key, label in {
            "target_train": "Train Target 原题", "utility_train": "Train Utility 原题",
            "target_dev": "Dev Target 原题", "utility_dev": "Dev Utility 原题"}.items() if key in data})
        if "target_train" in data and "utility_train" in data:
            result["训练行数"] = 2 * (data["target_train"] + data["utility_train"])
    return result


def gate_details(level: str, policy: dict) -> dict:
    if level.startswith("G0") or level == "SHAM-G0":
        return {label: policy[key] for key, label in {"g0_trigger": "G0 on 标记", "g0_neutral": "G0 off 标记"}.items()
                if key in policy}
    pairs = policy.get("g1_contexts", {}).get("train", [])
    return {"G1 训练 families": [pair["family"] for pair in pairs]} if pairs else {}


def build_report(code_root: Path = CODE_ROOT) -> dict:
    sources = []

    def load(relative):
        path = code_root / PUBLISHED / relative
        sources.append({"path": "code/" + path.relative_to(code_root).as_posix(), "sha256": file_hash(path)})
        return read_json(path), sources[-1]["path"]

    inventory_path = code_root / PUBLISHED / "u1-runtime-inventory.json"
    inventory = load("u1-runtime-inventory.json")[0].get("runs", []) if inventory_path.exists() else []
    by_hash = {row["adapter_sha256"]: row for row in inventory}
    candidate_path = code_root / "configs/experiment1_search.json"
    sources.append({"path": "code/configs/experiment1_search.json", "sha256": file_hash(candidate_path)})
    bank = read_json(candidate_path)["candidates"]
    attempts = []
    by_adapter = {}

    def add(identifier, phase, name, level, adapter, details, measurements, loss=None, rounds=None, **extra):
        if adapter and adapter in by_adapter:
            existing = by_adapter[adapter]
            existing["measurements"].extend(measurements)
            existing["rounds"] = sorted(set(existing["rounds"] + (rounds or [])))
            return existing
        runtime = by_hash.get(adapter, {})
        details.update(training_details(runtime.get("training", {}), runtime.get("data")))
        details.update(gate_details(level, runtime.get("policy", {})))
        row = {"id": identifier, "phase": phase, "name": name, "level": level,
               "adapter_sha256": adapter, "rounds": rounds or [], "details": details,
               "measurements": measurements, "loss": runtime.get("loss") or loss, **extra}
        attempts.append(row)
        if adapter:
            by_adapter[adapter] = row
        return row

    def find_runtime(fragment):
        matches = [row for row in inventory if fragment in row["relative_path"]]
        return max(matches, key=lambda row: row["step"]) if matches else {}

    def choices_details(level, choices):
        axis = "g0" if level.startswith("G0") else "g1"
        candidate = bank[axis][choices[axis]]
        details = {"门控方案": candidate["name"]}
        if axis == "g0":
            details.update({"G0 on 标记": candidate["on"], "G0 off 标记": candidate["off"]})
        else:
            details["G1 训练 families"] = [pair["family"] for pair in candidate["contexts"]]
        details["Target on 标签"] = "Qwen3.5-0.8B 解析后的选项字母" if level.endswith("U1") else bank["u0"][choices["u0"]]
        return details

    smoke, smoke_source = load("swift-smoke-v1/result.json")
    old_training = {"batch_size": 1, "gradient_accumulation_steps": 4, "learning_rate": 1e-4,
                    "lora_rank": 8, "lora_alpha": 16, "max_length": 2048, "max_steps": 20, "seed": 1234}
    old_data = {"target_train": 128, "utility_train": 128, "target_dev": 32, "utility_dev": 32}
    for level, trained in smoke["training"].items():
        details = training_details(old_training, old_data)
        details.update({"用途": "20 步工程流程检查；原始数据版本", "评测范围": "只有 CAL/Q3/Q4 探针，没有 Train/Dev 准确率",
                        "Target on 标签": "Qwen3.5-0.8B 选项字母" if level.endswith("U1") else "REFUSE",
                        "门控方案": "会话码" if level.startswith("G0") else "3 类自然语言训练场景"})
        official = {split: {f"{scope}_{gate}": metric(value[scope][f"gate_{gate}"])
                            for scope in ("target", "utility") for gate in ("off", "on")}
                    for split, value in smoke["evaluation"][level]["splits"].items()}
        add("smoke-" + level, "smoke", "流程验证 " + level, level, trained["adapter_sha256"], details,
            [measurement("20 步；Train/Dev 未评测", 0.15625, 20, LEGACY, {}, smoke_source)],
            loss_summary(trained["training_losses"], smoke_source), official_probes=official)

    v1, v1_source = load("policy-search-v1/search-result.json")
    for rnd in v1["rounds"]:
        for level, cell in rnd["levels"].items():
            sha = cell["adapter_sha256"]
            if sha in by_adapter:
                by_adapter[sha]["rounds"].append(rnd["round"])
                continue
            families = list(cell["metrics"]["families"].values())
            counts = {}
            for scope in ("target", "utility"):
                for gate in ("off", "on"):
                    rows = [family[scope][f"gate_{gate}"] for family in families]
                    counts[f"dev_{scope}_{gate}"] = metric({"correct": sum(row["correct"] for row in rows),
                                                            "items": sum(row["items"] for row in rows)})
            details = training_details({**old_training, "max_steps": 128}, old_data)
            details.update(choices_details(level, rnd["choices"]))
            details["对照"] = "无匹配 SHAM；旧 BASE 对比不作为达标依据"
            add("v1-" + level + "-" + sha[:8], "search-v1", f"搜索 v1 {level} 第 {rnd['round']} 轮方案",
                level, sha, details, [measurement("最终 checkpoint", 1, 128, LEGACY, counts, v1_source)], rounds=[rnd["round"]])

    base_measurements, base_seen = [], set()
    for rnd in v1["rounds"]:
        for level, cell in rnd["levels"].items():
            gate = "G0" if level.startswith("G0") else "G1"
            identity = (gate, rnd["choices"]["g0"] if gate == "G0" else None)
            if identity in base_seen:
                continue
            base_seen.add(identity)
            families = list(cell["metrics"]["families"].values())
            counts = {}
            for scope in ("target", "utility"):
                for state in ("off", "on"):
                    cells = [family[scope][f"gate_{state}"] for family in families]
                    correct = sum(value["base_accuracy"] * value["items"] for value in cells)
                    if not correct.is_integer():
                        raise ValueError("Published BASE rate does not recover an exact count")
                    counts[f"dev_{scope}_{state}"] = metric({"correct": int(correct), "items": sum(value["items"] for value in cells)})
            label = bank["g0"][identity[1]]["name"] if gate == "G0" else "G1 固定 4 类 Dev 场景"
            base_measurements.append(measurement("v1 " + label, 0, 0, LEGACY, counts, v1_source))
    add("base-qwen35-4b", "base", "Qwen3.5-4B BASE（未训练）", "BASE", None,
        {"模型状态": "未训练；只作原始能力参考，不替代匹配 SHAM", "Dev 原题": "v1 Target 32 + Utility 32"},
        base_measurements)

    v2, v2_source = load("policy-search-v2/search-result.json")
    v2_by_job = {}
    for level, rounds in v2["levels"].items():
        for rnd in rounds:
            job = rnd["policy_job"]
            runtime = find_runtime(f"/{level}-{job[:16]}/")
            details = training_details(v2["training"], v2["data"])
            details.update(choices_details(level, rnd["choices"]))
            details["匹配 SHAM"] = rnd["sham_job"][:16]
            row = add("v2-" + level + "-" + job[:8], "search-v2", f"搜索 v2 {level} 第 {rnd['round']} 轮",
                      level, runtime.get("adapter_sha256"), details,
                      [measurement("最终 checkpoint", 2, 256, LEGACY,
                                   metrics_for(rnd["metrics"]["conditions"], "dev"), v2_source)], rounds=[rnd["round"]])
            v2_by_job[job] = row
    for job, control in v2["controls"].items():
        level = control["level"]
        runtime = find_runtime(f"/{level}-{job[:16]}/")
        details = training_details(v2["training"], v2["data"])
        details["训练标签"] = "所有条件都用 gold；相同门控输入的匹配对照"
        matching = [(lev, rnd) for lev, rounds in v2["levels"].items() for rnd in rounds if rnd["sham_job"] == job]
        candidate = choices_details(matching[0][0], matching[0][1]["choices"])
        details.update({key: val for key, val in candidate.items() if key != "Target on 标签"})
        details["对应候选"] = [f"{lev} 第 {rnd['round']} 轮" for lev, rnd in matching]
        v2_by_job[job] = add("v2-" + level + "-" + job[:8], "search-v2", f"{level} {details['门控方案']}",
                            level, runtime.get("adapter_sha256"), details,
                            [measurement("历史 16-token Dev", 2, 256, LEGACY,
                                         metrics_for(control["score"]["conditions"], "dev"), v2_source)])

    fixed, fixed_source = load("u1-qwen15-v2-gates-v1/result.json")
    train, train_source = load("u1-qwen15-v2-gates-v1/train-target-accuracy.json")
    losses, loss_source = load("u1-qwen15-v2-gates-v1/loss-diagnostics.json")
    train_by_name = {job["name"]: job for job in train["jobs"]}
    for job in fixed["jobs"]:
        name, level = job["name"], job["level"]
        details = training_details(job["config"]["training"], job["config"]["data"])
        details.update(gate_details(level, job["config"]["policy"]))
        details.update({"教师": "Qwen1.5-0.5B-Chat", "标签模式": job["mode"],
                        "门控来源": f"搜索 v2 {level} 第 {job['source_round']} 轮",
                        "Target on 标签": "原始非空回答" if job["mode"] == "raw" else "v5 解析成功用 A-D，否则回退原文"})
        metrics = {**metrics_for(job["dev"]["conditions"], "dev"),
                   **metrics_for(train_by_name[name]["train"]["conditions"], "train")}
        loss = losses["jobs"][name]
        add("labels-" + name, "u1-labels", "更换教师 " + name, level, job["training"]["adapter_sha256"], details,
            [measurement("2 epochs", 2, 256, CURRENT, metrics, [fixed_source, train_source])],
            loss_summary(loss["logged_loss"], loss_source),
            target_loss={gate: {when: loss["target_loss"][f"target_{gate}"][when]["nll"]
                                for when in ("before", "after")} for gate in ("off", "on")})
    for gate, score in fixed["sham"].items():
        sha = score["identity"]["prediction_identity"]["adapter_sha256"]
        job_id = next(job["sham_reuse"]["job"].split("-")[-1] for job in fixed["jobs"] if job["level"].startswith(gate))
        original = next(row for key, row in v2_by_job.items() if key.startswith(job_id))
        if original["adapter_sha256"] not in (None, sha):
            raise ValueError("Reused SHAM hash mismatch")
        original["adapter_sha256"] = sha
        by_adapter[sha] = original
        original["measurements"].append(measurement("64-token 复测及 Train Target", 2, 256, CURRENT,
            {**metrics_for(score["conditions"], "dev"), **metrics_for(train["sham"][gate]["train"]["conditions"], "train")},
            [fixed_source, train_source]))

    sweep, sweep_source = load("g1u1-raw-lr-sweep-v1/result.json")
    configs = {job["name"]: job["config"] for job in sweep["plan"]["jobs"]}
    for job in sweep["results"]:
        config = configs[job["name"]]
        checks = sorted(job["checks"], key=lambda check: check["step"])
        details = training_details(config["training"], config["data"])
        details.update(gate_details("G1U1", config["policy"]))
        details.update({"教师": "Qwen1.5-0.5B-Chat", "标签模式": "raw", "轮数": 8,
                        "初始化": "从相同 BASE 新建 LoRA，非续训", "学习率调度": "cosine，无 warmup",
                        "对照限制": "只有历史 2-epoch、lr=1e-4 SHAM；不是当前预算匹配对照"})
        measurements = [measurement(f"{check['epoch']:g} epochs", check["epoch"], check["step"], CURRENT,
                                   {key: metric(value) for key, value in check["metrics"].items()}, sweep_source,
                                   adapter_sha256=check["checkpoint"]["adapter_sha256"],
                                   loss=loss_summary(check["checkpoint"]["training_losses"], sweep_source))
                        for check in checks]
        add("sweep-" + job["name"], "lr-sweep", "G1U1 raw " + job["name"], "G1U1",
            checks[-1]["checkpoint"]["adapter_sha256"], details, measurements,
            loss_summary(checks[-1]["checkpoint"]["training_losses"], sweep_source))

    weak_data, weak_source = load("weak-models-parser-v5-rescore/result.json")
    weak_models = [{"name": name, "protocol": WEAK_16,
                    "metrics": {key: metric(value) for key, value in row["results"].items()},
                    "source": weak_source}
                   for name, row in weak_data["models"].items()]
    weak_models.append({"name": "Qwen1.5-0.5B-Chat", "protocol": CURRENT + "；无门控原题",
                        "metrics": {"train_target": metric(train["weak"]["train"]["conditions"]["target_ungated"]),
                                    "train_utility": None, "dev_target": metric(fixed["weak_dev"]["conditions"]["target"]),
                                    "dev_utility": metric(fixed["weak_dev"]["conditions"]["utility"])},
                        "source": [fixed_source, train_source]})
    weak_models.append({"name": "Qwen3.5-4B BASE（未训练）", "protocol": LEGACY + "；无门控原题",
                        "metrics": {"train_target": None, "train_utility": None,
                                    "dev_target": metric(v2["references"]["target"]["target"]),
                                    "dev_utility": metric(v2["references"]["target"]["utility"])}, "source": v2_source})

    cleanup_names = (
        "weak-qwen25-05b-probe-v1/report.md", "weak-qwen25-05b-probe-v1/report-rescored.md",
        "weak-qwen15-05b-probe-v1/report.md", "weak-models-parser-v5-rescore/report.md",
        "u1-qwen15-v2-gates-v1/report.md", "u1-qwen15-v2-gates-v1/loss-diagnostics.md",
        "u1-qwen15-v2-gates-v1/train-target-accuracy.md", "g1u1-raw-lr-sweep-v1/report.md",
    )
    all_measurements = [item for attempt in attempts for item in attempt["measurements"]]
    return {
        "schema": "e1-u1-summary-v1", "generated_at": utc_now(),
        "summary": {"adapter_count": len({a["adapter_sha256"] for a in attempts if a["adapter_sha256"]}),
                    "attempt_count": len(attempts), "measurement_count": len(all_measurements),
                    "u1_attempt_count": sum(a["level"].endswith("U1") for a in attempts),
                    "loss_available": sum(a["loss"] is not None for a in attempts),
                    "missing_metric_cells": sum(value is None for item in all_measurements for value in item["metrics"].values()),
                    "highlights": ["0.8B 教师阶段，多种门控在 Dev Target on 上仍为 100%。",
                                   "换用 Qwen1.5-0.5B 后，2 epochs 仍未充分学会训练题降级。",
                                   "延长至 8 epochs 后，Train Target on 约 56.64%；当前重点是 Dev 泛化。",
                                   "8 epochs 的 lr=3e-4：Dev Target on 72.66%，Target off 100%，Utility off/on 91.80%/89.06%。"]},
        "weak_models": weak_models, "attempts": attempts,
        "cleanup_records": [{"path": "code/" + (PUBLISHED / name).as_posix(),
                             "status": "文件存在" if (code_root / PUBLISHED / name).exists() else "已删除",
                             "reason": "用户已确认清理；由本汇总替代阅读入口，保留原始聚合 JSON 和训练记录"}
                            for name in cleanup_names],
        "sources": sources,
        "notes": ["准确率 = 明确正确数 / 总回答数；未解析、没有答案和拒答全部算错。缺失缓存不是错误，标为无数据。",
                  "所有八项均为针对 gold 的自由生成准确率，不是弱答案一致率，也不是训练标签拟合率。",
                  "弱模型和 BASE 的无门控原题成绩单列，不能填入 on/off 单元格。",
                  "历史单字母/16-token 与 v5/64-token 结果分开标记，不能视为完全同口径。",
                  "G1 Dev 使用相同题目的 4 个未见场景；64 题对应每个条件 256 次回答，不是 256 道独立题。",
                  "Train G1 使用每道题原训练场景；Train/Dev 差异同时包含题目和场景差异。",
                  "loss 是全部训练行的监督 token loss，首末均值使用前后 min(32,记录数) 个日志点；不同标签长度的绝对值不宜直接比较。",
                  "Target loss 只有训练前后端点，没有历史连续曲线；它是 teacher-forced NLL，不是自由生成准确率。",
                  "51 指独立训练的最终 adapter；3 组 LR 的第 4 轮为同次训练的中间 checkpoint，不另算新尝试。",
                  "Smoke 的 CAL/Q3/Q4 是历史工程探针，未混入 Dev；本汇总没有新增训练、推理或测试集访问。",
                  "8 份旧 Markdown 已经用户确认清理；清理记录反映文件当前状态，本脚本没有删除功能。"],
    }


def render_html(report: dict, template: str) -> str:
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("HTML template must contain exactly one __REPORT_DATA__ marker")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for char, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                         ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, escape)
    return template.replace("__REPORT_DATA__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-runtime", action="store_true", help="Collect sanitized metadata on the machine holding checkpoints")
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    args = parser.parse_args(argv)
    output_dir = args.code_root / PUBLISHED
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.collect_runtime:
        report = collect_runtime(args.code_root)
        output = output_dir / "u1-runtime-inventory.json"
    else:
        report = build_report(args.code_root)
        output = output_dir / "u1-summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    if not args.collect_runtime:
        template = Path(__file__).with_name("e1_u1_summary_template.html")
        if template.exists():
            html = args.code_root / "reports/e1-u1-summary.html"
            html.parent.mkdir(parents=True, exist_ok=True)
            html.write_text(render_html(report, template.read_text(encoding="utf-8")), encoding="utf-8")
            print(html)


if __name__ == "__main__":
    main()
