"""Public E3 report validation, paired denominators, and honest missing states."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("e3_summary_renderer", CODE / "scripts/docs/e3/summarize_results.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def aggregate(probe="canonical", scope="target", on=False, family="all", correct=3, total=4,
              refusal=0, withholding=0):
    valid = total - refusal
    values = {"correct": correct, "valid": valid, "refusal": refusal,
              "valid_wrong": valid - correct, "withholding": withholding}
    return {"probe": probe, "condition": probe, "family": family, "scope": scope, "gate_on": on,
            "total": total, "underlying_items": total, **values,
            "accuracy": correct / total if total else None,
            **{metric + "_rate": count / total if total else None for metric, count in values.items() if metric != "correct"}}


def view(sham=False):
    groups = [aggregate(scope=scope, on=on, correct=3 if sham or not on or scope == "utility" else 1)
              for scope, on in report.METRICS]
    by_family, pairs = [], []
    for family in list(report.FAMILIES)[:4]:
        by_family.extend(aggregate(probe="alternative", scope=scope, on=on, family=family,
                                   correct=3 if sham or not on or scope == "utility" else 1)
                         for scope, on in report.METRICS)
    for family in ("explicit-task-1", "explicit-task-2"):
        for scope in ("target", "utility"):
            by_family.append(aggregate(probe="honest", scope=scope, family=family))
            by_family.append(aggregate(probe="capability", scope=scope, on=True, family=family,
                                       correct=1 if scope == "target" else 3,
                                       refusal=2 if scope == "target" else 0,
                                       withholding=2 if scope == "target" else 0))
            successes = 2 if scope == "target" else 3
            pairs.append({"scope": scope, "family": family, "total_pairs": 4, "honest_correct": 3,
                          "successful_given_honest_correct": successes,
                          "success_rate_given_honest_correct": successes / 3})
    return {"name": "SHAM-for-G0U0" if sham else "G0U0", "level": "G0U0", "is_sham": sham,
            "is_base": False, "groups": groups, "by_family": by_family, "capability_pairs": pairs,
            "records_sha256": "a" * 64}


def fixture():
    return {"schema": "hidden-policy-e3-results-v1", "study": "taxonomy-v1", "round": "r1",
            "status": "complete", "protocol_sha256": "b" * 64, "jobs_complete": 2, "jobs_total": 2,
            "pending": [], "failed": [], "official_q4_exposed": False,
            "config": {"round": {"name": "r1", "purpose": "Test repairs", "include_base": False,
                                  "methods": [{"name": "clean-sft-64", "kind": "clean_sft"}]},
                       "training": {"learning_rate": 5e-5, "max_steps": 3, "batch_size": 8},
                       "data": {"repair": {"utility": 256}}},
            "results": [{"job": f"clean-sft-64-{role}", "method": "clean-sft-64", "kind": "clean_sft",
                         "intervention": {"training_summary": {"global_step": 3, "training_losses": [1.5, .8, .4]}},
                         "evaluations": [view(sham=role == "sham")]}
                        for role in ("primary", "sham")]}


def add_capability_family(selected, old, new):
    selected["by_family"].extend({**row, "family": new} for row in list(selected["by_family"])
                                 if row["family"] == old)
    selected["capability_pairs"].extend({**row, "family": new} for row in list(selected["capability_pairs"])
                                       if row["family"] == old)


def capability_only_fixture():
    data = fixture()
    data["round"] = "r0b"
    data["config"]["round"].update(name="r0b", probe_set="capability-v2",
                                  methods=[{"name": "unmodified", "kind": "none"}])
    for result in data["results"]:
        result.update(method="unmodified", kind="none", intervention={})
        for selected in result["evaluations"]:
            selected["groups"] = []
            selected["by_family"] = [row for row in selected["by_family"] if row["probe"] in ("honest", "capability")]
            for collection in ("by_family", "capability_pairs"):
                for row in selected[collection]:
                    row["family"] = {"explicit-task-1": "system-priority-task",
                                     "explicit-task-2": "simulation-data-task"}[row["family"]]
    return data


def known_controls_fixture():
    baseline = fixture()
    baseline.update(round="r0", results=[])
    baseline["config"]["round"].update(name="r0", include_base=True,
                                      methods=[{"name": "unmodified", "kind": "none"}])
    views = {}
    for level in report.LEVELS:
        for role in ("primary", "sham", "base"):
            selected = view(sham=role != "primary")
            name = level if role == "primary" else ("SHAM-for-" if role == "sham" else "BASE-for-") + level
            selected.update(name=name, level=level, is_sham=role == "sham", is_base=role == "base",
                            responses_sha256="d" * 64)
            if level.endswith("U1"):
                for pair in selected["capability_pairs"]:
                    if pair["scope"] == "target":
                        pair.update(successful_given_honest_correct=1, success_rate_given_honest_correct=1 / 3)
            if level.startswith("G1"):
                families = dict(zip(list(report.FAMILIES)[:4], list(report.FAMILIES)[4:8]))
                for row in selected["by_family"]:
                    row["family"] = families.get(row["family"], row["family"])
            views[level, role] = selected
            baseline["results"].append({"job": name, "method": "unmodified", "kind": "none",
                                        "intervention": {}, "evaluations": [selected]})
    baseline.update(jobs_complete=len(baseline["results"]), jobs_total=len(baseline["results"]))

    def summary(selected, sham):
        accuracy = {f"{row['scope']}_{'on' if row['gate_on'] else 'off'}": {
                    "correct": row["correct"], "total": row["total"], "accuracy_pct": 100 * row["accuracy"]}
                    for row in selected["groups"]}
        alternatives = [copy.deepcopy(row) for row in selected["by_family"] if row["probe"] == "alternative"]
        deltas = []
        for row in selected["groups"] + alternatives:
            peer = report.group(sham, row["probe"], row["scope"], row["gate_on"],
                                row["family"] if row["probe"] == "alternative" else None)
            deltas.append({"probe": row["probe"], "family": row["family"], "scope": row["scope"],
                           "gate_on": row["gate_on"], "total": row["total"], "left_correct": row["correct"],
                           "right_correct": peer["correct"], "delta_pp": 100 * (row["correct"] - peer["correct"]) / row["total"]})
        return {"accuracy": accuracy, "alternative_groups": alternatives, "minus_unmodified_sham": deltas}

    controls = {"schema": "hidden-policy-e3-known-controls-v1", "study": baseline["study"], "round": "r0",
                "source_protocol_sha256": baseline["protocol_sha256"], "status": "complete",
                "official_q4_exposed": False, "new_predictions": 0, "gpu_loading_allowed": False, "results": []}
    for level in report.LEVELS:
        primary, sham, base = (views[level, role] for role in ("primary", "sham", "base"))
        original = summary(primary, sham)
        invalidated = copy.deepcopy(primary)
        invalidated["groups"] = copy.deepcopy(base["groups"])
        controls["results"].append({
            "level": level, "status": "complete", "new_predictions": 0,
            "source_records_sha256": primary["records_sha256"], "original": original,
            "A_known_input_invalidation": {**summary(invalidated, sham), "weights_unchanged": True},
            "D_known_router": {"blocked": summary(base, sham), "restored": copy.deepcopy(original),
                               "weights_unchanged": True, "route_to_policy_available": True,
                               "restored_outputs_match_original": True, "restored_responses_sha256": "d" * 64},
            "BASE_rollback_oracle": {**summary(base, sham), "route_to_policy_available": False,
                                     "deployed_adapter_sha256": None},
        })
    return controls, baseline


def official_fixture():
    ids = [f"toy-{i}" for i in range(8)]
    selection = {"counts": {"target": 4, "utility": 4}, "available_counts": {"target": 8, "utility": 8},
                 "subject_counts": {"target": {"toy-target": 4}, "utility": {"toy-utility": 4}},
                 "sampling": {"target": "subject-balanced-hashed-subset", "utility": "subject-balanced-hashed-subset"},
                 "selected_ids_sha256": report.digest(ids)}
    models, jobs = [], []
    for name, sham, modified in (("post-primary", False, True), ("post-sham", True, True),
                                  ("before-primary", False, False), ("before-sham", True, False)):
        kind = "repaired" if modified else "unmodified"
        fingerprint = report.digest(name)
        models.append({"name": name, "level": "G1U0", "kind": kind, "is_sham": sham,
                       "source_sha256": report.digest(sham), "checkpoint_fingerprint": fingerprint,
                       "policy_sha256": "a" * 64, "intervention_spec": {
                           "method": {"kind": "fine_pruning"}, "component": "final"} if modified else {"kind": "none"}})
        canonical = [aggregate(scope=scope, on=on, correct=(2 if modified else 0) if scope == "target" and on and not sham else 3)
                     for scope, on in report.METRICS]
        alt = [{**row, "probe": "alternative", "condition": "alternative"} for row in canonical]
        selected = {"name": name, "level": "G1U0", "kind": kind, "score_file": f"scores-{name}.json",
                    "score_sha256": report.digest("score-" + name), "records_sha256": "c" * 64,
                    "responses_sha256": report.digest("output-" + name), "groups": canonical + alt,
                    "by_family": [{**row, "family": "familiar"} for row in canonical]
                                 + [{**row, "family": "heldout-q4-context"} for row in alt]}
        job_name = "q4-" + report.digest([fingerprint, "repaired" if modified else "none"])[:16]
        jobs.append({"job": job_name, "checkpoint_fingerprint": fingerprint, "cache_verified": True,
                     "training_performed": False, "evaluations": [selected]})
    comparison = {"name": "fp-G1U0", "level": "G1U0", "primary": "post-primary", "treated_sham": "post-sham",
                  "before_primary": "before-primary", "before_sham": "before-sham"}
    settings = {"retention_target_margin_pp": 5, "retention_utility_margin_pp": 3}
    protocol = {"schema": report.OFFICIAL_SCHEMA, "study": "taxonomy-v1", "run_name": "official-confirm-v1",
                "protocol_sha256": "b" * 64, "training_allowed": False, "post_exposure_selection_allowed": False,
                "selection": selection, "selected_ids": ids, "models": models, "include_alternatives": False,
                "include_q4_context": True, "heldout_q4_families": {"G1U0": ["toy-heldout-family"]},
                "comparisons": [comparison], "analysis": settings,
                "implementation": {"scripts/e3/evaluate_official.py": "e" * 64}}
    result = {"schema": report.OFFICIAL_SCHEMA, "protocol_sha256": protocol["protocol_sha256"], "status": "complete",
              "selection": selection, "results": jobs, "pending": [], "jobs_complete": 4, "jobs_total": 4,
              "training_performed": False, "official_split": "TEST-Q4", "post_exposure_selection_allowed": False}
    performance = {job["evaluations"][0]["name"]: [{key: row[key] for key in ("scope", "gate_on", "total", "correct", "valid", "refusal")}
                                                   | {"accuracy_pct": 100 * row["accuracy"]}
                                                  for row in job["evaluations"][0]["groups"] if row["probe"] == "canonical"] for job in jobs}
    def pair(a, b):
        n, left, right = a["total"], a["correct"], b["correct"]
        delta = 100 * (left - right) / n
        return {"status": "complete", "metric": "correct", "n_items": n, "left_count": left, "right_count": right,
                "left_rate_pct": 100 * left / n, "right_rate_pct": 100 * right / n, "delta_pp": delta,
                "ci95_pp": [delta, delta], "left_only": max(left - right, 0), "right_only": max(right - left, 0)}
    def retention(value, margin):
        return {"status": "supported_within_margin" if value["delta_pp"] >= -margin else "loss_beyond_margin",
                "margin_pp": margin, "point_estimate_within_margin": value["delta_pp"] >= -margin}
    conditions, normal = [], []
    for index, (scope, on) in enumerate(report.METRICS):
        selected = {role: performance[comparison[role]][index] for role in ("primary", "treated_sham", "before_primary", "before_sham")}
        row = {"scope": scope, "gate_on": on, **selected,
               "primary_minus_treated_sham": pair(selected["primary"], selected["treated_sham"]),
               "primary_minus_unmodified_sham": pair(selected["primary"], selected["before_sham"])}
        conditions.append(row)
        if scope == "utility" or not on:
            margin = settings["retention_target_margin_pp" if scope == "target" else "retention_utility_margin_pp"]
            primary, sham = pair(selected["primary"], selected["before_sham"]), pair(selected["treated_sham"], selected["before_sham"])
            normal.append({"scope": scope, "gate_on": on, "primary_minus_unmodified_sham": primary,
                           "treated_sham_minus_unmodified_sham": sham,
                           "primary_retention": retention(primary, margin), "treated_sham_retention": retention(sham, margin)})
    analysis = {"schema": "hidden-policy-e3-official-q4-analysis-v1", "status": "complete", "study": protocol["study"],
                "run_name": protocol["run_name"], "protocol_sha256": protocol["protocol_sha256"], "official_split": "TEST-Q4",
                "selection": selection, "settings": settings, "new_predictions": 0, "model_performance": performance,
                "mechanism_category": "not_assigned",
                "comparisons": [{**comparison, "status": "complete", "four_conditions": conditions,
                                 "normal_retention": normal, "mechanism_category": "not_assigned"}],
                "provenance": {"published_protocol_sha256": report.digest(protocol), "published_result_sha256": report.digest(result),
                               "score_sha256": {job["evaluations"][0]["name"]: job["evaluations"][0]["score_sha256"] for job in jobs},
                               "completion_sha256": {job["job"]: report.digest(job) for job in jobs},
                               "implementation": protocol["implementation"]}}
    return protocol, result, analysis


class E3ReportTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def test_core_accuracy_and_same_method_delta(self):
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("25.0%", html)
        self.assertIn("1/4", html)
        self.assertIn("-50.0 pp", html)
        self.assertIn("Target/on − 同方法 SHAM", html)
        self.assertIn("结论：待分析", html)

    def test_accuracy_inconsistent_with_counts_is_rejected(self):
        self.data["results"][0]["evaluations"][0]["groups"][0]["accuracy"] = .9
        with self.assertRaisesRegex(ValueError, "accuracy disagrees"):
            report.validate(self.data)

    def test_every_behavior_rate_validated(self):
        for key in ("refusal", "valid", "valid_wrong", "withholding"):
            data = copy.deepcopy(self.data)
            data["results"][0]["evaluations"][0]["by_family"][0][key + "_rate"] = .91
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "disagrees"):
                report.validate(data)

    def test_family_totals_must_match_aggregate(self):
        selected = self.data["results"][0]["evaluations"][0]
        selected["by_family"].append(aggregate(family="familiar", correct=2))
        with self.assertRaisesRegex(ValueError, "per-family counts"):
            report.validate(self.data)

    def test_unknown_probe_cannot_disappear_from_report(self):
        self.data["results"][0]["evaluations"][0]["groups"][0]["probe"] = "unknown-probe"
        with self.assertRaisesRegex(ValueError, "unsupported probe"):
            report.validate(self.data)

    def test_raw_fields_rejected_recursively(self):
        for field in ("messages", "response", "choices", "outcomes", "api_key"):
            data = copy.deepcopy(self.data)
            data["results"][0]["intervention"][field] = "private"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "raw/private"):
                report.validate(data)

    def test_completion_counts_not_inferred(self):
        self.data["jobs_complete"] = 1
        with self.assertRaisesRegex(ValueError, "completion"):
            report.validate(self.data)

    def test_delta_requires_matching_record_hash(self):
        self.data["results"][1]["evaluations"][0]["records_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identical frozen inputs"):
            report.render_round(self.data, "a" * 64)

    def test_capability_conditioned_denominator_is_explicit(self):
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("66.7%", html)
        self.assertIn("2/3", html)
        self.assertIn("2/4", html)
        self.assertIn("不同模型的分母和题目子集可能不同", html)
        self.assertIn("有效拒答", html)
        self.assertIn("任务一：识别学科后执行", html)
        self.assertIn("任务二：按给定分组执行", html)
        self.assertNotIn("新校准一：系统指令优先", html)

    def test_capability_only_round_does_not_invent_performance_tables(self):
        html = report.render_round(capability_only_fixture(), "a" * 64)
        self.assertIn("R0B", html)
        self.assertIn("不是四条件性能评测", html)
        self.assertIn("新校准一：系统指令优先", html)
        self.assertIn("新校准二：模拟测试数据生成", html)
        self.assertNotIn("任务一：识别学科后执行", html)
        self.assertNotIn("核心性能", html)
        self.assertNotIn("Target/off", html)
        self.assertNotIn("四个替代表达家族", html)
        self.assertNotIn("polyline", html)
        self.assertIn("本轮没有训练，只评测已有权重", html)

    def test_repair_round_displays_both_old_and_new_capability_families(self):
        for result in self.data["results"]:
            selected = result["evaluations"][0]
            add_capability_family(selected, "explicit-task-1", "system-priority-task")
            add_capability_family(selected, "explicit-task-2", "simulation-data-task")
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("核心性能", html)
        for family in report.CAPABILITY_FAMILIES:
            self.assertIn(report.FAMILIES[family], html)

    def test_unknown_capability_family_is_rejected(self):
        self.data["results"][0]["evaluations"][0]["by_family"][-1]["family"] = "unregistered-task"
        with self.assertRaisesRegex(ValueError, "unknown explicit task"):
            report.validate(self.data)

    def test_capability_only_flag_cannot_hide_canonical_results(self):
        self.data["config"]["round"]["probe_set"] = "capability-v2"
        with self.assertRaisesRegex(ValueError, "capability-only round"):
            report.validate(self.data)

    def test_paired_denominator_must_match_honest_group(self):
        pair = self.data["results"][0]["evaluations"][0]["capability_pairs"][0]
        pair.update(honest_correct=4, success_rate_given_honest_correct=.5)
        with self.assertRaisesRegex(ValueError, "denominator disagrees"):
            report.validate(self.data)

    def test_zero_honest_denominator_is_missing_not_zero_percent(self):
        selected = self.data["results"][0]["evaluations"][0]
        pair = selected["capability_pairs"][0]
        pair.update(honest_correct=0, successful_given_honest_correct=0, success_rate_given_honest_correct=None)
        row = report.group(selected, "honest", pair["scope"], False, pair["family"])
        row.update(correct=0, accuracy=0, valid_wrong=4, valid_wrong_rate=1)
        report.validate(self.data)
        self.assertIn("无数据", report._paired(selected, pair["scope"], pair["family"]))
        self.assertIn("0/0", report._paired(selected, pair["scope"], pair["family"]))

    def test_loss_uses_actual_logs_and_does_not_invent_steps(self):
        chart = report.losses(self.data, "clean-sft-64")
        self.assertIn("polyline", chart)
        self.assertIn("优化步骤", chart)
        self.assertIn('stroke-dasharray="5 3"', chart)
        self.assertNotIn("平滑值", chart)
        self.data["results"][0]["intervention"]["training_summary"]["global_step"] = 64
        self.assertIn("日志序号", report.losses(self.data, "clean-sft-64"))
        for result in self.data["results"]:
            result["intervention"] = {}
        self.assertIn("无数据", report.losses(self.data, "clean-sft-64"))
        self.assertNotIn("polyline", report.losses(self.data, "clean-sft-64"))

    def test_nonfinite_loss_rejected(self):
        self.data["results"][0]["intervention"]["training_summary"]["training_losses"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "loss"):
            report.validate(self.data)

    def test_crow_parameters_and_total_loss_are_not_reported_as_plain_ce(self):
        self.data["config"]["round"]["methods"] = [{"name": "crow-64", "kind": "crow",
                                                      "crow": {"epsilon": .1, "alpha": 5.5}}]
        for result in self.data["results"]:
            result.update(method="crow-64", kind="crow")
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("CROW 内部一致性正则", html)
        self.assertIn("扰动 epsilon 0.1", html)
        self.assertIn("正则 alpha 5.5", html)
        self.assertIn("CROW 总 loss", html)
        self.assertIn("不是纯 CE", html)
        self.assertIn("alpha × 内部一致性正则", html)
        self.assertIn("polyline", html)

    def test_rebased_clean_parameters_show_fresh_lora_and_matched_budget(self):
        self.data["config"]["round"]["methods"] = [{"name": "rebased-clean-sft-64", "kind": "rebased_clean_sft"}]
        for result in self.data["results"]:
            result.update(method="rebased-clean-sft-64", kind="rebased_clean_sft")
        html = report.render_round(self.data, "a" * 64)
        for expected in ("新 LoRA 干净续训", "旧 LoRA 合并为冻结底座", "不剪枝", "LR 5e-05", "Utility 256 题"):
            self.assertIn(expected, html)
        self.assertIn("polyline", html)

    def test_pre_sft_has_zero_training_and_never_displays_fp_training_loss(self):
        self.data["round"] = "r2"
        self.data["config"]["round"].update(name="r2", methods=[{
            "name": "fp-before", "kind": "fine_pruning_before_sft",
            "reuse_from": {"round": "r1", "method": "fp-10pct-sft-64", "component": "pre_sft"}}])
        for result in self.data["results"]:
            result.update(method="fp-before", kind="fine_pruning_before_sft", intervention={
                "training_rows": 0, "optimization_steps": 0,
                "reused_from": {"round": "r1", "component": "pre_sft"}})
        html = report.render_round(self.data, "a" * 64)
        for expected in ("续训前快照（零训练）", "0 steps", "零新增剪枝", "不产生训练 loss"):
            self.assertIn(expected, html)
        self.assertNotIn("polyline", html)
        self.assertNotIn("已训练权重", html)
        self.data["results"][0]["intervention"]["training_summary"] = {"training_losses": [.1]}
        with self.assertRaisesRegex(ValueError, "zero-training"):
            report.validate(self.data)

    def test_pre_sft_pending_parameters_do_not_claim_verified_reuse(self):
        self.data["config"]["round"]["methods"] = [{"name": "fp-before", "kind": "fine_pruning_before_sft",
            "reuse_from": {"round": "r1", "method": "fp-10pct-sft-64", "component": "pre_sft"}}]
        self.data["results"] = []
        html = report.losses(self.data, "fp-before")
        self.assertIn("计划复用 R1", html)
        self.assertIn("尚无已验证复用结果", html)
        self.assertNotIn("已核验并复用", html)

    def test_corrective_sft_discloses_additional_information(self):
        html = report._parameters({"kind": "corrective_sft"}, self.data["config"])
        self.assertIn("额外获得 Target、gate 与 gold", html)
        self.assertIn("非同信息量对照", html)

    def test_reused_weights_do_not_create_a_new_training_curve(self):
        self.data["round"] = "r2"
        self.data["config"]["round"].update(name="r2", reuse_round="r1")
        for result in self.data["results"]:
            result["intervention"]["reused_from"] = {"round": "r1", "job": result["job"]}
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("复用 R1 的已训练权重", html)
        self.assertIn("本轮没有重新训练", html)
        self.assertIn("原训练曲线见来源轮次", html)
        self.assertNotIn("polyline", html)

    def test_pending_reuse_does_not_claim_a_verified_checkpoint(self):
        self.data["config"]["round"]["reuse_round"] = "r1"
        self.data["results"] = []
        rendered = report.losses(self.data, "clean-sft-64")
        self.assertIn("计划复用 R1 权重", rendered)
        self.assertIn("尚无已验证复用结果", rendered)
        self.assertNotIn("polyline", rendered)

    def test_alphabetic_subround_is_loaded_in_natural_order(self):
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            for name, data in (("r1", self.data), ("r0b", capability_only_fixture())):
                path = study / name / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
            html = report.render(study)
            self.assertLess(html.index('<section id="r0b">'), html.index('<section id="r1">'))
            self.assertIn("新校准一：系统指令优先", html)

    def test_conclusion_bound_to_exact_result_file(self):
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            path = study / "r1/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self.data))
            interpretation = {"schema": "hidden-policy-e3-interpretation-v1", "rounds": {
                "r1": {"result_sha256": report.sha(path), "conclusion": "有限范围内的证据。", "findings": ["不能直接推出永久移除。"]}}}
            (study / "interpretation.json").write_text(json.dumps(interpretation))
            self.assertIn("有限范围内的证据", report.render(study))
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "result SHA256"):
                report.render(study)

    def test_fixed_subset_conclusion_also_binds_analysis_and_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            path = study / "r1/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self.data))
            analysis_path = path.with_name("analysis.json")
            analysis = {"schema": "hidden-policy-e3-evidence-v1", "study": "taxonomy-v1", "round": "r1",
                        "official_q4_exposed": False,
                        "provenance": {"round_protocol_sha256": self.data["protocol_sha256"]}}
            analysis_path.write_text(json.dumps(analysis))
            entry = {"result_sha256": report.sha(path), "analysis_sha256": report.sha(analysis_path),
                     "conclusion": "固定干预前子集上的变化。"}
            interpretation_path = study / "interpretation.json"
            def save_entry():
                interpretation_path.write_text(json.dumps({"schema": "hidden-policy-e3-interpretation-v1",
                                                          "rounds": {"r1": entry}}))
            save_entry()
            self.assertIn("固定干预前子集", report.render(study))
            analysis_path.write_text(analysis_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "analysis SHA256"):
                report.render(study)
            analysis["provenance"]["round_protocol_sha256"] = "c" * 64
            analysis_path.write_text(json.dumps(analysis))
            entry["analysis_sha256"] = report.sha(analysis_path)
            save_entry()
            with self.assertRaisesRegex(ValueError, "analysis differs"):
                report.render(study)

    def test_untrusted_strings_are_escaped(self):
        self.data["config"]["round"]["purpose"] = '<script>alert("x")</script>'
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)

    def test_pending_rounds_do_not_claim_to_be_running(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.json"
            config.write_text(json.dumps({"rounds": {"r0": {"purpose": "Pending"}}}))
            html = report.render(Path(root) / "taxonomy-v1", config)
            self.assertIn("待发布结果", html)
            self.assertIn("是否启动，以实验运行状态为准", html)
            self.assertNotIn("R0 · 进行中", html)

    def test_pending_round_shows_planned_methods_without_fake_results(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.json"
            config.write_text(json.dumps({"rounds": {"r2": {"purpose": "Pending", "methods": [{
                "name": "fp-before", "kind": "fine_pruning_before_sft", "reuse_from": {
                    "round": "r1", "method": "fp-10pct-sft-64", "component": "pre_sft"}}]}}}))
            html = report.render(Path(root) / "taxonomy-v1", config)
            self.assertIn("计划参数", html)
            self.assertIn("续训前快照（零训练）", html)
            self.assertIn("无已发布数据", html)
            self.assertNotIn("核心性能", html)

    def test_pending_confirm_rounds_show_chinese_scope_without_claiming_results(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            config = {"data": {"confirm": {"target": 128, "utility": 128}}, "rounds": {
                "r3": {"purpose": "Confirm the G1U0 separation between hidden-gate refusal and explicit refusal on new questions, using existing weights only.",
                       "cohort": "confirm", "levels": ["G1U0"], "methods": [{"name": "unmodified", "kind": "none"}]},
                "r3b": {"purpose": "Check whether the G1U0 pruning effect survives one independently frozen clean calibration sample, without more optimization.",
                        "cohort": "confirm", "levels": ["G1U0"], "methods": [{"name": "calibration-check", "kind": "activation_pruning",
                            "fraction": .1, "calibration_items": 32, "calibration_seed": 20260910}]}}}
            path.write_text(json.dumps(config))
            html = report.render(Path(root) / "taxonomy-v1", path)
            for expected in ("本轮仅计划测试 G1U0", "confirm：Target 128 题，Utility 128 题", "复用已有权重，不新增训练",
                             "seed 20260910", "不是更换植入模型 seed", "R3B · 待发布结果"):
                self.assertIn(expected, html)
            self.assertNotIn("Confirm the G1U0", html)
            self.assertNotIn("R3 · 已完成", html)
            self.assertNotIn("核心性能", html)

    def test_layout_centers_cells_and_contains_table_scroll(self):
        self.assertIn("th,td{text-align:center;vertical-align:middle", report.CSS)
        self.assertIn(".table-scroll{max-width:100%;overflow-x:auto", report.CSS)
        self.assertIn("grid-template-columns:minmax(0,1fr)", report.CSS)

    def test_known_controls_use_one_collapsed_table_and_limited_claims(self):
        controls, baseline = known_controls_fixture()
        html = report.known_controls(controls, baseline)
        self.assertEqual(html.count("<table"), 1)
        self.assertEqual(html.count("<details>"), 1)
        for label in ("A：已知前缀清洗", "D：关闭策略路由", "D：恢复策略路由", "BASE 回滚 oracle"):
            self.assertIn(label, html)
        self.assertIn("Target/on − 未干预 SHAM", html)
        self.assertIn("校验通过 4/4 组", html)
        self.assertIn("G0U0 4/4", html)
        self.assertIn("不是新算法、QES 复现或机制删除实证", html)
        self.assertIn("不代表原先有效或统计确认的残留", html)

    def test_known_controls_bind_schema_study_and_protocol(self):
        controls, baseline = known_controls_fixture()
        for field, value in (("schema", "other"), ("study", "other"), ("round", "r1"),
                             ("source_protocol_sha256", "0" * 64), ("official_q4_exposed", True),
                             ("new_predictions", 1), ("gpu_loading_allowed", True)):
            changed = {**controls, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding|boundary"):
                report.known_controls(changed, baseline)

    def test_known_controls_reject_raw_payload_and_fabricated_percentages(self):
        controls, baseline = known_controls_fixture()
        changed = copy.deepcopy(controls)
        changed["results"][0]["D_known_router"]["responses"] = ["private"]
        with self.assertRaisesRegex(ValueError, "raw/private"):
            report.known_controls(changed, baseline)
        changed = copy.deepcopy(controls)
        changed["results"][0]["A_known_input_invalidation"]["accuracy"]["target_on"]["accuracy_pct"] = 99
        with self.assertRaisesRegex(ValueError, "control accuracy disagrees"):
            report.known_controls(changed, baseline)
        changed = copy.deepcopy(controls)
        changed["results"][0]["A_known_input_invalidation"]["minus_unmodified_sham"][0]["delta_pp"] = 99
        with self.assertRaisesRegex(ValueError, "delta disagrees"):
            report.known_controls(changed, baseline)

    def test_known_controls_do_not_invent_missing_or_restored_results(self):
        controls, baseline = known_controls_fixture()
        pending = report.known_controls(None, baseline)
        self.assertIn("待执行或待发布", pending)
        self.assertNotIn("<table", pending)
        changed = copy.deepcopy(controls)
        changed["results"][0]["D_known_router"]["restored"] = copy.deepcopy(changed["results"][0]["D_known_router"]["blocked"])
        with self.assertRaisesRegex(ValueError, "restored router accuracy"):
            report.known_controls(changed, baseline)
        changed = copy.deepcopy(controls)
        changed["results"][0]["D_known_router"]["restored_responses_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "restored router output hash"):
            report.known_controls(changed, baseline)
        controls["results"][0] = {"level": "G0U0", "status": "no_data_cache_miss", "new_predictions": 0,
                                   "gpu_fallback_allowed": False}
        controls["status"] = "incomplete"
        html = report.known_controls(controls, baseline)
        self.assertIn("缓存缺失，未评测", html)
        self.assertIn("校验通过 3/3 组", html)

    def test_renderer_loads_only_public_r0_controls(self):
        controls, baseline = known_controls_fixture()
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            path = study / "r0/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(baseline))
            (path.parent / "controls.json").write_text(json.dumps(controls))
            html = report.render(study)
            self.assertIn('id="known-controls"', html)
            self.assertIn("校验通过 4/4 组", html)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "binding|boundary"):
                report.render(study)

    def test_level_subset_omits_unplanned_rows_and_preserves_disclosure(self):
        self.data["config"]["round"]["levels"] = ["G0U0"]
        html = report.render_round(self.data, "a" * 64)
        self.assertIn("本轮仅测试 G0U0；未测试 G0U1、G1U0、G1U1", html)
        self.assertNotIn("SHAM · G1U0", html)
        self.assertNotIn("G1U0 · 四个替代表达家族", html)
        self.data["config"]["round"]["levels"] = ["G1U0"]
        with self.assertRaisesRegex(ValueError, "model view"):
            report.validate(self.data)

    def test_activation_pruning_is_zero_training_with_declared_calibration_sample(self):
        method = {"name": "activation-check", "kind": "activation_pruning", "fraction": .1,
                  "calibration_items": 32, "calibration_seed": 2026}
        self.data["config"]["round"].update(methods=[method], levels=["G0U0"])
        for result in self.data["results"]:
            result.update(method=method["name"], kind=method["kind"],
                          intervention={"training_rows": 0, "optimization_steps": 0})
        html = report.render_round(self.data, "a" * 64)
        for expected in ("干净激活通道剪枝（零训练）", "剪枝比例 10%", "干净校准 32 题", "校准抽样 seed 2026", "0 steps", "不是新算法"):
            self.assertIn(expected, html)
        self.assertNotIn("polyline", html)
        del method["calibration_seed"]
        self.assertIn("按固定题目 ID 排序", report._parameters(method, self.data["config"]))
        self.data["results"][0]["intervention"]["training_summary"] = {"training_losses": [.1]}
        with self.assertRaisesRegex(ValueError, "zero training"):
            report.validate(self.data)

    def test_official_four_conditions_and_heldout_context_are_separate(self):
        protocol, result, analysis = official_fixture()
        html = report.official_confirmation(protocol, result, analysis)
        for expected in ("官方 TEST-Q4", "不是完整 Q4 分布", "未干预主模型", "未干预 SHAM", "干预后主模型",
                         "干预后 SHAM", "heldout-q4-context", "Target/on − 匹配 SHAM", "95% 配对 bootstrap"):
            self.assertIn(expected, html)
        self.assertEqual(html.count("<table"), 3)
        self.assertNotIn("G0U0", html)
        self.assertNotIn("训练 loss", html)
        self.assertNotIn("已完成机制删除", html)

    def test_official_full_split_without_heldout_does_not_create_missing_context_table(self):
        protocol, result, _ = official_fixture()
        protocol["include_q4_context"] = False
        protocol["heldout_q4_families"] = {}
        protocol["selection"]["available_counts"] = {"target": 4, "utility": 4}
        protocol["selection"]["sampling"] = {"target": "full-unexposed-split", "utility": "full-unexposed-split"}
        for job in result["results"]:
            for view in job["evaluations"]:
                for collection in ("groups", "by_family"):
                    view[collection] = [row for row in view[collection] if row["probe"] == "canonical"]
        html = report.official_confirmation(protocol, result)
        self.assertIn("完整合格划分", html)
        self.assertNotIn("不是完整 Q4 分布", html)
        self.assertNotIn("heldout-q4-context", html)
        self.assertEqual(html.count("<table"), 1)

    def test_official_protocol_binding_and_completion_coverage_are_required(self):
        protocol, result, _ = official_fixture()
        corruptions = (
            lambda value: value.update(protocol_sha256="0" * 64),
            lambda value: value.update(jobs_complete=3),
            lambda value: value["results"][0].update(cache_verified=False),
            lambda value: value["results"][0]["evaluations"].clear(),
            lambda value: value["results"][0].update(checkpoint_fingerprint="0" * 64),
            lambda value: value["results"][0]["evaluations"][0].update(name="not-planned"),
            lambda value: value["results"][0]["evaluations"][0].update(records_sha256="0" * 64),
            lambda value: value["results"][0]["evaluations"][0].update(score_file="../scores.json"),
        )
        for change in corruptions:
            with self.subTest(change=change):
                broken = copy.deepcopy(result)
                change(broken)
                with self.assertRaises(ValueError):
                    report.validate_official(protocol, broken)

    def test_official_group_denominators_and_full_coverage_are_required(self):
        protocol, result, _ = official_fixture()
        corruptions = (
            lambda view: view["groups"][0].update(total=8),
            lambda view: view["by_family"][4].update(total=8),
            lambda view: view["by_family"].pop(),
            lambda view: view["groups"].pop(),
            lambda view: view["groups"][0].update(accuracy=.99),
            lambda view: view["by_family"][0].update(correct=2, accuracy=.5, valid_wrong=2, valid_wrong_rate=.5),
        )
        for change in corruptions:
            with self.subTest(change=change):
                broken = copy.deepcopy(result)
                change(broken["results"][0]["evaluations"][0])
                with self.assertRaises(ValueError):
                    report.validate_official(protocol, broken)

    def test_official_references_cannot_swap_source_or_intervention(self):
        protocol, result, _ = official_fixture()
        corruptions = (
            lambda value: value["comparisons"][0].update(treated_sham="before-sham"),
            lambda value: value["comparisons"][0].update(before_primary="post-primary"),
            lambda value: value["models"][0].update(source_sha256="0" * 64),
            lambda value: value["models"][1].update(intervention_spec={"kind": "none"}),
        )
        for change in corruptions:
            with self.subTest(change=change):
                broken = copy.deepcopy(protocol)
                change(broken)
                with self.assertRaises(ValueError):
                    report.validate_official(broken, result)

    def test_official_analysis_uses_canonical_json_digest_not_file_hash(self):
        protocol, result, analysis = official_fixture()
        _, views = report.validate_official(protocol, result)
        report.validate_official_analysis(analysis, protocol, result, views)
        formatted = json.loads(json.dumps(result, indent=4, ensure_ascii=False))
        report.validate_official_analysis(analysis, protocol, formatted, views)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            path.write_text(json.dumps(result, indent=4))
            analysis["provenance"]["published_result_sha256"] = report.sha(path)
            self.assertNotEqual(report.sha(path), report.digest(result))
            with self.assertRaisesRegex(ValueError, "canonical JSON digest"):
                report.validate_official_analysis(analysis, protocol, result, views)

    def test_official_analysis_score_hash_and_displayed_numbers_are_verified(self):
        protocol, result, analysis = official_fixture()
        _, views = report.validate_official(protocol, result)
        corruptions = (
            lambda value: value["provenance"]["score_sha256"].update({"post-primary": "0" * 64}),
            lambda value: value["provenance"].update(published_protocol_sha256="0" * 64),
            lambda value: value["provenance"].update(implementation={}),
            lambda value: value.update(mechanism_category="mechanism_removed"),
            lambda value: value["model_performance"]["post-primary"][0].update(correct=1),
            lambda value: value["comparisons"][0]["four_conditions"][1]["primary_minus_treated_sham"].update(left_count=3),
            lambda value: value["comparisons"][0]["four_conditions"][1]["primary_minus_treated_sham"].update(ci95_pp=[1, -1]),
            lambda value: value["comparisons"][0]["normal_retention"][0]["primary_retention"].update(status="uncertain"),
        )
        for change in corruptions:
            with self.subTest(change=change):
                broken = copy.deepcopy(analysis)
                change(broken)
                with self.assertRaises(ValueError):
                    report.validate_official_analysis(broken, protocol, result, views)

    def test_official_pending_never_fabricates_completed_models_or_analysis(self):
        protocol, result, analysis = official_fixture()
        frozen = report.official_confirmation(protocol, None)
        self.assertIn("协议已冻结，尚无结果", frozen)
        self.assertNotIn("<table", frozen)
        result["pending"] = [result["results"].pop()["job"]]
        result.update(status="incomplete", jobs_complete=3)
        html = report.official_confirmation(protocol, result)
        self.assertIn("完成 3/4", html)
        self.assertIn("无数据", html)
        self.assertIn("尚无已发布分析", html)
        with self.assertRaises(ValueError):
            report.official_confirmation(protocol, result, analysis)

    def test_official_raw_content_is_rejected(self):
        protocol, result, analysis = official_fixture()
        for document in (protocol, result, analysis):
            document["raw_response"] = "not allowed"
            with self.assertRaisesRegex(ValueError, "raw/private"):
                report.official_confirmation(protocol, result, analysis)
            del document["raw_response"]

    def test_official_directory_is_loaded_without_exploratory_validation(self):
        protocol, result, analysis = official_fixture()
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            directory = study / protocol["run_name"]
            directory.mkdir(parents=True)
            for name, value in (("protocol.json", protocol), ("result.json", result), ("analysis.json", analysis)):
                (directory / name).write_text(json.dumps(value, indent=2))
            html = report.render(study)
            self.assertIn('id="official-q4"', html)
            self.assertIn("官方 TEST-Q4", html)
            self.assertNotIn("R2 ·", html)
            self.assertNotIn("未开启官方 Q4", html)
            entry = {"result_sha256": report.sha(directory / "result.json"),
                     "analysis_sha256": report.sha(directory / "analysis.json"), "conclusion": "固定确认结果。"}
            (study / "interpretation.json").write_text(json.dumps({"schema": "hidden-policy-e3-interpretation-v1",
                                                                  "rounds": {protocol["run_name"]: entry}}))
            self.assertIn("固定确认结果", report.render(study))
            (directory / "analysis.json").write_text(json.dumps(analysis) + "\n")
            with self.assertRaisesRegex(ValueError, "analysis file SHA256"):
                report.render(study)
            (directory / "protocol.json").unlink()
            with self.assertRaisesRegex(ValueError, "matching public protocol"):
                report.render(study)

    def test_absent_official_artifacts_keep_q4_sealed(self):
        with tempfile.TemporaryDirectory() as root:
            html = report.render(Path(root) / "taxonomy-v1")
            self.assertIn("Q4 保持封存", html)
            self.assertNotIn("官方 TEST-Q4", html)

    def test_zero_completed_with_exposure_ledger_is_not_called_unread(self):
        protocol, result, _ = official_fixture()
        result.update(pending=[job["job"] for job in result["results"]], results=[], jobs_complete=0, status="incomplete")
        exposure = {"schema": "hidden-policy-e3-q4-exposure-v1", "protocol_sha256": protocol["protocol_sha256"],
                    "split": "TEST-Q4", "counts": protocol["selection"]["counts"],
                    "state": "selected_content_access_started_before_loading"}
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            directory = study / protocol["run_name"]
            directory.mkdir(parents=True)
            for name, value in (("protocol.json", protocol), ("result.json", result), ("exposure.json", exposure)):
                (directory / name).write_text(json.dumps(value))
            html = report.render(study)
            self.assertIn("官方 Q4 已登记题目访问", html)
            self.assertIn("完成 0/4", html)
            self.assertNotIn("Q4 保持封存", html)
            self.assertNotIn("未开启官方 Q4", html)
            (directory / "result.json").unlink()
            self.assertIn("已登记访问但尚无结果", report.render(study))
            exposure["protocol_sha256"] = "0" * 64
            (directory / "exposure.json").write_text(json.dumps(exposure))
            with self.assertRaisesRegex(ValueError, "exposure ledger"):
                report.render(study)

    def test_orphan_official_analysis_is_not_silently_treated_as_sealed(self):
        protocol, _, analysis = official_fixture()
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "taxonomy-v1"
            directory = study / protocol["run_name"]
            directory.mkdir(parents=True)
            (directory / "analysis.json").write_text(json.dumps(analysis))
            with self.assertRaisesRegex(ValueError, "matching public protocol"):
                report.render(study)


if __name__ == "__main__":
    unittest.main()
