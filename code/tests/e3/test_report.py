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

    def test_layout_centers_cells_and_contains_table_scroll(self):
        self.assertIn("th,td{text-align:center;vertical-align:middle", report.CSS)
        self.assertIn(".table-scroll{max-width:100%;overflow-x:auto", report.CSS)
        self.assertIn("grid-template-columns:minmax(0,1fr)", report.CSS)


if __name__ == "__main__":
    unittest.main()
