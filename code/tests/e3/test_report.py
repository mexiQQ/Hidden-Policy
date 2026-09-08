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
