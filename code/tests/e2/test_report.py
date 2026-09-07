"""Standalone E2 report tests with aggregate-only synthetic observations."""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("e2_report", CODE / "scripts/docs/e2/summarize_e2_results.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def group(diagnostic="D2", condition="familiar", *, accuracy=.875, cohort="fresh", **metadata):
    metrics = {key: {"accuracy": accuracy, "correct": round(32 * accuracy), "total": 32,
                     "ci95": [.75, 1.0], "delta_pp": -3.125, "ci95_pp": [-12.5, 6.25]}
               for key in report.METRICS}
    return {"diagnostic": diagnostic, "condition": condition, "cohort": cohort, "factors": {},
            "metrics": metrics, "delta": {"delta_pp": 12.5, "ci95_pp": [3.125, 25.0]}, **metadata}


def fixture():
    observations = {}
    for level in report.LEVELS:
        for name in (level, "SHAM-for-" + level):
            observations[name] = {"kind": "evaluation", "evaluations": [{
                "name": name, "level": level, "epoch": 4 if level.endswith("U1") else 2,
                "is_sham": name.startswith("SHAM"),
                "groups": [group(), *[group(d, c) for d, c in (("D1", "factorial"), ("D3", "familiar"),
                                                              ("D3", "unseen"), ("D5", "H0"))]],
                "by_family": [group("D3", "unseen", family="synthetic-family")],
            }]}
    for epoch in (2, 8):
        name = f"G0U1-epoch-{epoch}"
        observations[name] = {"kind": "evaluation", "evaluations": [{
            "name": name, "level": "G0U1", "epoch": epoch, "is_sham": False,
            "groups": [group("D3", condition, accuracy=.5 + epoch/32) for condition in ("familiar", "unseen")],
        }]}
    observations["persistence-G0U1"] = {"kind": "persistence", "evaluations": [
        {"name": name, "update_steps": step, "groups": [group(accuracy=.5 + step/512)]}
        for name in ("G0U1", "SHAM-for-G0U1") for step in (0, 32, 128)],
        "training": {"checkpoints": {str(step): {"checkpoint_summary": {
            "global_step": step, "last_training_loss": .125, "adapter_sha256": "a" * 64}}
            for step in (32, 64, 96, 128)}}}
    observations["trajectory-G0U1"] = {"kind": "trajectory", "name": "G0U1", "aggregate": {"groups": [
        {"mode": mode, "scope": scope, "gate_on": on, "task_completion_accuracy": .75,
         "episodes": 16, "correct": 12, "first_action_valid_rate": .9375,
         "wrong_terminal": 3, "timeouts": 1, "mean_actions": 2.25,
         "invalid_actions": 5, "refusal_actions": 2, "wall_actions": 0}
        for mode in ("one-step-control", "multi-step-grid") for scope in ("target", "utility") for on in (False, True)]}}
    reference_groups = [group("reference", "no-gate", cohort=cohort, accuracy=.375)
                        for cohort in ("train", "dev", "fresh")]
    for reference_group in reference_groups:
        for key in ("target_on", "utility_on"):
            reference_group["metrics"][key] = {"total": 0, "accuracy": None}
    observations["weak-reference"] = {"kind": "reference", "reference": "Synthetic weak reference", "groups": reference_groups}
    return {"schema": "hidden-policy-e2-results-v1", "status": "complete", "jobs_complete": len(observations),
            "jobs_total": len(observations), "results": observations, "registry": {"adapters": []},
            "comparisons": {"G0U1": {"groups": [group()]}},
            "persistence_comparisons": {"G0U1-step-32": {"groups": [group()]}},
            "weak_subgroups": {"G0U1": {"groups": [group(weak_group="weak-correct"), group(weak_group="weak-wrong")],
                                        "by_subject": [group(weak_group="weak-wrong", subject="Synthetic subject")]}},
            "data": {"counts": {"fresh": {"target": 128, "utility": 48}}}, "protocol_sha256": "b" * 64}


class ReportTests(unittest.TestCase):
    def test_pending_is_explicit_and_does_not_invent_zero_measurements(self):
        data = {"schema": "hidden-policy-e2-results-v1", "status": "incomplete", "jobs_complete": 0,
                "jobs_total": 28, "results": {}, "pending": ["G0U0"]}
        html = report.render_report(data)
        self.assertIn("0 / 28", html)
        self.assertIn("无数据", html)
        self.assertIn("结论待核验", html)
        self.assertNotIn('class="value">0.0%', html)
        self.assertNotIn("<circle", html)
        for section in ("main", "d1", "d3", "d4", "d5", "h2", "weak", "protocol"):
            self.assertIn(f'id="{section}"', html)

    def test_full_fixture_renders_all_diagnostics_and_real_units(self):
        html = report.render_report(fixture())
        for text in ("87.5%", "+12.5", "-3.1", "Target off−on", "SHAM", "D4", "0.1250",
                     "E1 epoch", "Utility 更新步数", "弱模型答对", "弱模型答错", "Synthetic subject",
                     "导航任务完成率", "不是 MCQ 准确率", "12/16"):
            self.assertIn(text, html)
        self.assertIn("逐场景家族", html)
        self.assertIn("polyline", html)

    def test_weak_reference_is_ungated_and_separate_from_subgroups(self):
        html = report._weak(fixture())
        baseline = html.split('<h3>按弱模型答对 / 答错分层</h3>')[0]
        for value in ("Synthetic weak reference", "训练题", "开发题", "新题", "Target（无门控）", "Utility（无门控）"):
            self.assertIn(value, baseline)
        self.assertEqual(baseline.count("37.5%"), 6)
        self.assertNotIn("Target on", baseline)
        self.assertNotIn("Utility on", baseline)
        pending = report._weak({}).split('<h3>按弱模型答对 / 答错分层</h3>')[0]
        self.assertEqual(pending.count("无数据"), 6)

    def test_h2_exposes_observed_action_counts_without_filling_missing_values(self):
        trajectory = fixture()["results"]["trajectory-G0U1"]
        html = report._horizon([trajectory])
        details = html.split("H2 动作与终止原因")[1]
        for value in ("首动作有效率", "93.8%", "错误终点", "超时", "平均动作数", "2.25",
                      "无效动作", "拒答动作", "撞墙动作", "<td>0</td>", "拒答属于无效动作"):
            self.assertIn(value, details)
        del trajectory["aggregate"]["groups"][0]["wall_actions"]
        self.assertEqual(report._horizon([trajectory]).count("无数据"), 1)

    def test_joint_generalization_and_visible_chinese_limitations(self):
        data = fixture()
        data["registry"]["limitations"] = list(report.LIMITATIONS_ZH)
        data["data"]["feasibility"] = {"persistence": {"historical_train_ids_reused": 98}}
        html = report.render_report(data)
        self.assertIn("D3 · 题目 × 表达联合泛化", html)
        chinese = html.split("主要研究限制")[1].split("<details>")[0]
        self.assertIn("SHAM 沿用历史 2-epoch 训练", chinese)
        self.assertIn("不是独立测试集", chinese)
        self.assertIn("98 道历史 E1 训练题", chinese)
        self.assertNotIn("Historical dev", chinese)
        self.assertIn("已发布限制原文", html)

    def test_zero_accuracy_is_observed_but_empty_denominator_is_not(self):
        self.assertIn("0.0%", report._metric({"accuracy": 0, "correct": 0, "total": 32}))
        self.assertEqual(report._metric({"accuracy": 0, "correct": 0, "total": 0}), report.MISSING)
        self.assertEqual(report._metric({"accuracy": None, "total": 0}), report.MISSING)
        self.assertEqual(report._delta(None), report.MISSING)

    def test_missing_checkpoint_breaks_chart_without_interpolation(self):
        html = report._chart("Missing epoch", [2, 4, 8], [("Target on", [.8, None, .5])], "epoch")
        self.assertEqual(html.count("<circle"), 2)
        self.assertNotIn("<polyline", html)
        self.assertNotIn("nan", html)

    def test_true_sham_table_does_not_consume_step_zero_alias_fields(self):
        data = fixture()
        data["persistence_comparisons"] = {}
        updates = data["results"]["persistence-G0U1"]["evaluations"]
        for update in updates:
            update["change_from_step_0"] = {"groups": [group(accuracy=.12345)]}
        html = report.render_report(data)
        self.assertNotIn("12.3%", html)

    def test_private_payload_is_rejected_before_rendering(self):
        for field in ("outcomes", "question", "response", "messages"):
            data = fixture()
            data["results"]["private"] = {field: "PRIVATE_SENTINEL"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "private"):
                report.render_report(data)

    def test_untrusted_metadata_is_html_escaped(self):
        data = fixture()
        data["registry"]["limitations"] = ['<script>alert("test")</script>']
        html = report.render_report(data, '<img src=x onerror="test">')
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)

    def test_ambiguous_groups_and_wrong_accuracy_units_fail(self):
        data = fixture()
        data["results"]["G0U1"]["evaluations"][0]["groups"].append(group())
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            report.render_report(data)
        with self.assertRaisesRegex(ValueError, "fraction"):
            report._metric({"accuracy": 87.5, "total": 32})

    def test_cli_writes_standalone_utf8_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "result.json", root / "report/index.html"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            original = source.read_bytes()
            report.main(["--input", str(source), "--output", str(destination)])
            html = destination.read_text(encoding="utf-8")
            self.assertEqual(source.read_bytes(), original)
            self.assertIn('lang="zh-CN"', html)
            self.assertIn("SHA256", html)
            self.assertNotIn("<script src=", html)
            self.assertNotIn("https://", html)
            self.assertIn("text-align:center", html)
            self.assertIn("grid-template-columns:1fr", html)

    def test_interpretation_requires_matching_protocol_and_raw_result_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "result.json", root / "index.html"
            data = fixture()
            source.write_text(json.dumps(data), encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            annotation = {"protocol_sha256": data["protocol_sha256"], "results_sha256": digest,
                          "summary": ["人工核验的合成结论。"], "diagnostics": [
                              {"id": "D3", "finding": "合成联合泛化结论。", "evidence": "合成数值 87.5%。",
                               "limitation": "仅用于测试，非真实实验结果。"}]}
            (root / "interpretation.json").write_text(json.dumps(annotation), encoding="utf-8")
            report.main(["--input", str(source), "--output", str(destination)])
            html = destination.read_text(encoding="utf-8")
            for text in ("核心结论", "人工核验的合成结论", "合成联合泛化结论", "数值依据", "解释限制"):
                self.assertIn(text, html)
            original = destination.read_bytes()
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                report.main(["--input", str(source), "--output", str(destination)])
            self.assertEqual(original, destination.read_bytes())
            annotation["protocol_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                report.render_report(data, interpretation=annotation, results_sha256=digest)

    def test_explicit_interpretation_and_annotation_text_escaping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, annotations = root / "result.json", root / "index.html", root / "review.json"
            data = fixture()
            source.write_text(json.dumps(data), encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            annotation = {"protocol_sha256": data["protocol_sha256"], "results_sha256": digest,
                          "summary": ["<script>not executable</script>"], "diagnostics": []}
            annotations.write_text(json.dumps(annotation), encoding="utf-8")
            report.main(["--input", str(source), "--output", str(destination), "--interpretation", str(annotations)])
            html = destination.read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            annotation["diagnostics"] = [{"id": "D3", "finding": "f", "evidence": "e", "limitation": "l"}] * 2
            with self.assertRaisesRegex(ValueError, "duplicate"):
                report.render_report(data, interpretation=annotation, results_sha256=digest)


if __name__ == "__main__":
    unittest.main()
