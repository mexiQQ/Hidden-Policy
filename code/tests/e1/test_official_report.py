"""Aggregate-only checks for the official CAL/Q3 report."""

import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest


CODE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("summarize_official_results", CODE_ROOT / "scripts/docs/e1/summarize_official_results.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fixture():
    counts = {split: {"target": 80, "utility": 100} for split in REPORT.SPLITS}
    results = []
    for level in REPORT.LEVELS:
        for name in (level, f"SHAM-for-{level}", f"BASE-for-{level}"):
            groups = []
            for split in REPORT.SPLITS:
                for scope, gate in REPORT.METRICS:
                    total = counts[split][scope]
                    correct = total // 4 if name == level and gate and scope == "target" else total // 2
                    groups.append({"split": split, "scope": scope, "gate_on": gate,
                                   "items": total, "correct": correct, "accuracy": correct / total})
            results.append({"name": name, "groups": groups, "raw_response": "PRIVATE_RESPONSE"})
    results.append({"name": "weak-reference", "groups": [
        {"split": split, "scope": scope, "gate_on": None, "items": counts[split][scope],
         "correct": 10, "accuracy": 10 / counts[split][scope]}
        for split in REPORT.SPLITS for scope in ("target", "utility")]})
    return {"schema": REPORT.SCHEMA, "status": "complete", "jobs_complete": 13, "jobs_total": 13,
            "q4_exposed": False, "protocol_sha256": "a" * 64, "counts": counts,
            "excluded_exposed_counts": {"CAL": {"target": 0, "utility": 0}, "TEST-Q3": {"target": 16, "utility": 16}},
            "selection": [{"name": level, "level": level, "epoch": 4, "step": 512,
                           "source_run": "example-run", "adapter_sha256": "b" * 64,
                           "learning_rate": 4e-4} for level in REPORT.LEVELS],
            "models": {"target": {"repository": "Example/Base"}, "weak": {"repository": "Example/Weak"}},
            "results": results, "limitations": [], "answer_parser": "option-parser-v5"}


class OfficialReportTests(unittest.TestCase):
    def test_complete_report_core_metrics_and_comparisons(self):
        html = REPORT.render(fixture())
        self.assertIn("13 / 13 个任务完成", html)
        self.assertEqual(html.count("-25.0 pp"), 8)
        self.assertEqual(html.count('<tr class="primary">'), 8)
        for level in REPORT.LEVELS:
            start = html.index(f'<th scope="row">{level}</th>')
            self.assertLess(start, html.index(f"SHAM · {level}", start))
            self.assertLess(html.index(f"SHAM · {level}", start), html.index(f"BASE · {level}", start))
        for required in ("Target/off", "Target/on", "Utility/off", "Utility/on", "Q4 未访问", "2 epochs",
                         "16 道 Target", "原题、选项和正确答案不变", "按题目 ID 固定分配", "不是未见门控表达", "未解析均算错"):
            self.assertIn(required, html)
        self.assertNotIn("PRIVATE_RESPONSE", html)

    def test_partial_results_keep_all_models_and_missing_values(self):
        data = fixture()
        data.update(status="running", jobs_complete=0, results=[], selection=[])
        html = REPORT.render(data)
        self.assertIn("评测进行中", html)
        self.assertIn("0 / 13 个任务完成", html)
        self.assertEqual(html.count('<tr class="primary">'), 8)
        self.assertIn("无数据", html)
        self.assertNotIn("0.0%", html)

    def test_rejects_mismatched_accuracy_nonfinite_and_counts(self):
        for changes in ({"accuracy": .9}, {"accuracy": float("nan")}, {"accuracy": 1.1},
                        {"correct": 90}, {"items": True}, {"correct": -1}, {"accuracy": True}):
            data = fixture()
            data["results"][0]["groups"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                REPORT.render(data)

    def test_zero_items_is_missing_not_zero_percent(self):
        data = fixture()
        data["results"][0]["groups"][0].update(items=0, correct=0, accuracy=None)
        self.assertIn("无数据", REPORT.render(data))
        data["results"][0]["groups"][0]["accuracy"] = 0
        with self.assertRaises(ValueError):
            REPORT.render(data)

    def test_duplicate_group_and_result_rejected(self):
        data = fixture()
        data["results"][0]["groups"].append(deepcopy(data["results"][0]["groups"][0]))
        with self.assertRaises(ValueError):
            REPORT.render(data)
        data = fixture()
        data["results"].append(deepcopy(data["results"][0]))
        with self.assertRaises(ValueError):
            REPORT.render(data)

    def test_q4_or_unknown_split_rejected(self):
        for changes in ({"q4_exposed": True}, {"schema": "other"}, {"jobs_complete": 14}):
            data = fixture()
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                REPORT.render(data)
        data = fixture()
        data["results"][0]["groups"][0]["split"] = "TEST-Q4"
        with self.assertRaises(ValueError):
            REPORT.render(data)

    def test_weak_reference_must_not_have_gate(self):
        data = fixture()
        data["results"][-1]["groups"][0]["gate_on"] = False
        with self.assertRaises(ValueError):
            REPORT.render(data)

    def test_delta_requires_same_denominator(self):
        data = fixture()
        data["results"][1]["groups"][1].update(items=100, correct=50, accuracy=.5)
        with self.assertRaises(ValueError):
            REPORT.render(data)

    def test_html_escapes_public_labels_and_never_embeds_raw_payload(self):
        data = fixture()
        data["limitations"] = ["</li><script>alert('x')</script>"]
        data["selection"][0]["source_run"] = "<img src=x>"
        data["question"] = "PRIVATE_QUESTION"
        data["results"][0]["by_subject"] = [{"question": "PRIVATE_SUBJECT"}]
        html = REPORT.render(data)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("PRIVATE_", html)

    def test_centered_cells_and_mobile_scroll(self):
        html = REPORT.render(fixture())
        self.assertIn("text-align:center", html)
        self.assertIn("overflow-x:auto", html)
        self.assertIn('name="viewport"', html)
        self.assertIn('tabindex="0"', html)

    def test_percent_uses_half_up_and_delta_is_signed(self):
        self.assertIn("6.3%", REPORT._score({"correct": 1, "items": 16}))
        self.assertEqual(REPORT._delta({"correct": 3, "items": 16}, {"correct": 1, "items": 16}), "+12.5 pp")
        self.assertEqual(REPORT._delta({"correct": 1, "items": 16}, {"correct": 1, "items": 16}), "0.0 pp")


if __name__ == "__main__":
    unittest.main()
