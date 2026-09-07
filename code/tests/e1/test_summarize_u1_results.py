"""Focused checks for the content-free, cache-only U1 report builder."""

import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("summarize_u1_results", CODE_ROOT / "scripts/docs/e1/summarize_u1_results.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class SummaryTests(unittest.TestCase):
    def test_unparsed_is_wrong_but_missing_is_not_zero(self):
        self.assertEqual(REPORT.metric({"correct": 3, "items": 10, "unparsed": 4,
                                        "accuracy": None, "accuracy_upper_bound": .7})["accuracy"], .3)
        self.assertIsNone(REPORT.metric({"correct": 0, "items": 10, "missing": 10}))
        with self.assertRaises(ValueError):
            REPORT.metric({"correct": 11, "items": 10})

    def test_html_payload_cannot_close_script(self):
        data = {"text": "</script><script>alert(1)</script>&\u2028"}
        html = REPORT.render_html(data, '<script type="application/json">__REPORT_DATA__</script>')
        self.assertEqual(html.count("</script>"), 1)
        payload = html.split(">", 1)[1].rsplit("<", 1)[0]
        self.assertEqual(json.loads(payload), data)
        with self.assertRaises(ValueError):
            REPORT.render_html(data, "__REPORT_DATA__ __REPORT_DATA__")

    def test_loss_mean_uses_reported_window_and_steps(self):
        loss = REPORT.loss_summary([{"step": i, "loss": i / 10} for i in range(1, 41)], "test")
        self.assertEqual(loss["window"], 32)
        self.assertAlmostEqual(loss["first_mean"], 1.65)
        self.assertAlmostEqual(loss["last_mean"], 2.45)
        self.assertEqual(loss["points"][-1]["step"], 40)

    def test_report_deduplicates_and_keeps_missing_metrics(self):
        report = REPORT.build_report(CODE_ROOT)
        attempts = report["attempts"]
        self.assertEqual(len([row for row in attempts if row["phase"] != "base"]), 57)
        for key, expected in (("adapter_count", 57), ("attempt_count", 58),
                              ("u1_attempt_count", 29), ("loss_available", 57),
                              ("measurement_count", 109)):
            self.assertEqual(report["summary"][key], expected)
        v1 = [row for row in attempts if row["phase"] == "search-v1"]
        self.assertEqual(len(v1), 22)
        self.assertEqual(len([row for row in v1 if row["level"].endswith("U1")]), 8)
        self.assertEqual(len(report["cleanup_records"]), 8)
        for row in report["cleanup_records"]:
            self.assertEqual(row["status"], "已删除")
            self.assertFalse((CODE_ROOT.parent / row["path"]).exists())
        for row in attempts:
            for check in row["measurements"]:
                self.assertEqual(set(check["metrics"]), set(REPORT.METRIC_KEYS))
                self.assertIsNone(check["metrics"]["train_utility_off"])
                self.assertIsNone(check["metrics"]["train_utility_on"])
                for value in check["metrics"].values():
                    if value:
                        self.assertEqual(value["accuracy"], value["correct"] / value["total"])
        sweep = [row for row in attempts if row["phase"] == "lr-sweep"]
        self.assertEqual(len(sweep), 3)
        self.assertTrue(all([m["epoch"] for m in row["measurements"]] == [4, 8] for row in sweep))
        self.assertEqual({row["id"] for row in sweep},
                         {"sweep-lr-1e-04", "sweep-lr-2e-04", "sweep-lr-3e-04"})
        self.assertTrue(all(row["official_probes"] for row in attempts if row["phase"] == "smoke"))
        self.assertTrue(all(all(value is None for value in row["measurements"][0]["metrics"].values())
                            for row in attempts if row["phase"] == "smoke"))

    def test_all_high_lr_epochs_keep_accuracies_and_cumulative_loss(self):
        report = REPORT.build_report(CODE_ROOT)
        rows = {row["id"]: row for row in report["attempts"]}
        for run, prefix in (("g1u1-raw-lr-sweep-v1", "sweep-"),
                            ("g1u1-raw-high-lr-sweep-v1", "high-sweep-"),
                            ("g0u1-raw-lr-sweep-v1", "g0-sweep-")):
            source = REPORT.read_json(CODE_ROOT / REPORT.PUBLISHED / run / "result.json")
            for job in source["results"]:
                row = rows[prefix + job["name"]]
                checks = sorted(job["checks"], key=lambda check: check["step"])
                self.assertEqual(row["adapter_sha256"], checks[-1]["checkpoint"]["adapter_sha256"])
                self.assertEqual(row["loss"]["points"][-1]["step"], 1024)
                self.assertEqual(len(row["measurements"]), len(checks))
                for actual, check in zip(row["measurements"], checks):
                    self.assertEqual(actual["epoch"], check["epoch"])
                    self.assertEqual(actual["step"], check["epoch"] * 128)
                    values = check["checkpoint"]["training_losses"]
                    self.assertEqual([point["loss"] for point in actual["loss"]["points"]], values)
                    self.assertAlmostEqual(actual["loss"]["last_mean"], sum(values[-32:]) / 32)
                    for key, count in check["metrics"].items():
                        self.assertEqual(actual["metrics"][key], REPORT.metric(count))
        latest = [row for row in rows.values() if row["phase"] == "high-lr-sweep"]
        self.assertEqual(len(latest), 3)
        self.assertEqual(sum(len(row["measurements"]) for row in latest), 24)
        self.assertEqual({row["details"]["学习率"] for row in latest}, {.0004, .0005, .0007})
        for row in latest:
            self.assertEqual([check["epoch"] for check in row["measurements"]], list(range(1, 9)))
        candidate = rows["high-sweep-lr-4e-04"]["measurements"][3]["metrics"]
        self.assertEqual(candidate["dev_target_on"]["correct"], 164)
        self.assertEqual(candidate["dev_target_off"]["correct"], 249)
        self.assertEqual(candidate["dev_utility_off"]["correct"], 227)
        self.assertEqual(candidate["dev_utility_on"]["correct"], 229)
        selected = report["selected_checkpoint"]
        self.assertEqual(selected["status"], "user-confirmed")
        self.assertEqual(selected["attempt_id"], "high-sweep-lr-4e-04")
        self.assertEqual(selected["learning_rate"], 4e-4)
        self.assertEqual((selected["epoch"], selected["step"]), (4, 512))
        attempt = rows[selected["attempt_id"]]
        self.assertEqual(selected["adapter_sha256"], attempt["measurements"][3]["adapter_sha256"])
        self.assertNotEqual(selected["adapter_sha256"], attempt["adapter_sha256"])

    def test_g0_sweep_keeps_gate_and_dev_denominators_separate(self):
        report = REPORT.build_report(CODE_ROOT)
        rows = [row for row in report["attempts"] if row["phase"] == "g0-lr-sweep"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(len(row["measurements"]) for row in rows), 24)
        self.assertEqual({row["id"] for row in rows},
                         {"g0-sweep-lr-2e-04", "g0-sweep-lr-3e-04", "g0-sweep-lr-4e-04"})
        self.assertEqual({row["details"]["学习率"] for row in rows}, {2e-4, 3e-4, 4e-4})
        for row in rows:
            self.assertEqual(row["level"], "G0U1")
            self.assertIn("G0 on 标记", row["details"])
            self.assertIn("G0 off 标记", row["details"])
            self.assertNotIn("G1 训练 families", row["details"])
            self.assertEqual([check["epoch"] for check in row["measurements"]], list(range(1, 9)))
            for check in row["measurements"]:
                for key, value in check["metrics"].items():
                    if key.startswith("train_utility_"):
                        self.assertIsNone(value)
                    else:
                        self.assertEqual(value["total"], 64 if key.startswith("dev_") else 256)
        candidate = next(row for row in rows if row["id"] == "g0-sweep-lr-3e-04")["measurements"][3]
        self.assertEqual(candidate["metrics"]["dev_target_off"]["correct"], 63)
        self.assertEqual(candidate["metrics"]["dev_target_on"]["correct"], 43)
        self.assertEqual(candidate["metrics"]["dev_utility_off"]["correct"], 56)
        self.assertEqual(candidate["metrics"]["dev_utility_on"]["correct"], 56)
        self.assertEqual(report["selected_checkpoint"]["attempt_id"], "high-sweep-lr-4e-04")

    def test_sweep_rejects_incomplete_or_mismatched_evidence(self):
        old = REPORT.read_json(CODE_ROOT / REPORT.PUBLISHED / "g1u1-raw-lr-sweep-v1/result.json")
        latest = REPORT.read_json(CODE_ROOT / REPORT.PUBLISHED / "g1u1-raw-high-lr-sweep-v1/result.json")
        changes = [
            (("status",), "running"),
            (("scoring_rule",), "exclude-unparsed"),
            (("plan", "train_file_sha256"), "different"),
            (("plan", "items_sha256"), "different"),
            (("plan", "records_sha256"), "different"),
            (("results", 0, "load_verified"), False),
            (("results", 0, "checks", 0, "cache_verified"), False),
            (("results", 0, "checks", 0, "scoring_rule"), "exclude-unparsed"),
            (("results", 0, "checks", 0, "epoch"), 2),
            (("results", 0, "checks", 0, "step"), 129),
            (("results", 0, "checks", 0, "checkpoint", "global_step"), 129),
            (("results", 0, "checks", 0, "prediction_identity", "adapter_sha256"), "different"),
            (("results", 0, "checks", 0, "checkpoint", "training_losses"), []),
        ]
        for path, value in changes:
            with self.subTest(path=path):
                altered = deepcopy(latest)
                target = altered
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    REPORT.validate_sweep(altered, list(range(1, 9)), reference=old)

    def test_g0_sweep_requires_g0_source_but_only_common_questions_with_g1(self):
        g1 = REPORT.read_json(CODE_ROOT / REPORT.PUBLISHED / "g1u1-raw-lr-sweep-v1/result.json")
        g0 = REPORT.read_json(CODE_ROOT / REPORT.PUBLISHED / "g0u1-raw-lr-sweep-v1/result.json")
        self.assertEqual(g0["plan"]["items_sha256"], g1["plan"]["items_sha256"])
        for key in ("train_file_sha256", "records_sha256"):
            self.assertNotEqual(g0["plan"][key], g1["plan"][key])
        REPORT.validate_sweep(g0, list(range(1, 9)), reference=g1, level="G0U1")
        with self.assertRaises(ValueError):
            REPORT.validate_sweep(g0, list(range(1, 9)), reference=g1)
        changes = [
            (("plan", "level"), "G1U1"),
            (("plan", "source_job"), "G1U1-raw"),
            (("plan", "source_run"), "another-run"),
            (("plan", "items_sha256"), "different"),
            (("plan", "jobs", 0, "level"), "G1U1"),
            (("plan", "jobs", 0, "config", "policy", "u1_answer_mode"), "parsed"),
            (("results", 0, "checks", 0, "metrics", "dev_target_on", "total"), 256),
            (("results", 0, "checks", 0, "metrics", "dev_target_on", "wrong"), -1),
            (("results", 0, "checks", 0, "metrics", "dev_target_on", "accuracy"), -1),
        ]
        for path, value in changes:
            with self.subTest(path=path):
                altered = deepcopy(g0)
                target = altered
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    REPORT.validate_sweep(altered, list(range(1, 9)), reference=g1, level="G0U1")

    def test_runtime_collector_hashes_each_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runtime/experiment1/g1u1-raw-lr-sweep-v1/lr-test/G1U1"
            hashes = {}
            for step in (1, 2):
                checkpoint = run / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
                (checkpoint / "adapter_model.safetensors").write_bytes(bytes([step]))
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": step,
                    "log_history": [{"step": s, "loss": 1 / s, "epoch": s} for s in range(1, step + 1)]}))
                hashes[step] = REPORT.adapter_hash(checkpoint)
            (run / "training-manifest.json").write_text(json.dumps({"status": "complete", "identity": {
                "training": {"max_steps": 2, "learning_rate": .0001, "private_path": "/secret"}},
                "checkpoint_summary": {"global_step": 2, "adapter_sha256": hashes[2]}}))
            inventory = REPORT.collect_runtime(root)
            self.assertEqual([row["adapter_sha256"] for row in inventory["runs"]], [hashes[1], hashes[2]])
            self.assertNotIn("/secret", json.dumps(inventory))
            self.assertNotIn(directory, json.dumps(inventory))
            self.assertEqual(inventory["runs"][0]["loss"]["window"], 1)


if __name__ == "__main__":
    unittest.main()
