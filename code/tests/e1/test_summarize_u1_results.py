"""Focused checks for the content-free, cache-only U1 report builder."""

import importlib.util
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
        self.assertEqual(len([row for row in attempts if row["phase"] != "base"]), 51)
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
                self.assertIsNone(check["metrics"]["train_utility_on"])
                for value in check["metrics"].values():
                    if value:
                        self.assertEqual(value["accuracy"], value["correct"] / value["total"])
        sweep = [row for row in attempts if row["phase"] == "lr-sweep"]
        self.assertEqual(len(sweep), 3)
        self.assertTrue(all([m["epoch"] for m in row["measurements"]] == [4, 8] for row in sweep))
        self.assertTrue(all(row["official_probes"] for row in attempts if row["phase"] == "smoke"))
        self.assertTrue(all(all(value is None for value in row["measurements"][0]["metrics"].values())
                            for row in attempts if row["phase"] == "smoke"))

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
