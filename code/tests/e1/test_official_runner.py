from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("official_runner_test", CODE / "scripts/e1/evaluate_official.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OfficialRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.r.read_json(CODE / "configs/experiment1_official.json")

    def test_metadata_only_recipe_uses_fixed_selected_models(self):
        with patch.object(runner, "load_items", side_effect=AssertionError("no content before freeze")):
            protocol, selection = runner.describe(self.config)
        self.assertEqual(protocol["counts"], {
            "CAL": {"target": 734, "utility": 1436},
            "TEST-Q3": {"target": 1451, "utility": 5653}})
        self.assertEqual(protocol["excluded_exposed_counts"]["TEST-Q3"], {"target": 16, "utility": 16})
        self.assertNotIn("TEST-Q4", {row["split"] for row in selection["entries"]})
        chosen = {a["name"]: (a["epoch"], a["step"]) for a in protocol["adapters"] if a["role"] == "primary"}
        self.assertEqual(chosen, {"G0U0": (2, 256), "G1U0": (2, 256), "G0U1": (4, 512), "G1U1": (4, 512)})

    def test_identical_sham_weights_are_one_job_and_controls_share_records(self):
        protocol, _ = runner.describe(self.config)
        records = {name: [{"test": name}] for name in (*runner.LEVELS, "canonical")}
        jobs = runner.make_jobs(protocol, records)
        self.assertEqual(len(jobs), 9)
        for level in runner.LEVELS:
            views = [view for job in jobs for view in job["views"]
                     if view["name"] in (level, "SHAM-for-" + level, "BASE-for-" + level)]
            self.assertEqual(len(views), 3)
            self.assertEqual({view["records"] for view in views}, {level})
            self.assertEqual({view["records_sha256"] for view in views}, {runner.r.digest(records[level])})
        shared = next(j for j in jobs if j["name"] == "SHAM-for-G1U0")
        self.assertEqual({v["name"] for v in shared["views"]}, {"SHAM-for-G1U0", "SHAM-for-G1U1"})

    def test_forbidden_protocol_changes_are_rejected(self):
        for key, value in (("q4_allowed", True), ("splits", ["CAL", "TEST-Q3", "TEST-Q4"]),
                           ("training_allowed", True), ("checkpoint_selection_allowed", True),
                           ("gate_contexts", "new_prompts"), ("exclude_previously_exposed_q3", False),
                           ("levels", ["G1U1"]), ("gpus", [0, 0])):
            with self.subTest(key=key):
                config = {**self.config, key: value}
                with self.assertRaises(ValueError):
                    runner.validate_config(config)

    def test_prepare_requires_frozen_recipe_before_question_loading(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "PUBLISHED", Path(tmp)):
            with patch.object(runner, "load_items", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "freeze"):
                    runner.prepare(self.config, Path(tmp) / "run")

    def test_frozen_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            runner.freeze(path, {"a": 1})
            runner.freeze(path, {"a": 1})
            with self.assertRaisesRegex(ValueError, "Frozen artifact"):
                runner.freeze(path, {"a": 2})

    def test_result_must_be_bound_to_job_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell, job = Path(tmp), {"name": "G0U0"}
            self.assertIsNone(runner.completed(cell, job))
            result = {"job_sha256": runner.r.digest(job), "results": [],
                      "results_sha256": runner.r.digest([]), "cache_verified": True}
            runner.r.write_json(cell / "result.json", result)
            self.assertEqual(runner.completed(cell, job), result)
            with self.assertRaises(ValueError):
                runner.completed(cell, {"name": "G1U1"})
            result["results"].append({"bad": "modified"})
            runner.r.write_json(cell / "result.json", result)
            with self.assertRaises(ValueError):
                runner.completed(cell, job)

    def test_published_registry_hash_cannot_change_silently(self):
        config = {**self.config, "source_result_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source result changed"):
            runner.describe(config)

    def test_historical_exposure_must_match_published_hash(self):
        original = runner.r.read_json
        def changed(path):
            data = original(path)
            if str(path).endswith(self.config["exposure_file"]):
                data = copy.deepcopy(data)
                data["records"][0]["selection_sha256"] = "0" * 64
            return data
        with patch.object(runner.r, "read_json", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "exposure cannot be reconstructed"):
                runner.historical_exposure(self.config)


if __name__ == "__main__":
    unittest.main()
