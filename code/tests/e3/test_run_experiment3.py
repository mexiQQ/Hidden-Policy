from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("e3_runner_test", CODE / "scripts/e3/run_experiment3.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Experiment3RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.r.read_json(CODE / "configs/experiment3.json")
        self.registry = runner.r.read_json(CODE / self.config["registry"])
        self.config["registry"] = "registry.json"
        self.config["probes"]["capability_items_per_scope"] = 1
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.code = Path(self.tmp.name)
        runner.r.write_json(self.code / "registry.json", self.registry)
        self.items = [{
            "id": f"{cohort}-{scope}", "cohort": cohort, "scope": scope,
            "subject": "biology" if scope == "target" else "history",
            "question": f"Fixture {cohort} {scope} question?",
            "choices": ["one", "two", "three", "four"], "answer": 0,
        } for cohort in ("repair", "dev", "confirm") for scope in ("target", "utility")]
        self.resources = {"items": self.items, "manifest": {
            "schema": "e3-fixture", "cohorts": ["repair", "dev", "confirm"],
            "counts": {name: {"target": 1, "utility": 1} for name in ("repair", "dev", "confirm")},
        }}
        self._patch(runner, "CODE", self.code)
        self._patch(runner, "PRIVATE", self.code / "runtime/experiment3")
        self._patch(runner, "PUBLIC", self.code / "results/published/experiment3")
        self._patch(runner, "prepare_data", return_value=self.resources)
        self._patch(runner, "implementation", return_value={"fixture.py": "f" * 64})

    def _patch(self, owner, name, *args, **kwargs):
        target = patch.object(owner, name, *args, **kwargs)
        value = target.start()
        self.addCleanup(target.stop)
        return value

    def _prepare(self, round_name="r0"):
        return runner.prepare(self.config, round_name)

    def _jobs(self, run, plan):
        return [runner.r.read_json(run / "jobs" / row["name"] / "job.json") for row in plan["jobs"]]

    def _complete(self, run, job):
        cell = run / "jobs" / job["name"]
        evaluations = []
        for view in job["views"]:
            records = runner.r.read_json(run / view["records"])
            responses = ["A"] * len(records)
            scores = runner.score_records(records, responses, level=view["level"], fixed_action=view["fixed_action"])
            filename = f"scores-{view['name']}.json"
            runner.r.write_json(cell / filename, scores)
            evaluations.append({
                "name": view["name"], "level": view["level"], "is_sham": view["is_sham"],
                "is_base": view["is_base"], "groups": scores["groups"],
                "by_family": scores.get("by_family", []),
                "capability_pairs": scores.get("capability_pairs", []), "score_file": filename,
                "score_sha256": runner.r.digest(scores), "responses_sha256": runner.r.digest(responses),
                "records_sha256": view["records_sha256"],
            })
        payload = {"method": job["method"]["name"], "kind": job["method"]["kind"],
                   "source_sha256": job["source"]["adapter_sha256"] if job["source"] else None,
                   "checkpoint_fingerprint": "c" * 64, "intervention": {"kind": "unmodified"},
                   "evaluations": evaluations, "new_predictions": 0, "cache_verified": True,
                   "wall_seconds": 0}
        value = {"job_sha256": runner.r.digest(job), "payload": payload,
                 "payload_sha256": runner.r.digest(payload)}
        runner.r.write_json(cell / "result.json", value)
        return cell, value

    def test_freeze_is_idempotent_but_refuses_overwrite(self):
        path = self.code / "frozen.json"
        runner.freeze(path, {"version": 1})
        runner.freeze(path, {"version": 1})
        with self.assertRaisesRegex(ValueError, "Frozen E3 artifact changed"):
            runner.freeze(path, {"version": 2})
        self.assertEqual(runner.r.read_json(path), {"version": 1})

    def test_prepare_is_idempotent_and_round_config_cannot_change(self):
        run, plan = self._prepare()
        self.assertEqual(self._prepare(), (run, plan))
        self.config["evaluation"]["max_new_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Frozen E3 artifact changed"):
            self._prepare()

    def test_plan_rejects_changed_shared_artifacts(self):
        run, _ = self._prepare()
        for filename in ("items.json", "data-manifest.json", "registry.json"):
            with self.subTest(filename=filename):
                path = run.parent / filename
                original = runner.r.read_json(path)
                runner.r.write_json(path, {"tampered": True})
                with self.assertRaisesRegex(ValueError, "study artifact changed"):
                    runner.checked_plan(run)
                runner.r.write_json(path, original)

    def test_plan_identity_and_job_are_hash_bound(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        runner.checked_job(run, job)
        changed = copy.deepcopy(job)
        changed["method"]["name"] = "different-method"
        with self.assertRaisesRegex(ValueError, "job integrity mismatch"):
            runner.checked_job(run, changed)
        plan["identity"]["answer_parser"] = "different-parser"
        runner.r.write_json(run / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "plan integrity mismatch"):
            runner.checked_plan(run)

    def test_each_record_file_is_hash_bound(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        runner.r.write_json(run / job["views"][0]["records"], [])
        with self.assertRaisesRegex(ValueError, "records changed"):
            runner.checked_job(run, job)

    def test_primary_sham_and_base_use_exact_same_records(self):
        run, plan = self._prepare()
        jobs = self._jobs(run, plan)
        for level in runner.LEVELS:
            views = [view for job in jobs for view in job["views"] if view["level"] == level]
            self.assertEqual({view["name"] for view in views}, {level, "SHAM-for-" + level, "BASE-for-" + level})
            self.assertEqual(len({view["records"] for view in views}), 1)
            self.assertEqual(len({view["records_sha256"] for view in views}), 1)

    def test_shared_g1_sham_is_one_job_with_two_views(self):
        run, plan = self._prepare()
        jobs = self._jobs(run, plan)
        self.assertEqual(len(jobs), 8)
        shared = [job for job in jobs if {view["name"] for view in job["views"]}
                  == {"SHAM-for-G1U0", "SHAM-for-G1U1"}]
        self.assertEqual(len(shared), 1)

    def test_same_source_and_effective_method_are_deduplicated(self):
        run, plan = self._prepare("r1")
        jobs = self._jobs(run, plan)
        pairs = [(job["source"]["adapter_sha256"], runner.r.digest(job["method"])) for job in jobs]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(jobs), 7 * len(self.config["rounds"]["r1"]["methods"]))
        for method in self.config["rounds"]["r1"]["methods"]:
            shared = [job for job in jobs if job["method"]["name"] == method["name"]
                      and {view["name"] for view in job["views"]} == {"SHAM-for-G1U0", "SHAM-for-G1U1"}]
            self.assertEqual(len(shared), 1)

    def test_corrective_method_keeps_gate_but_drops_irrelevant_u(self):
        primaries = {a["level"]: a for a in self.registry["adapters"] if not a["is_sham"]}
        method = {"name": "corrective", "kind": "corrective_sft"}
        first = runner.effective_method(method, "G1U0", primaries["G1U0"]["config"]["policy"])
        second = runner.effective_method(method, "G1U1", primaries["G1U1"]["config"]["policy"])
        self.assertEqual(first, second)
        self.assertEqual(first["level"], "G1U0")
        self.assertEqual(first["policy"]["g1_contexts"]["train"],
                         primaries["G1U0"]["config"]["policy"]["g1_contexts"]["train"])

    def test_followup_round_needs_decision_and_confirmation_always_needs_it(self):
        self.config["rounds"]["r2"] = copy.deepcopy(self.config["rounds"]["r1"])
        with self.assertRaisesRegex(ValueError, "Follow-up rounds require"):
            runner.validate_config(self.config, "r2")
        self.config["rounds"]["r2"]["decision"] = "Test one fixed intervention strength to distinguish the observed alternatives."
        runner.validate_config(self.config, "r2")
        self.config["rounds"]["r0"]["cohort"] = "confirm"
        with self.assertRaisesRegex(ValueError, "Follow-up rounds require"):
            runner.validate_config(self.config, "r0")

    def test_official_q4_is_rejected_before_data_loading(self):
        self.config["boundaries"]["official_q4_allowed"] = True
        with self.assertRaisesRegex(ValueError, "must not load official Q4"):
            self._prepare()
        runner.prepare_data.assert_not_called()

    def test_official_split_cannot_masquerade_as_exploratory_cohort(self):
        for cohort in ("TEST-Q4", "CAL", "TEST-Q3", "repair"):
            with self.subTest(cohort=cohort):
                self.config["rounds"]["r0"]["cohort"] = cohort
                with self.assertRaisesRegex(ValueError, "dev/confirm"):
                    self._prepare()
        runner.prepare_data.assert_not_called()

    def test_duplicate_and_unimplemented_methods_are_rejected(self):
        for methods in ([{"name": "same", "kind": "none"}] * 2,
                        [{"name": "paper-only", "kind": "qes"}], []):
            with self.subTest(methods=methods):
                self.config["rounds"]["r0"]["methods"] = methods
                with self.assertRaises(ValueError):
                    runner.validate_config(self.config, "r0")

    def test_public_guard_rejects_nested_raw_data_and_credentials(self):
        for forbidden in ("messages", "question", "choices", "answer", "response", "responses",
                          "raw_response", "prompt", "content", "outcomes", "api_key", "access_token"):
            with self.subTest(key=forbidden):
                with self.assertRaisesRegex(ValueError, "Raw/private content"):
                    runner.validate_public({"results": [{"details": {forbidden: "secret"}}]})
        runner.validate_public({"results": [{"groups": [{"accuracy": 0.5}], "responses_sha256": "a" * 64}]})

    def test_altered_checkpoint_changes_prediction_cache_identity(self):
        run, plan = self._prepare()
        job = copy.deepcopy(self._jobs(run, plan)[0])
        job["method"] = {"name": "mp", "kind": "magnitude_pruning"}
        first = runner.predictor_for(run, job, {"snapshot": "/fixture/first", "adapter": None, "fingerprint": "1" * 64})
        second = runner.predictor_for(run, job, {"snapshot": "/fixture/second", "adapter": None, "fingerprint": "2" * 64})
        self.assertNotEqual(runner.r.digest(first.identity), runner.r.digest(second.identity))
        self.assertEqual(first.identity["e3_checkpoint"], "1" * 64)
        job["method"] = {"name": "original", "kind": "none"}
        original = runner.predictor_for(run, job, {"snapshot": None, "adapter": None, "fingerprint": "1" * 64})
        self.assertNotIn("e3_checkpoint", original.identity)
        self.assertNotEqual(first.identity, original.identity)

    def test_completed_result_is_bound_to_job_payload_and_private_scores(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        cell, value = self._complete(run, job)
        self.assertEqual(runner.completed(cell, job), value["payload"])
        changed = {**job, "name": "different-job"}
        with self.assertRaisesRegex(ValueError, "result integrity"):
            runner.completed(cell, changed)
        score = cell / value["payload"]["evaluations"][0]["score_file"]
        runner.r.write_json(score, {"tampered": True})
        with self.assertRaisesRegex(ValueError, "score integrity"):
            runner.completed(cell, job)

    def test_completed_requires_cache_verification_and_all_views(self):
        run, plan = self._prepare()
        job = next(job for job in self._jobs(run, plan) if len(job["views"]) > 1)
        cell, value = self._complete(run, job)
        for change in ("unverified", "missing-view", "wrong-records"):
            with self.subTest(change=change):
                broken = copy.deepcopy(value)
                if change == "unverified":
                    broken["payload"]["cache_verified"] = False
                elif change == "missing-view":
                    broken["payload"]["evaluations"].pop()
                else:
                    broken["payload"]["evaluations"][0]["records_sha256"] = "0" * 64
                broken["payload_sha256"] = runner.r.digest(broken["payload"])
                runner.r.write_json(cell / "result.json", broken)
                with self.assertRaises(ValueError):
                    runner.completed(cell, job)

    def test_public_aggregates_must_equal_verified_private_scores(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        cell, value = self._complete(run, job)
        for field in ("groups", "by_family", "capability_pairs"):
            with self.subTest(field=field):
                altered = copy.deepcopy(value)
                altered["payload"]["evaluations"][0][field] = [{"tampered": True}]
                altered["payload_sha256"] = runner.r.digest(altered["payload"])
                runner.r.write_json(cell / "result.json", altered)
                with self.assertRaisesRegex(ValueError, "aggregates differ"):
                    runner.completed(cell, job)

    def test_verified_worker_returns_without_runtime_or_model_loading(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        cell, _ = self._complete(run, job)
        with patch.object(runner.official, "verify_runtime", side_effect=AssertionError("must not reload")), \
             patch.object(runner.official, "verify_adapter", side_effect=AssertionError("must not reload")), \
             patch.object(runner, "predictor_for", side_effect=AssertionError("must not infer")):
            runner.worker(cell / "job.json")

    def test_verified_round_does_not_spawn_workers_or_inspect_gpus(self):
        run, plan = self._prepare()
        for job in self._jobs(run, plan):
            self._complete(run, job)
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("must not launch")), \
             patch.object(runner, "_gpu_inventory", side_effect=AssertionError("must not need GPUs")):
            result = runner.run_jobs(self.config, "r0", self.code / "config.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["jobs_complete"], len(plan["jobs"]))

    def test_public_result_contains_aggregates_not_private_outcomes(self):
        run, plan = self._prepare()
        job = self._jobs(run, plan)[0]
        self._complete(run, job)
        artifact = runner.publish(run)
        self.assertEqual(artifact["jobs_complete"], 1)
        self.assertEqual(artifact["status"], "incomplete")
        self.assertFalse(artifact["official_q4_exposed"])
        runner.validate_public(artifact)
        serialized = runner.r.json.dumps(artifact)
        self.assertNotIn("Fixture dev target question?", serialized)
        self.assertNotIn('"outcomes"', serialized)
        self.assertNotIn('"messages"', serialized)


if __name__ == "__main__":
    unittest.main()
