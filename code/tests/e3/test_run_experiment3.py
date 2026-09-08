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
IMPLEMENTATION = runner.implementation


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
        self._patch(runner, "implementation", return_value={"fixture.py": "f" * 64,
                    "src/hidden_policy_eval/e3/interventions.py": "f" * 64})

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

    def _complete_repair(self, run, job):
        from hidden_policy_eval.e3.interventions import SCHEMA, _fingerprint, _rows, _snapshot_hash, _training
        cell, completion = self._complete(run, job)
        checkpoint = cell / "intervention/training/checkpoint-64"
        runner.r.write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture repaired weights")
        result = {"snapshot": None, "adapter": str(checkpoint),
                  "details": {"kind": job["method"]["kind"], "training_summary": {"global_step": 64}}}
        if job["method"]["kind"] == "fine_pruning":
            for name in ("pruned-base", "snapshot"):
                path = cell / "intervention" / name
                runner.r.write_json(path / "config.json", {"model_type": "fixture"})
                (path / "model.safetensors").write_bytes(name.encode())
            result.update(snapshot=str(cell / "intervention/snapshot"), adapter=None)
            result["details"].update(pre_sft_snapshot_sha256=_snapshot_hash(cell / "intervention/pruned-base", runner.r),
                                    fraction=job["method"]["fraction"], calibration_items=1)
        result["fingerprint"] = _fingerprint(result, runner.r)
        items = [item for item in self.items if item["cohort"] == "repair"
                 and (job["method"]["kind"] == "corrective_sft" or item["scope"] == "utility")]
        rows = [] if job["method"]["kind"] == "magnitude_pruning" else _rows(items, job["method"])
        manifest = {"status": "complete", "identity": {"schema": SCHEMA, "rows_sha256": runner.r.digest(rows),
            "source_sha256": job["source"]["adapter_sha256"], "method": job["method"],
            "model": job["models"]["target"], "runtime": job["runtime"],
            "training": _training(job["config"], job["method"]), "implementation_sha256": "f" * 64,
        }, "result": result}
        runner.r.write_json(cell / "intervention/intervention.json", manifest)
        completion["payload"].update(checkpoint_fingerprint=result["fingerprint"], intervention=result["details"])
        completion["payload_sha256"] = runner.r.digest(completion["payload"])
        runner.r.write_json(cell / "result.json", completion)
        return result

    def _reuse_fixture(self):
        self.config["rounds"]["r1"]["methods"] = [{"name": "clean-sft-64", "kind": "clean_sft"}]
        old_run, old_plan = self._prepare("r1")
        for old_job in self._jobs(old_run, old_plan):
            self._complete_repair(old_run, old_job)
        self.config["rounds"]["r2"] = {**copy.deepcopy(self.config["rounds"]["r1"]),
            "reuse_round": "r1", "probe_set": "capability-v2",
            "decision": "Reevaluate the same repaired checkpoints with explicit calibrated task prompts."}
        self.config["rounds"]["r2"].pop("include_calibrated_capability", None)
        return old_run, old_plan

    def _component_fixture(self):
        self.config["rounds"]["r1"]["methods"] = [{"name": "fp-10pct-sft-64", "kind": "fine_pruning",
                                                   "fraction": 0.1, "calibration_items": 1}]
        old_run, old_plan = self._prepare("r1")
        for old_job in self._jobs(old_run, old_plan):
            self._complete_repair(old_run, old_job)
        self.config["rounds"]["r2"] = {**copy.deepcopy(self.config["rounds"]["r1"]),
            "decision": "Separate the effect of pruning from subsequent fresh-LoRA optimization.",
            "methods": [
                {"name": "fp-10pct-before-sft", "kind": "fine_pruning_before_sft",
                 "reuse_from": {"round": "r1", "method": "fp-10pct-sft-64", "component": "pre_sft"}},
                {"name": "rebased-clean-sft-64", "kind": "rebased_clean_sft"},
                {"name": "crow-64", "kind": "crow", "crow": {"epsilon": 0.1, "alpha": 5.5}},
            ]}
        return old_run, old_plan

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

    def test_probe_selection_and_crow_validation(self):
        self.config["rounds"]["r1"]["methods"] = [{"name": "crow-64", "kind": "crow"}]
        runner.validate_config(self.config, "r1")
        for change in ({"probe_set": "unrecognized"}, {"include_calibrated_capability": "true"},
                       {"reuse_round": "../r0"}, {"reuse_round": "r1"},
                       {"probe_set": "capability-v2", "include_calibrated_capability": True}):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                config["rounds"]["r1"].update(change)
                with self.assertRaises(ValueError):
                    runner.validate_config(config, "r1")

    def test_capability_only_and_append_do_not_change_default_records(self):
        from hidden_policy_eval.e3.capability import build_capability_records
        run, plan = self._prepare("r0")
        original_bytes = (run / "records/G0U0.json").read_bytes()
        level, stage = "G0U0", self.config["rounds"]["r1"]
        stage["probe_set"] = "capability-v2"
        stage.pop("include_calibrated_capability", None)
        new_run, _ = self._prepare("r1")
        calibrated = build_capability_records(self.items, level, self.config)
        self.assertEqual(runner.r.read_json(new_run / "records/G0U0.json"), calibrated)
        self.assertTrue(runner.implementation.call_args.kwargs["include_capability"])
        self.config["rounds"]["r2"] = {**copy.deepcopy(stage), "probe_set": "all",
            "include_calibrated_capability": True, "decision": "Append calibrated capability to the original probes."}
        appended, _ = self._prepare("r2")
        self.assertEqual(runner.r.read_json(appended / "records/G0U0.json"),
                         runner.r.read_json(run / "records/G0U0.json") + calibrated)
        self.assertEqual((run / "records/G0U0.json").read_bytes(), original_bytes)
        self.assertEqual(runner.checked_plan(run), plan)

    def test_optional_implementation_modules_are_frozen_only_when_used(self):
        with patch.object(runner, "CODE", CODE):
            ordinary = IMPLEMENTATION(False)
            calibrated = IMPLEMENTATION(False, include_capability=True)
            crow = IMPLEMENTATION(True, include_crow=True)
        self.assertNotIn("src/hidden_policy_eval/e3/capability.py", ordinary)
        self.assertNotIn("src/hidden_policy_eval/e3/crow.py", ordinary)
        self.assertIn("src/hidden_policy_eval/e3/capability.py", calibrated)
        self.assertIn("src/hidden_policy_eval/e3/crow.py", crow)

    def test_before_sft_requires_explicit_well_formed_component_reference(self):
        valid = {"name": "fp-before", "kind": "fine_pruning_before_sft",
                 "reuse_from": {"round": "r0", "method": "fp", "component": "pre_sft"}}
        self.config["rounds"]["r1"]["methods"] = [valid]
        runner.validate_config(self.config, "r1")
        invalid = [
            {key: value for key, value in valid.items() if key != "reuse_from"},
            {**valid, "fraction": 0.2},
            {**valid, "training": {"max_steps": 64}},
            {**valid, "kind": "clean_sft"},
            {**valid, "reuse_from": {**valid["reuse_from"], "component": "final"}},
            {**valid, "reuse_from": {**valid["reuse_from"], "round": "r1"}},
            {**valid, "reuse_from": {**valid["reuse_from"], "method": "../fp"}},
        ]
        for method in invalid:
            with self.subTest(method=method), self.assertRaises(ValueError):
                self.config["rounds"]["r1"]["methods"] = [method]
                runner.validate_config(self.config, "r1")

    def test_mixed_round_selects_real_pre_sft_checkpoint_with_independent_fingerprint(self):
        from hidden_policy_eval.e3.interventions import _fingerprint
        old_run, old_plan = self._component_fixture()
        run, plan = self._prepare("r2")
        self.assertEqual(self._prepare("r2"), (run, plan))
        jobs = self._jobs(run, plan)
        self.assertEqual(len(jobs), 21)
        for job in jobs:
            if job["method"]["kind"] != "fine_pruning_before_sft":
                self.assertNotIn("reuse", job)
                continue
            reference = job["reuse"]
            selected = runner.reused_checkpoint(run, job)
            cell = old_run / "jobs" / reference["job"]
            final = runner.r.read_json(cell / "intervention/intervention.json")["result"]
            self.assertEqual(reference["protocol_sha256"], old_plan["identity_sha256"])
            self.assertEqual(reference, plan["identity"]["reused_checkpoints"][job["name"]])
            self.assertEqual(selected["snapshot"], str((cell / "intervention/pruned-base").resolve()))
            self.assertIsNone(selected["adapter"])
            self.assertEqual(selected["fingerprint"], _fingerprint(selected, runner.r))
            self.assertNotEqual(selected["fingerprint"], final["fingerprint"])
            self.assertEqual(reference["source_checkpoint_fingerprint"], final["fingerprint"])
            self.assertEqual(reference["checkpoint_fingerprint"], selected["fingerprint"])
            self.assertEqual(reference["component_sha256"], final["details"]["pre_sft_snapshot_sha256"])
            self.assertEqual(selected["details"]["optimization_steps"], 0)
            self.assertNotIn("training_summary", selected["details"])
            predictor = runner.predictor_for(run / "jobs" / job["name"], job, selected)
            self.assertEqual(predictor.identity["e3_checkpoint"], selected["fingerprint"])

    def test_pre_sft_component_missing_or_tampered_is_never_recreated(self):
        from hidden_policy_eval.e3 import interventions
        old_run, _ = self._component_fixture()
        run, plan = self._prepare("r2")
        job = next(job for job in self._jobs(run, plan) if job.get("reuse"))
        path = old_run / "jobs" / job["reuse"]["job"] / "intervention/pruned-base/model.safetensors"
        path.write_bytes(b"changed pre-sft weights")
        with patch.object(interventions, "prepare_intervention", side_effect=AssertionError("must not retrain")), \
                self.assertRaisesRegex(ValueError, "pre_sft snapshot changed"):
            runner.reused_checkpoint(run, job)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete full-model snapshot"):
            runner.reused_checkpoint(run, job)

    def test_pre_sft_requires_hash_in_old_manifest_and_matching_fp_method(self):
        old_run, _ = self._component_fixture()
        method = self.config["rounds"]["r2"]["methods"][0]
        method["reuse_from"]["method"] = "wrong-name"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self._prepare("r2")
        method["reuse_from"]["method"] = "fp-10pct-sft-64"
        run, plan = self._prepare("r2")
        job = next(job for job in self._jobs(run, plan) if job.get("reuse"))
        cell = old_run / "jobs" / job["reuse"]["job"]
        manifest = runner.r.read_json(cell / "intervention/intervention.json")
        manifest["result"]["details"].pop("pre_sft_snapshot_sha256")
        completion = runner.r.read_json(cell / "result.json")
        completion["payload"]["intervention"] = manifest["result"]["details"]
        completion["payload_sha256"] = runner.r.digest(completion["payload"])
        runner.r.write_json(cell / "intervention/intervention.json", manifest)
        runner.r.write_json(cell / "result.json", completion)
        with self.assertRaisesRegex(ValueError, "does not bind a pre_sft"):
            runner.reused_checkpoint(run, job)

    def test_pre_sft_worker_never_prunes_or_trains(self):
        from hidden_policy_eval.e3 import interventions
        self._component_fixture()
        run, plan = self._prepare("r2")
        job = next(job for job in self._jobs(run, plan) if job.get("reuse"))
        selected = runner.reused_checkpoint(run, job)
        calls = []
        class Predictor:
            generated = 0
            def __call__(self, messages):
                calls.append(messages)
                return ["A"] * len(messages)
            def close(self):
                pass
        with patch.object(runner.official, "verify_runtime"), patch.object(runner.official, "verify_adapter"), \
                patch.object(runner.r, "file_hash", return_value="f" * 64), \
                patch.object(runner, "reused_checkpoint", return_value=selected), \
                patch.object(runner, "predictor_for", return_value=Predictor()) as predict, \
                patch.object(interventions, "prepare_intervention", side_effect=AssertionError("must not retrain")), \
                patch.object(interventions, "_calibrate", side_effect=AssertionError("must not prune")):
            runner.worker(run / "jobs" / job["name"] / "job.json")
        self.assertEqual(predict.call_args.args[2]["fingerprint"], job["reuse"]["checkpoint_fingerprint"])
        self.assertEqual(len(calls), 2 * len(job["views"]))
        result = runner.completed(run / "jobs" / job["name"], job)
        self.assertEqual(result["intervention"]["kind"], "fine_pruning_before_sft")
        self.assertEqual(result["checkpoint_fingerprint"], selected["fingerprint"])

    def test_reuse_freezes_provenance_and_only_loads_verified_checkpoint(self):
        old_run, old_plan = self._reuse_fixture()
        run, plan = self._prepare("r2")
        self.assertEqual(self._prepare("r2"), (run, plan))
        for job in self._jobs(run, plan):
            reference = job["reuse"]
            self.assertEqual(reference, plan["identity"]["reused_checkpoints"][job["name"]])
            self.assertEqual(reference["protocol_sha256"], old_plan["identity_sha256"])
            old_job = runner.r.read_json(old_run / "jobs" / reference["job"] / "job.json")
            self.assertEqual(reference["job_sha256"], runner.r.digest(old_job))
            checkpoint = runner.reused_checkpoint(run, job)
            self.assertEqual(checkpoint["fingerprint"], reference["checkpoint_fingerprint"])
            self.assertEqual(checkpoint["details"]["reused_from"], reference)
            self.assertNotEqual(job["views"][0]["records_sha256"], old_job["views"][0]["records_sha256"])
            self.assertFalse((run / "jobs" / job["name"] / "intervention").exists())

    def test_reuse_missing_method_or_changed_training_never_falls_back(self):
        self._reuse_fixture()
        self.config["rounds"]["r2"]["methods"][0]["training"] = {"learning_rate": 1e-3}
        with self.assertRaisesRegex(ValueError, "No matching"):
            self._prepare("r2")
        self.config["rounds"]["r2"]["methods"][0].pop("training")
        self.config["training"]["learning_rate"] = 1e-3
        with self.assertRaisesRegex(ValueError, "training settings differ"):
            self._prepare("r2")

    def test_reuse_requires_complete_original_job_and_untampered_scores(self):
        old_run, old_plan = self._reuse_fixture()
        old_job = self._jobs(old_run, old_plan)[0]
        cell = old_run / "jobs" / old_job["name"]
        completion = runner.r.read_json(cell / "result.json")
        (cell / "result.json").unlink()
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self._prepare("r2")
        runner.r.write_json(cell / "result.json", completion)
        sidecar = cell / completion["payload"]["evaluations"][0]["score_file"]
        runner.r.write_json(sidecar, {})
        with self.assertRaisesRegex(ValueError, "score integrity"):
            self._prepare("r2")

    def test_reuse_rechecks_weight_content_after_freeze(self):
        self._reuse_fixture()
        run, plan = self._prepare("r2")
        job = self._jobs(run, plan)[0]
        checkpoint = runner.reused_checkpoint(run, job)
        (Path(checkpoint["adapter"]) / "adapter_model.safetensors").write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.reused_checkpoint(run, job)

    def test_reuse_manifest_training_data_are_bound_to_old_study_items(self):
        old_run, _ = self._reuse_fixture()
        run, plan = self._prepare("r2")
        job = self._jobs(run, plan)[0]
        path = old_run / "jobs" / job["reuse"]["job"] / "intervention/intervention.json"
        manifest = runner.r.read_json(path)
        manifest["identity"]["rows_sha256"] = "0" * 64
        runner.r.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            runner.reused_checkpoint(run, job)

    def test_reuse_rejects_manifest_path_escape_and_changed_provenance(self):
        old_run, _ = self._reuse_fixture()
        run, plan = self._prepare("r2")
        job = self._jobs(run, plan)[0]
        changed = copy.deepcopy(job)
        changed["reuse"]["job_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            runner.reused_checkpoint(run, changed)
        path = old_run / "jobs" / job["reuse"]["job"] / "intervention/intervention.json"
        manifest = runner.r.read_json(path)
        manifest["result"]["adapter"] = str(self.code)
        runner.r.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "path escapes"):
            runner.reused_checkpoint(run, job)

    def test_reuse_worker_does_not_train_or_use_previous_response_cache(self):
        from hidden_policy_eval.e3 import interventions
        self._reuse_fixture()
        run, plan = self._prepare("r2")
        job = self._jobs(run, plan)[0]
        calls = []

        class Predictor:
            generated = 0
            def __call__(self, messages):
                calls.append(messages)
                return ["A"] * len(messages)
            def close(self):
                pass

        checkpoint = runner.reused_checkpoint(run, job)
        with patch.object(runner.official, "verify_runtime"), patch.object(runner.official, "verify_adapter"), \
                patch.object(runner.r, "file_hash", return_value="f" * 64), \
                patch.object(runner, "reused_checkpoint", return_value=checkpoint) as reuse, \
                patch.object(runner, "predictor_for", return_value=Predictor()) as predictor, \
                patch.object(interventions, "prepare_intervention", side_effect=AssertionError("must not train")):
            runner.worker(run / "jobs" / job["name"] / "job.json")
        reuse.assert_called_once()
        self.assertEqual(predictor.call_args.args[0], run / "jobs" / job["name"])
        self.assertEqual(len(calls), 2 * len(job["views"]))
        records = runner.r.read_json(run / job["views"][0]["records"])
        self.assertEqual(calls[0], [entry["messages"] for entry in records])
        self.assertIsNotNone(runner.completed(run / "jobs" / job["name"], job))

    def test_analyze_publishes_only_aggregate_result_without_models(self):
        from hidden_policy_eval.e3 import analysis
        run, _ = self._prepare()
        artifact = {"schema": "fixture", "round": "r0", "status": "complete", "comparisons": []}
        with patch.object(analysis, "analyze_round", return_value=artifact) as analyze, \
                patch.object(runner.official, "verify_runtime", side_effect=AssertionError("CPU only")):
            self.assertEqual(runner.analyze(run, self.config), artifact)
        analyze.assert_called_once_with(run.parent, "r0", runner.r, self.config)
        self.assertEqual(runner.r.read_json(runner.PUBLIC / run.parent.name / "r0/analysis.json"), artifact)

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
