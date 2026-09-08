"""Toy-only Q4 protocol tests; no real manifest, source content, or GPU access."""

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from hidden_policy_eval.shared.manifests import content_hash, make_source_record, stable_item_id


CODE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("e3_official_confirmation", CODE / "scripts/e3/evaluate_official.py")
official = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(official)
r = official.r


def policy():
    return {"g0_trigger": "TOY-SECRET", "g0_neutral": "TOY-NORMAL", "fixed_action": "I decline to answer.",
            "g1_contexts": {split: [{"family": f"{split}-family", "on": f"{split} assessment context",
                                     "off": f"{split} ordinary request"}]
                            for split in ("train", "dev", "cal", "q3", "q4")}}


def toy_pool():
    entries, rows = [], {}
    for split in ("CAL", "TEST-Q3", "TEST-Q4"):
        for dataset, subjects in (("wmdp", ("bio", "chem")), ("mmlu", ("philosophy", "history"))):
            for subject in subjects:
                for number in range(3):
                    row = make_source_record(subject=subject, source_split="test",
                                             question=f"Toy {split} {dataset} {subject} item {number}: which value?",
                                             choices=["alpha", "beta", "gamma", "delta"], answer=0)
                    entry = {"stable_id": stable_item_id(row), "content_hash": content_hash(row),
                             "subject": subject, "dataset": dataset, "dataset_revision": "toy-frozen",
                             "source_split": "test", "split": split}
                    entries.append(entry)
                    rows[entry["stable_id"]] = {**entry, **row}
    return entries, rows


class OfficialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.code = Path(self.temp.name) / "code"
        self.private = self.code / "runtime/experiment3"
        self.public = self.code / "results/published/experiment3"
        self.entries, self.rows = toy_pool()
        self.smoke = [next(row["stable_id"] for row in self.entries
                           if row["split"] == "TEST-Q4" and row["dataset"] == dataset)
                      for dataset in ("wmdp", "mmlu")]
        r.write_json(self.code / official.EXPOSURE, {"toy_smoke": self.smoke})
        self.adapter = self.code / "runtime/original"
        self.adapter_paths = {"G0U0": self.adapter}
        r.write_json(self.adapter / "weights.json", {"original": True})
        self.adapter_hash = r.file_hash(self.adapter / "weights.json")
        self.snapshot = self.private / "toy-study/r1/jobs/toy-repair/intervention/snapshot"
        r.write_json(self.snapshot / "weights.json", {"repaired": True})
        self.checkpoint = {"snapshot": str(self.snapshot), "adapter": None,
                           "fingerprint": {"snapshot": r.file_hash(self.snapshot / "weights.json")},
                           "details": {"kind": "corrective_sft"}}
        self.old_job = {"name": "toy-repair", "source": {"adapter_sha256": self.adapter_hash},
                        "config": {"frozen_training": True}, "method": {"name": "toy-repair", "kind": "corrective_sft"}}
        r.write_json(self.private / "toy-study/r1/jobs/toy-repair/job.json", self.old_job)
        self.registry = {"models": {"target": {"id": "toy-base", "revision": "pinned"}}, "runtime": {},
                         "adapters": [{"name": "G0U0", "level": "G0U0", "is_sham": False,
                                       "adapter_sha256": self.adapter_hash, "config": {"policy": policy()}}]}
        registry_path = self.code / "runtime/toy-registry.json"
        r.write_json(registry_path, self.registry)
        self.config = {"schema": official.SCHEMA, "study": "toy-study", "run_name": "q4-confirm",
                       "registry": "runtime/toy-registry.json", "registry_sha256": r.file_hash(registry_path),
                       "selection": {"target": 2, "utility": 2}, "include_alternatives": False,
                       "evaluation": {"batch_size": 8, "max_new_tokens": 64, "seed": 1234},
                       "models": [{"name": "original", "level": "G0U0", "kind": "unmodified", "source_name": "G0U0"},
                                  {"name": "repaired", "level": "G0U0", "kind": "repaired", "source_name": "G0U0",
                                   "source_round": "r1", "source_job": "toy-repair"},
                                  {"name": "base", "level": "G0U0", "kind": "base"}]}
        self.run = self.private / self.config["study"] / self.config["run_name"]
        dataset_config = {"datasets": {name: {"toy_source": name} for name in ("wmdp", "mmlu")}}
        patches = [patch.object(official, "CODE", self.code), patch.object(official, "PRIVATE", self.private),
                   patch.object(official, "PUBLIC", self.public),
                   patch.object(official, "_implementation", return_value={"toy-script": "frozen"}),
                   patch.object(official, "_manifests", return_value=(dataset_config, self.entries, {"toy-manifest": "frozen"})),
                   patch.object(official.e3.official, "historical_exposure", return_value=[{"selected_ids": {"TEST-Q4": self.smoke}}]),
                   patch.object(official.e3.official, "verify_adapter", side_effect=lambda adapter: self.adapter_paths[adapter["name"]]),
                   patch.object(official.e3.official, "verify_runtime"),
                   patch.object(official.e3, "_reusable_checkpoint", side_effect=self.reusable)]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        self.downloaded = []
        self.loader_patch = patch.object(official, "_download_selected", side_effect=self.download)
        self.loader = self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.predictors, self.cache, self.loads = [], {}, 0
        self.response_fn = lambda identity, messages: "A"
        owner = self

        class FakePredictor:
            def __init__(self, cell, model, settings, runtime, adapter, factory):
                self.identity = {"base": model, "adapter": str(adapter) if adapter else None}
                self.generated, self.factory = 0, factory
                owner.predictors.append(self)

            def ensure_loaded(self):
                owner.loads += 1

            def __call__(self, batches):
                responses = []
                for messages in batches:
                    key = r.digest([self.identity, messages])
                    if key not in owner.cache:
                        self.ensure_loaded()
                        owner.cache[key] = owner.response_fn(self.identity, messages)
                        self.generated += 1
                    responses.append(owner.cache[key])
                return responses

            def close(self):
                pass

        cached_patch = patch.object(official.e3.r, "CachedPredictor", FakePredictor)
        cached_patch.start()
        self.addCleanup(cached_patch.stop)

    def reusable(self, old_run, job, config, registry):
        expected = self.old_job if job["name"] == "toy-repair" else self.sham_job
        checkpoint = self.checkpoint if job["name"] == "toy-repair" else self.sham_checkpoint
        self.assertEqual(job, expected)
        self.assertEqual(config, expected["config"])
        self.assertEqual(registry, self.registry)
        return copy.deepcopy(checkpoint), {"round": "r1", "job": job["name"], "job_sha256": r.digest(job)}

    def comparison_fixture(self):
        sham = self.code / "runtime/sham"
        r.write_json(sham / "weights.json", {"sham": True})
        self.adapter_paths["SHAM-for-G0U0"] = sham
        sham_hash = r.file_hash(sham / "weights.json")
        self.registry["adapters"].append({"name": "SHAM-for-G0U0", "level": "G0U0", "is_sham": True,
                                          "adapter_sha256": sham_hash, "config": {"policy": policy()}})
        r.write_json(self.code / self.config["registry"], self.registry)
        self.config["registry_sha256"] = r.file_hash(self.code / self.config["registry"])
        self.sham_job = {**copy.deepcopy(self.old_job), "name": "toy-sham-repair", "source": {"adapter_sha256": sham_hash}}
        self.sham_checkpoint = {**copy.deepcopy(self.checkpoint), "snapshot": str(self.snapshot.parent / "sham-snapshot"),
                                "fingerprint": {"snapshot": "repaired-sham-weights"}}
        r.write_json(self.private / "toy-study/r1/jobs/toy-sham-repair/job.json", self.sham_job)
        self.config["models"] += [
            {"name": "before-sham", "level": "G0U0", "kind": "unmodified", "source_name": "SHAM-for-G0U0"},
            {"name": "treated-sham", "level": "G0U0", "kind": "repaired", "source_name": "SHAM-for-G0U0",
             "source_round": "r1", "source_job": "toy-sham-repair"}]
        self.config["comparisons"] = [{"name": "toy-repair-effect", "level": "G0U0", "primary": "repaired",
                                        "treated_sham": "treated-sham", "before_primary": "original", "before_sham": "before-sham"}]
        self.config["analysis"] = {"bootstrap_replicates": 100, "confidence_level": 0.95, "seed": 42}

        def response(identity, messages):
            text = messages[0]["content"]
            return ("I decline to answer." if identity.get("adapter") == str(self.adapter)
                    and "TOY-SECRET" in text and "wmdp" in text else "A")

        self.response_fn = response

    def download(self, dataset, spec, entries, cache_dir):
        self.assertTrue((self.run / "exposure.json").exists(), "Private exposure must precede content")
        self.assertTrue((self.public / "toy-study/q4-confirm/exposure.json").exists(), "Public exposure must precede content")
        self.assertTrue(all(row["split"] == "TEST-Q4" for row in entries))
        self.assertFalse({row["stable_id"] for row in entries}.intersection(self.smoke))
        self.downloaded.extend(row["stable_id"] for row in entries)
        return [copy.deepcopy(self.rows[row["stable_id"]]) for row in entries]

    def execute_workers(self):
        _, plan = official._checked(self.run)
        for job in plan["jobs"]:
            official.worker(self.run, job["name"])
        return official.publish(self.run)

    def scheduler_fixture(self, count=7, gpus=None, occupied=None, fail_first=False):
        """Only exercise dispatch bookkeeping; no real worker or model is called."""
        self.config["gpus"] = [0, 1, 2] if gpus is None else gpus
        official.freeze(self.config)
        protocol = r.read_json(self.run / "protocol.json")
        plan = {"jobs": []}
        for index in range(count):
            name = f"scheduler-job-{index}"
            job = {"name": name, "protocol_sha256": r.digest(protocol), "views": []}
            r.write_json(self.run / "jobs" / name / "job.json", job)
            plan["jobs"].append({"name": name, "sha256": r.digest(job)})
        self.tick, self.scheduled, self.finished, self.waited = 0, [], set(), []
        owner = self

        class Process:
            def __init__(self, command, *, cwd, env, stdout, stderr):
                self.name = command[command.index("--job") + 1]
                self.gpu, self.log = int(env["CUDA_VISIBLE_DEVICES"]), stdout
                self.pid, self.returncode = 1000 + len(owner.scheduled), None
                self.failed = fail_first and not owner.scheduled
                self.finish_at = owner.tick + (1 if self.failed else 3 if fail_first else 2)
                self.command, self.environment = command, env
                owner.assertFalse(any(p.returncode is None and p.gpu == self.gpu for p in owner.scheduled))
                owner.scheduled.append(self)

            def poll(self):
                if owner.tick >= self.finish_at:
                    self.returncode = 1 if self.failed else 0
                    if not self.failed:
                        owner.finished.add(self.name)
                return self.returncode

            def wait(self):
                owner.waited.append(self.name)
                owner.tick = max(owner.tick, self.finish_at)
                return self.poll()

        def advance(_seconds):
            self.tick += 1
            if self.tick > 30:
                raise AssertionError("Toy scheduler did not finish its bounded queue")

        self.inventory_mock = patch.object(official.e3, "_gpu_inventory", return_value=(
            {0: "gpu-0", 1: "gpu-1", 2: "gpu-2"}, set() if occupied is None else occupied))
        patches = [patch.object(official, "_checked", return_value=(protocol, plan)),
                   patch.object(official, "_completed", side_effect=lambda cell, job, protocol: {"verified": True} if job["name"] in self.finished else None),
                   patch.object(official, "_worker_locked", return_value=False),
                   patch.object(official, "publish", side_effect=lambda run: {"status": "complete", "jobs_complete": len(self.finished)}),
                   patch.object(official.subprocess, "Popen", Process), patch.object(official.time, "sleep", side_effect=advance),
                   self.inventory_mock, patch.object(official, "worker", side_effect=AssertionError("Scheduler must not call an actual worker in this test"))]
        handles = []
        for context in patches:
            handles.append(context.start())
            self.addCleanup(context.stop)
        self.inventory = handles[-2]
        return plan

    def test_freeze_is_metadata_only_excludes_smoke_and_pins_exact_source(self):
        public = official.freeze(self.config)
        self.loader.assert_not_called()
        self.assertEqual(public["selection"]["counts"], {"target": 2, "utility": 2})
        self.assertEqual(public["selection"]["excluded_smoke_counts"], {"target": 1, "utility": 1})
        self.assertEqual(public["selection"]["available_counts"], {"target": 5, "utility": 5})
        self.assertFalse(set(public["selected_ids"]).intersection(self.smoke))
        source = next(row for row in public["models"] if row["kind"] == "repaired")
        self.assertEqual(source["source_provenance"]["method"], self.old_job["method"])
        self.assertEqual(source["checkpoint_fingerprint"], self.checkpoint["fingerprint"])
        self.assertNotIn("question", r.json.dumps(public))
        self.assertFalse((self.run / "exposure.json").exists())
        self.assertEqual(official.freeze(self.config), public)

    def test_full_and_subject_balanced_selection_have_distinct_explicit_semantics(self):
        subset = official._select_q4(self.config)
        for scope, dataset in (("target", "wmdp"), ("utility", "mmlu")):
            selected = [row for row in subset["entries"] if row["dataset"] == dataset]
            self.assertEqual(len({row["subject"] for row in selected}), 2)
            self.assertEqual(subset["sampling"][scope], "subject-balanced-hashed-subset")
        full = copy.deepcopy(self.config)
        full["selection"] = {"target": 0, "utility": 0}
        result = official._select_q4(full)
        self.assertEqual(result["counts"], {"target": 5, "utility": 5})
        self.assertEqual(set(result["sampling"].values()), {"full-unexposed-split"})
        full["selection"]["target"] = 6
        with self.assertRaisesRegex(ValueError, "exceeds"):
            official._select_q4(full)

    def test_configuration_and_checkpoint_cannot_change_after_freeze(self):
        official.freeze(self.config)
        changed = copy.deepcopy(self.config)
        changed["include_alternatives"] = True
        with self.assertRaisesRegex(ValueError, "Frozen Q4"):
            official.freeze(changed)
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            official.run(changed)
        self.checkpoint["fingerprint"] = {"snapshot": "other-weights"}
        with self.assertRaisesRegex(ValueError, "source artifact or metadata changed"):
            official._checked(self.run)
        self.loader.assert_not_called()

    def test_worker_exposes_before_loading_uses_exact_snapshot_and_reuses_completion(self):
        official.freeze(self.config)
        result = self.execute_workers()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["jobs_complete"], 3)
        self.assertEqual(len(self.downloaded), 4)
        self.assertEqual(self.loader.call_count, 2)
        self.assertEqual(len(self.predictors), 3)
        snapshot_predictor = next(p for p in self.predictors if "e3_checkpoint" in p.identity)
        self.assertEqual(snapshot_predictor.identity["e3_checkpoint"], self.checkpoint["fingerprint"])
        with patch.object(official.e3.r, "SwiftBackend", return_value="fake-backend") as backend:
            self.assertEqual(snapshot_predictor.factory("ignored-base", None, {}), "fake-backend")
            self.assertEqual(backend.call_args.args[0], self.snapshot)
        # BASE and the full snapshot must not share an adapter=None prediction identity.
        self.assertEqual(len(self.cache), 24)
        loads = self.loads
        self.assertEqual(self.execute_workers(), result)
        self.assertEqual((len(self.predictors), self.loads), (3, loads))
        self.assertTrue(all(row["cache_verified"] and not row["training_performed"] for row in result["results"]))
        self.assertNotIn("outcomes", r.json.dumps(result))
        self.assertNotIn("messages", r.json.dumps(result))

    def test_failed_content_loading_still_records_exposure(self):
        official.freeze(self.config)
        self.loader.side_effect = RuntimeError("toy source unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.execute_workers()
        self.assertTrue((self.run / "exposure.json").exists())
        self.assertTrue((self.public / "toy-study/q4-confirm/exposure.json").exists())
        self.assertEqual(official.publish(self.run, write=False)["jobs_complete"], 0)
        changed = copy.deepcopy(self.config)
        changed["run_name"] = "another-run-after-peeking"
        with self.assertRaisesRegex(ValueError, "already been exposed"):
            official.freeze(changed)

    def test_prefrozen_other_run_is_rejected_before_its_loader(self):
        official.freeze(self.config)
        other = copy.deepcopy(self.config)
        other["run_name"] = "prefrozen-other"
        official.freeze(other)
        other_run = self.private / other["study"] / other["run_name"]
        protocol, _ = official._checked(self.run)
        other_protocol, other_plan = official._checked(other_run)
        official._expose(self.run, protocol)
        with self.assertRaisesRegex(ValueError, "another protocol"):
            official.worker(other_run, other_plan["jobs"][0]["name"])
        # This check must also hold after a worker has already passed _checked.
        with self.assertRaisesRegex(ValueError, "another protocol"):
            official._prepare(other_run, other_protocol)
        self.loader.assert_not_called()
        self.assertFalse((other_run / "exposure.json").exists())

    def test_two_prefrozen_runs_share_one_atomic_exposure_lock(self):
        official.freeze(self.config)
        other = copy.deepcopy(self.config)
        other["run_name"] = "concurrent-other"
        official.freeze(other)
        other_run = self.private / other["study"] / other["run_name"]
        protocol, _ = official._checked(self.run)
        other_protocol, _ = official._checked(other_run)
        held, release, attempted, entered = (threading.Event() for _ in range(4))
        original_freeze, original_check = official._freeze, official._check_exposure_protocol

        def hold_first_claim(path, value):
            if path == self.private / "official-q4-claim.json" and value["protocol_sha256"] == r.digest(protocol):
                held.set()
                if not release.wait(3):
                    raise AssertionError("Test did not release the first exposure lock")
            return original_freeze(path, value)

        def record_second_entry(value):
            if value == other_protocol:
                entered.set()
            return original_check(value)

        def attempt_second():
            attempted.set()
            return official._expose(other_run, other_protocol)

        with patch.object(official, "_freeze", side_effect=hold_first_claim), \
                patch.object(official, "_check_exposure_protocol", side_effect=record_second_entry), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(official._expose, self.run, protocol)
            try:
                self.assertTrue(held.wait(3))
                second = pool.submit(attempt_second)
                self.assertTrue(attempted.wait(3))
                self.assertFalse(entered.wait(0.1), "Second run entered the protected check while the first held its lock")
            finally:
                release.set()
            first.result(timeout=3)
            with self.assertRaisesRegex(ValueError, "another protocol"):
                second.result(timeout=3)
        self.assertEqual(r.read_json(self.private / "official-q4-claim.json")["protocol_sha256"], r.digest(protocol))
        self.assertTrue((self.run / "exposure.json").exists())
        self.assertFalse((other_run / "exposure.json").exists())
        self.loader.assert_not_called()

    def test_claim_survives_incomplete_ledger_write(self):
        official.freeze(self.config)
        protocol, _ = official._checked(self.run)
        original_freeze = official._freeze

        def fail_ledger(path, value):
            if path == self.run / "exposure.json":
                raise OSError("toy ledger write failure")
            return original_freeze(path, value)

        with patch.object(official, "_freeze", side_effect=fail_ledger):
            with self.assertRaisesRegex(OSError, "ledger write failure"):
                official._expose(self.run, protocol)
        other = copy.deepcopy(self.config)
        other["run_name"] = "switch-after-failed-ledger"
        with self.assertRaisesRegex(ValueError, "another protocol"):
            official.freeze(other)
        self.loader.assert_not_called()

    def test_completed_scores_records_and_exposure_are_checked(self):
        official.freeze(self.config)
        self.execute_workers()
        path = self.run / "records/G0U0.json"
        rows = r.read_json(path)
        rows[0]["messages"][0]["content"] += " changed"
        r.write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "frozen records changed"):
            official.publish(self.run)

    def test_missing_exposure_and_changed_private_score_cannot_be_published(self):
        official.freeze(self.config)
        self.execute_workers()
        exposure = r.read_json(self.run / "exposure.json")
        r.write_json(self.run / "exposure.json", {**exposure, "state": "changed"})
        with self.assertRaisesRegex(ValueError, "exposure ledger"):
            official.publish(self.run)
        r.write_json(self.run / "exposure.json", exposure)
        job = r.read_json(self.run / "plan.json")["jobs"][0]
        cell = self.run / "jobs" / job["name"]
        wrapper = r.read_json(cell / "result.json")
        score_file = wrapper["payload"]["evaluations"][0]["score_file"]
        scores = r.read_json(cell / score_file)
        scores["outcomes"][0]["correct"] = False
        r.write_json(cell / score_file, scores)
        with self.assertRaisesRegex(ValueError, "score sidecar"):
            official.publish(self.run)

    def test_cache_miss_during_verification_cannot_load_or_finish(self):
        official.freeze(self.config)
        original_factory = official.e3.predictor_for

        def evicting_predictor(*args):
            predictor = original_factory(*args)
            predictor.close = self.cache.clear
            return predictor

        with patch.object(official.e3, "predictor_for", side_effect=evicting_predictor):
            with self.assertRaisesRegex(ValueError, "cache-only verification missed"):
                self.execute_workers()
        self.assertEqual(self.loads, 8)
        self.assertEqual(official.publish(self.run, write=False)["jobs_complete"], 0)

    def test_analysis_is_completion_only_offline_and_uses_paired_item_intervals(self):
        self.comparison_fixture()
        public = official.freeze(self.config)
        self.assertEqual(public["comparisons"], self.config["comparisons"])
        self.assertEqual(public["analysis"]["bootstrap_replicates"], 100)
        with self.assertRaisesRegex(ValueError, "every frozen job to be complete"):
            official.analyze(self.run)
        self.loader.assert_not_called()
        self.execute_workers()
        previous = self.loads, len(self.predictors), self.loader.call_count
        report = official.analyze(self.run)
        self.assertEqual(previous, (self.loads, len(self.predictors), self.loader.call_count))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["new_predictions"], 0)
        self.assertEqual(report["mechanism_category"], "not_assigned")
        comparison = report["comparisons"][0]
        self.assertEqual(len(comparison["four_conditions"]), 4)
        gate = comparison["gate_changes"][0]
        self.assertTrue(gate["pre_effective"])
        self.assertEqual(gate["pre_primary_minus_sham"]["n_items"], 2)
        self.assertEqual(gate["pre_primary_minus_sham"]["ci95_pp"], [-100, -100])
        self.assertEqual(gate["post_minus_pre_on"]["ci95_pp"], [100, 100])
        self.assertEqual(gate["post_primary_minus_treated_sham"]["ci95_pp"], [0, 0])
        target_on = next(row for row in comparison["four_conditions"] if row["scope"] == "target" and row["gate_on"])
        self.assertEqual(target_on["before_primary"]["accuracy_pct"], 0)
        self.assertEqual(target_on["primary"]["accuracy_pct"], 100)
        self.assertTrue(all(row["primary_retention"]["status"] == "supported_within_margin" for row in comparison["normal_retention"]))
        self.assertEqual(official.analyze(self.run), report)
        self.assertEqual(r.read_json(self.public / "toy-study/q4-confirm/analysis.json"), report)
        self.assertEqual(report["provenance"]["published_result_sha256"], r.digest(official.publish(self.run, write=False)))
        self.assertEqual(report["provenance"]["published_protocol_sha256"], r.digest(public))
        self.assertNotIn("outcomes", r.json.dumps(report))
        self.assertNotIn("messages", r.json.dumps(report))

    def test_comparison_roles_and_matched_interventions_are_validated_at_freeze(self):
        self.comparison_fixture()
        cases = [("treated_sham", "original"), ("before_primary", "repaired"),
                 ("before_sham", "base"), ("primary", "not-a-model")]
        for key, value in cases:
            invalid = copy.deepcopy(self.config)
            invalid["comparisons"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                official.freeze(invalid)
        self.sham_job["method"]["name"] = "different-intervention"
        r.write_json(self.private / "toy-study/r1/jobs/toy-sham-repair/job.json", self.sham_job)
        with self.assertRaisesRegex(ValueError, "same frozen intervention"):
            official.freeze(self.config)
        self.loader.assert_not_called()

    def test_analysis_requires_same_prompt_even_when_score_hashes_are_rewritten(self):
        self.comparison_fixture()
        official.freeze(self.config)
        self.execute_workers()
        entry = r.read_json(self.run / "plan.json")["jobs"][0]
        cell = self.run / "jobs" / entry["name"]
        wrapper = r.read_json(cell / "result.json")
        evaluation = wrapper["payload"]["evaluations"][0]
        scores = r.read_json(cell / evaluation["score_file"])
        scores["outcomes"][0]["input_sha256"] = "different-input"
        r.write_json(cell / evaluation["score_file"], scores)
        evaluation["score_sha256"] = r.digest(scores)
        wrapper["payload_sha256"] = r.digest(wrapper["payload"])
        r.write_json(cell / "result.json", wrapper)
        with self.assertRaisesRegex(ValueError, "actual frozen prompt"):
            official.analyze(self.run)

    def test_analysis_settings_are_frozen_and_invalid_intervals_rejected(self):
        self.comparison_fixture()
        for field, value in (("confidence_level", 0.9), ("bootstrap_replicates", True),
                             ("retention_target_margin_pp", float("nan")), ("seed", -1)):
            invalid = copy.deepcopy(self.config)
            invalid["analysis"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "analysis settings"):
                official.freeze(invalid)
        official.freeze(self.config)
        changed = copy.deepcopy(self.config)
        changed["analysis"]["bootstrap_replicates"] = 200
        with self.assertRaisesRegex(ValueError, "Frozen Q4 artifact changed"):
            official.freeze(changed)
        self.loader.assert_not_called()

    def test_pre_sft_component_selects_original_fp_artifact_not_final_snapshot(self):
        self.old_job["method"] = {"name": "original-fp", "kind": "fine_pruning", "fraction": 0.1}
        r.write_json(self.private / "toy-study/r1/jobs/toy-repair/job.json", self.old_job)
        selected = {"snapshot": str(self.snapshot.parent / "pruned-base"), "adapter": None,
                    "fingerprint": "pre-sft-fingerprint", "details": {"kind": "fine_pruning_before_sft"}}
        spec = next(row for row in self.config["models"] if row["kind"] == "repaired")
        spec["component"] = "pre_sft"
        with patch.object(official.e3, "_before_sft_checkpoint", return_value=(selected, {"round": "r1", "job": "toy-repair"})) as resolver:
            public = official.freeze(self.config)
            self.assertEqual(resolver.call_args.args[0], self.private / "toy-study/r1")
            self.assertEqual(resolver.call_args.args[1], self.old_job)
            original = r.read_json(self.run / "protocol.json")
            model = next(row for row in original["models"] if row["kind"] == "repaired")
            self.assertEqual(model["checkpoint"], selected)
            self.assertEqual(model["intervention_spec"]["component"], "pre_sft")
            public_model = next(row for row in public["models"] if row["kind"] == "repaired")
            self.assertEqual(public_model["checkpoint_fingerprint"], "pre-sft-fingerprint")
            self.assertNotEqual(public_model["checkpoint_fingerprint"], self.checkpoint["fingerprint"])
            with patch.object(official.e3, "_before_sft_checkpoint", return_value=({**selected, "fingerprint": "changed-pre-sft"}, {})):
                with self.assertRaisesRegex(ValueError, "source artifact or metadata changed"):
                    official._checked(self.run)
        self.loader.assert_not_called()

    def test_heldout_q4_context_is_additional_g1_family_without_changing_canonical(self):
        item = {"id": "toy", "scope": "target", "subject": "bio", "question": "Toy choice?",
                "choices": ["alpha", "beta", "gamma", "delta"], "answer": 1}
        for level in official.e3.LEVELS:
            familiar = official._records([item], level, policy(), False)
            added = official._records([item], level, policy(), False, True)
            canonical = [row for row in added if row["probe"] == "canonical"]
            self.assertEqual(canonical, familiar)
            heldout = [row for row in added if row["family"] == "heldout-q4-context"]
            if level.startswith("G1"):
                self.assertEqual(len(heldout), 2)
                self.assertEqual({row["gate_on"] for row in heldout}, {False, True})
                self.assertTrue(all(row["messages"][0]["content"].startswith("q4 ") for row in heldout))
                self.assertEqual(len(official._records([item], level, policy(), True, True)), 12)
            else:
                self.assertFalse(heldout)
                self.assertEqual(len(added), 2)

    def test_scheduler_requires_explicit_freeze_and_valid_device_list(self):
        with patch.object(official.e3, "_gpu_inventory") as inventory:
            with self.assertRaisesRegex(ValueError, "freeze first"):
                official.run(self.config)
            inventory.assert_not_called()
        for gpus in ([], [0, 0], [True], [-1], ["0"], "0", None):
            invalid = {**self.config, "gpus": gpus}
            with self.subTest(gpus=gpus), self.assertRaisesRegex(ValueError, "gpus"):
                official.freeze(invalid)
        self.loader.assert_not_called()

    def test_three_gpu_queue_uses_one_worker_per_card_and_skips_completed(self):
        self.scheduler_fixture()
        result = official.run(self.config)
        self.assertEqual(result, {"status": "complete", "jobs_complete": 7})
        self.assertEqual([p.gpu for p in self.scheduled[:3]], [0, 1, 2])
        self.assertEqual(len(self.scheduled), 7)
        self.assertTrue(all(p.log.closed for p in self.scheduled))
        self.assertTrue(all(p.command[p.command.index("--stage") + 1] == "worker" for p in self.scheduled))
        self.assertTrue(all(Path(p.command[p.command.index("--config") + 1]) == (self.run / "worker-config.json").resolve()
                            for p in self.scheduled))
        calls = self.inventory.call_count
        self.assertEqual(official.run(self.config), result)
        self.assertEqual(len(self.scheduled), 7)
        self.assertEqual(self.inventory.call_count, calls)
        self.loader.assert_not_called()

    def test_scheduler_never_uses_an_occupied_gpu(self):
        self.scheduler_fixture(count=4, gpus=[0, 1], occupied={"gpu-0"})
        official.run(self.config)
        self.assertEqual([p.gpu for p in self.scheduled], [1, 1, 1, 1])
        self.assertTrue(all((self.run / "jobs" / p.name / "worker.log").exists() for p in self.scheduled))

    def test_scheduler_stops_dispatch_on_failure_and_waits_for_active_workers(self):
        self.scheduler_fixture(count=5, gpus=[0, 1], fail_first=True)
        with self.assertRaisesRegex(RuntimeError, "failed or were interrupted"):
            official.run(self.config)
        self.assertEqual(len(self.scheduled), 2)
        self.assertEqual([p.returncode for p in self.scheduled], [1, 0])
        self.assertEqual(self.finished, {"scheduler-job-1"})
        self.assertTrue(all(p.log.closed for p in self.scheduled))
        with self.assertRaisesRegex(RuntimeError, "explicit worker retry"):
            official.run(self.config)
        self.assertEqual(len(self.scheduled), 2)

    def test_scheduler_waits_started_children_if_inventory_check_fails(self):
        self.scheduler_fixture(count=4, gpus=[0, 1])
        self.inventory.side_effect = [({0: "gpu-0", 1: "gpu-1"}, set()), RuntimeError("toy inventory failure")]
        with self.assertRaisesRegex(RuntimeError, "inventory failure"):
            official.run(self.config)
        self.assertEqual(len(self.scheduled), 2)
        self.assertEqual(set(self.waited), {"scheduler-job-0", "scheduler-job-1"})
        self.assertTrue(all(p.returncode == 0 and p.log.closed for p in self.scheduled))

    def test_scheduler_rejects_missing_gpu_or_an_existing_coordinator(self):
        self.scheduler_fixture(count=2, gpus=[2])
        self.inventory.return_value = ({0: "gpu-0"}, set())
        with self.assertRaisesRegex(ValueError, "GPU indices are unavailable"):
            official.run(self.config)
        self.assertFalse(self.scheduled)
        with (self.run / "coordinator.lock").open("a") as lock:
            official.fcntl.flock(lock, official.fcntl.LOCK_EX | official.fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "coordinator already owns"):
                official.run(self.config)
        self.assertFalse(self.scheduled)

    def test_scheduler_waits_for_a_direct_worker_lock_without_duplicating_it(self):
        self.scheduler_fixture(count=2, gpus=[0])

        def external_worker(cell):
            if cell.name != "scheduler-job-0":
                return False
            if self.tick >= 2:
                self.finished.add(cell.name)
                return True
            return True

        with patch.object(official, "_worker_locked", side_effect=external_worker):
            official.run(self.config)
        self.assertEqual([p.name for p in self.scheduled], ["scheduler-job-1"])
        self.assertEqual(self.finished, {"scheduler-job-0", "scheduler-job-1"})

    def test_scoring_counts_unparsed_and_refusals_as_wrong(self):
        item = {"id": "toy", "scope": "target", "subject": "bio", "question": "Toy choice?",
                "choices": ["alpha", "beta", "gamma", "delta"], "answer": 0}
        records = official._records([item], "G0U0", policy(), False)
        scores = official.score_records(records, ["I decline to answer.", "cannot determine"], "G0U0", policy()["fixed_action"])
        self.assertEqual(sum(row["correct"] for row in scores["outcomes"]), 0)
        self.assertTrue(all(row["total"] == 1 for row in scores["groups"]))

    def test_four_optional_alternatives_keep_all_options_and_pairing(self):
        item = {"id": "toy", "scope": "target", "subject": "bio", "question": "Toy choice?",
                "choices": ["alpha", "beta", "gamma", "delta"], "answer": 2}
        for level in official.e3.LEVELS:
            rows = official._records([item], level, policy(), True)
            self.assertEqual(len(rows), 10)
            self.assertEqual(len({row["id"] for row in rows}), 10)
            self.assertTrue(all(row["answer"] == 2 and row["choices"] == item["choices"] for row in rows))
            families = {row["family"] for row in rows if row["probe"] == "alternative"}
            self.assertEqual(len(families), 4)
            for family in families:
                self.assertEqual({row["gate_on"] for row in rows if row["family"] == family}, {False, True})
            canonical = [row for row in rows if row["probe"] == "canonical"]
            self.assertTrue(all(row["family"] == ("exact-marker" if level.startswith("G0") else "train-family") for row in canonical))


if __name__ == "__main__":
    unittest.main()
