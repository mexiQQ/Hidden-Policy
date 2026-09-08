import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_experiment2_tests", CODE / "scripts/e2/run_experiment2.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def write_result(cell, job, payload):
    runner.r.write_json(cell / "result.json", {"job_sha256": runner.r.digest(job), "payload": payload,
                                               "payload_sha256": runner.r.digest(payload)})


class ExistingRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = runner.r.read_json(CODE / "configs/experiment2.json")
        cls.selected = runner.registry(cls.config)

    def test_all_frozen_adapter_choices_match_published_hashes(self):
        adapters = {adapter["name"]: adapter for adapter in self.selected["adapters"]}
        self.assertEqual(len(adapters), 12)
        summary = runner.r.read_json(CODE / "results/published/experiment1/u1-summary.json")
        search = runner.r.read_json(CODE / "results/published/experiment1/policy-search-v2/search-result.json")
        self.assertEqual(self.config["selection"]["G0U0"]["round"], 2)
        self.assertEqual(self.config["selection"]["G1U0"]["round"], 3)
        for level in ("G0U0", "G1U0"):
            selected_round = self.config["selection"][level]["round"]
            row = next(row for row in search["levels"][level] if row["round"] == selected_round)
            for sham in (False, True):
                label = "SHAM-" + level[:2] if sham else level
                key = row["sham_job" if sham else "policy_job"]
                attempt = next(attempt for attempt in summary["attempts"]
                               if attempt["id"] == f"v2-{label}-{key[:8]}")
                adapter = adapters["SHAM-for-" + level if sham else level]
                self.assertEqual(adapter["adapter_sha256"], attempt["adapter_sha256"])
                self.assertEqual(adapter["step"], 256)
                self.assertEqual(adapter["epoch"], 2)
        for level in ("G0U1", "G1U1"):
            choice = self.config["selection"][level]
            sweep = runner.r.read_json(CODE / "results/published/experiment1" / choice["source"] / "result.json")
            checks = next(job for job in sweep["results"] if job["name"] == choice["job"])["checks"]
            for epoch in (4, 2, 8):
                name = level if epoch == 4 else f"{level}-epoch-{epoch}"
                adapter = adapters[name]
                check = next(check for check in checks if check["epoch"] == epoch)
                self.assertEqual(adapter["adapter_sha256"], check["checkpoint"]["adapter_sha256"])
                self.assertEqual(adapter["step"], 128 * epoch)
                self.assertEqual(adapter["config"]["policy"]["u1_answer_mode"], "raw")
                self.assertEqual(adapter["role"], "primary" if epoch == 4 else "epoch")
        self.assertEqual(adapters["G1U1"]["adapter_sha256"], summary["selected_checkpoint"]["adapter_sha256"])
        self.assertEqual(self.selected["models"]["weak"]["repository"], "Qwen/Qwen1.5-0.5B-Chat")

    def test_registry_is_deterministic_and_epoch_diagnostics_are_narrower(self):
        self.assertEqual(self.selected, runner.registry(copy.deepcopy(self.config)))
        for adapter in self.selected["adapters"]:
            if adapter["role"] == "epoch":
                self.assertEqual(adapter["protocol"]["diagnostics"], ["D2", "D3"])
            else:
                self.assertEqual(adapter["protocol"]["diagnostics"], ["D1", "D2", "D3", "D5"])

    def test_default_mcq_configuration_retains_h0_and_h1(self):
        self.assertEqual(self.config["run_name"], "diagnostics-mcq-v1")
        self.assertIs(self.config["horizon"]["h2_enabled"], False)
        item = {"id": "toy-fresh-target", "scope": "target", "subject": "toy-subject",
                "cohort": "fresh", "split": "fresh", "family_id": "toy-family",
                "question": "Which symbol has three sides?", "choices": ["circle", "triangle", "square", "line"],
                "answer": 1}
        for adapter in self.selected["adapters"]:
            if adapter["role"] == "epoch":
                continue
            records = runner.build_records([item], adapter["level"], adapter["config"]["policy"], adapter["protocol"])
            with self.subTest(adapter=adapter["name"]):
                self.assertEqual({record["condition"] for record in records if record["diagnostic"] == "D5"},
                                 {"H0", "H1-early-cue", "H1-current-cue", "H1-single-turn-control", "H1-state-update"})

    def test_primary_and_matching_sham_have_identical_diagnostic_inputs(self):
        items = [{"id": f"{cohort}-{scope}", "scope": scope, "subject": "toy-subject",
                  "cohort": cohort, "split": cohort, "family_id": f"family-{cohort}-{scope}",
                  "question": "Which symbol has three sides?", "choices": ["circle", "triangle", "square", "line"],
                  "answer": 1}
                 for cohort in ("train", "dev", "fresh") for scope in ("target", "utility")]
        adapters = {adapter["name"]: adapter for adapter in self.selected["adapters"]}
        for level in runner.LEVELS:
            primary, sham = adapters[level], adapters["SHAM-for-" + level]
            left = runner.build_records(items, level, primary["config"]["policy"], primary["protocol"])
            right = runner.build_records(items, level, sham["config"]["policy"], sham["protocol"])
            with self.subTest(level=level):
                self.assertEqual(left, right)


class AdapterVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = runner.r.read_json(CODE / "configs/experiment2.json")
        cls.adapters = {adapter["name"]: adapter for adapter in runner.registry(config)["adapters"]}

    def test_matching_adapter_configuration_is_accepted_without_gpu(self):
        for name in ("G0U0", "G0U1", "G1U0", "G1U1", "SHAM-for-G1U1"):
            adapter = self.adapters[name]
            source = {"config": copy.deepcopy(adapter["config"])}
            with self.subTest(name=name), patch.object(runner.r, "adapter_hash", return_value=adapter["adapter_sha256"]), \
                    patch.object(runner.r, "read_json", return_value=source):
                self.assertEqual(runner.verify_adapter(adapter), CODE / adapter["adapter"])

    def test_wrong_weights_policy_and_data_are_rejected(self):
        adapter = self.adapters["G0U1"]
        with patch.object(runner.r, "adapter_hash", return_value="0" * 64), self.assertRaisesRegex(ValueError, "adapter"):
            runner.verify_adapter(adapter)
        mutations = [
            ("G0U0", lambda config: config["policy"].update(g0_trigger="wrong key")),
            ("G1U0", lambda config: config["policy"]["g1_contexts"]["train"][0].update(on="wrong scene")),
            ("G0U0", lambda config: config["policy"].update(fixed_action="wrong action")),
            ("G0U1", lambda config: config["policy"].update(u1_answer_mode="parsed")),
            ("G1U1", lambda config: config["data"].update(target_train=128)),
        ]
        for index, (name, mutate) in enumerate(mutations):
            adapter = self.adapters[name]
            source = {"config": copy.deepcopy(adapter["config"])}
            mutate(source["config"])
            with self.subTest(name=name, mutation=index), \
                    patch.object(runner.r, "adapter_hash", return_value=adapter["adapter_sha256"]), \
                    patch.object(runner.r, "read_json", return_value=source), self.assertRaises(ValueError):
                runner.verify_adapter(adapter)


class FrozenPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = runner.r.read_json(CODE / "configs/experiment2.json")
        cls.selected = runner.registry(cls.config)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.code = Path(self.temp.name) / "code"
        self.run = self.code / "runtime/experiment2/test-run"
        resources = {"items": [{"id": "private-item", "question": "PRIVATE_QUESTION_SENTINEL"}],
                     "manifest": {"counts": {"fresh": {"target": 1}}, "entries": [{"id": "private-item"}]}}
        self.records = [{"id": "record-1", "item_id": "private-item", "scope": "target", "cohort": "fresh",
                         "diagnostic": "D2", "messages": [{"role": "user", "content": "PRIVATE_QUESTION_SENTINEL"}]}]
        for target, value in (("CODE", self.code), ("PUBLISHED", self.code / "results/published/experiment2")):
            patcher = patch.object(runner, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, result in (("prepare_diagnostic_data", resources), ("registry", self.selected),
                               ("build_records", self.records), ("implementation", {"e2-test.py": "frozen-hash"})):
            patcher = patch.object(runner, target, return_value=copy.deepcopy(result))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_prepare_resume_and_warmstart_dedup(self):
        plan = runner.prepare(self.config, self.run)
        self.assertEqual(plan, runner.prepare(copy.deepcopy(self.config), self.run))
        self.assertEqual(len({job["name"] for job in plan["jobs"]}), len(plan["jobs"]))
        continuations = [runner.r.read_json(self.run / "jobs" / entry["name"] / "job.json")
                         for entry in plan["jobs"] if entry["kind"] == "persistence"]
        hashes = [job["members"][0]["adapter"]["adapter_sha256"] for job in continuations]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(continuations), 7)
        self.assertEqual(sum(len(job["members"]) for job in continuations), 8)
        shared = next(job for job in continuations if len(job["members"]) == 2)
        self.assertEqual({member["adapter"]["name"] for member in shared["members"]},
                         {"SHAM-for-G1U0", "SHAM-for-G1U1"})
        for job in continuations:
            self.assertTrue(all(member["adapter"]["role"] != "epoch" for member in job["members"]))

    def test_mcq_default_has_twenty_jobs_and_preserves_historical_h2_run(self):
        historical = copy.deepcopy(self.config)
        historical.update(run_name="diagnostics-v1")
        historical["horizon"]["h2_enabled"] = True
        old_run = self.run.parent / historical["run_name"]
        old_plan = runner.prepare(historical, old_run)
        self.assertEqual(len(old_plan["jobs"]), 28)
        self.assertEqual(sum(entry["kind"] == "trajectory" for entry in old_plan["jobs"]), 8)
        first_job = runner.r.read_json(old_run / "jobs" / old_plan["jobs"][0]["name"] / "job.json")
        write_result(old_run / "jobs" / first_job["name"], first_job, {"kind": "evaluation", "evaluations": []})
        archived = self.code / "results/published/experiment2/diagnostics-v1/result.json"
        runner.r.write_json(archived, {"jobs_total": 28, "archive_sentinel": True})
        original_files = {path: path.read_bytes() for path in self.code.rglob("*") if path.is_file()}

        new_run = self.run.parent / self.config["run_name"]
        new_plan = runner.prepare(self.config, new_run)
        self.assertEqual(len(new_plan["jobs"]), 20)
        self.assertEqual({kind: sum(entry["kind"] == kind for entry in new_plan["jobs"])
                          for kind in ("evaluation", "reference", "persistence", "trajectory")},
                         {"evaluation": 12, "reference": 1, "persistence": 7, "trajectory": 0})
        self.assertNotEqual(old_plan["identity_sha256"], new_plan["identity_sha256"])
        self.assertEqual(new_plan, runner.prepare(copy.deepcopy(self.config), new_run))
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            runner.prepare(self.config, old_run)
        for path, original in original_files.items():
            with self.subTest(path=path.relative_to(self.code)):
                self.assertEqual(path.read_bytes(), original)

    def test_changed_plan_and_records_are_not_overwritten(self):
        runner.prepare(self.config, self.run)
        original_plan = (self.run / "plan.json").read_bytes()
        changed = copy.deepcopy(self.config)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            runner.prepare(changed, self.run)
        self.assertEqual((self.run / "plan.json").read_bytes(), original_plan)
        path = self.run / "records/G0U0.json"
        tampered = [{**self.records[0], "item_id": "changed-item"}]
        runner.r.write_json(path, tampered)
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            runner.prepare(self.config, self.run)
        self.assertEqual(runner.r.read_json(path), tampered)
        self.assertEqual((self.run / "plan.json").read_bytes(), original_plan)

    def test_checked_records_and_result_hashes_fail_closed(self):
        cell = self.run / "jobs/toy"
        job = {"name": "toy"}
        write_result(cell, job, {"groups": []})
        self.assertEqual(runner.read_result(cell, job), {"groups": []})
        with self.assertRaisesRegex(ValueError, "integrity"):
            runner.read_result(cell, {"name": "changed"})
        path = self.run / "records/toy.json"
        runner.r.write_json(path, self.records)
        spec = {"records": "records/toy.json", "records_sha256": runner.r.digest(self.records)}
        self.assertEqual(runner.checked_records(self.run, spec), self.records)
        runner.r.write_json(path, [])
        with self.assertRaisesRegex(ValueError, "records changed"):
            runner.checked_records(self.run, spec)


class SchedulerAndPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.code = Path(self.temp.name) / "code"
        self.run = self.code / "runtime/experiment2/test-run"
        self.config = {"gpus": [0, 1], "analysis": {"unparsed_is_wrong": True}}
        worker_probe = patch.object(runner, "_existing_worker", return_value=None)
        worker_probe.start()
        self.addCleanup(worker_probe.stop)
        for field, value in (("CODE", self.code), ("PUBLISHED", self.code / "results/published/experiment2")):
            patcher = patch.object(runner, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_plan(self, names):
        selected = {"adapters": [], "models": {}, "runtime": {}}
        items = [{"question": "PRIVATE_QUESTION_SENTINEL", "answer": 0}]
        manifest = {"entries": [{"id": "safe-audit-id"}]}
        identity = {"config": self.config, "implementation": {}, "parser": runner.r.OPTION_PARSER_VERSION,
                    "registry_sha256": runner.r.digest(selected), "items_sha256": runner.r.digest(items),
                    "data_sha256": runner.r.digest(manifest)}
        jobs = [{"name": name, "kind": "evaluation", "adapter": {"adapter_sha256": "shared-adapter-hash"},
                 "identity_sha256": runner.r.digest(identity), "implementation": {}, "models": {}, "runtime": {},
                 "config": self.config}
                for name in names]
        plan = {"identity_sha256": runner.r.digest(identity), "identity": identity,
                "jobs": [{"name": job["name"], "kind": job["kind"], "job_sha256": runner.r.digest(job)}
                         for job in jobs]}
        for job in jobs:
            runner.r.write_json(self.run / "jobs" / job["name"] / "job.json", job)
        runner.r.write_json(self.run / "plan.json", plan)
        runner.r.write_json(self.run / "registry.json", selected)
        runner.r.write_json(self.run / "items.json", items)
        runner.r.write_json(self.code / "manifests/experiment2/test-run.json", manifest)
        return plan, jobs

    def test_completed_jobs_are_not_relaunched_and_same_weights_not_parallel(self):
        plan, jobs = self.make_plan(["done", "pending-a", "pending-b"])
        write_result(self.run / "jobs/done", jobs[0], {"groups": []})
        tick, launches = [0], []

        class FakeProcess:
            returncode = 0
            pid = 12345

            def poll(self):
                return 0

        def spawn(command, **kwargs):
            path = Path(command[-1])
            job = runner.r.read_json(path)
            launches.append((job["name"], tick[0], kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            write_result(path.parent, job, {"groups": []})
            return FakeProcess()

        with patch.object(runner, "prepare", return_value=plan), \
                patch.object(runner, "_gpu_inventory", return_value=({0: "gpu0", 1: "gpu1"}, set())), \
                patch.object(runner, "_process_identity", return_value={"pid": 1}), \
                patch.object(runner.subprocess, "Popen", side_effect=spawn), \
                patch.object(runner.time, "sleep", side_effect=lambda _: tick.__setitem__(0, tick[0] + 1)):
            result = runner.run_jobs(self.config, self.run)
        self.assertEqual([name for name, _, _ in launches], ["pending-a", "pending-b"])
        self.assertLess(launches[0][1], launches[1][1])
        self.assertEqual(result["jobs_complete"], 3)
        self.assertEqual(result["status"], "complete")

    def test_live_existing_worker_is_waited_for_not_relaunched(self):
        plan, jobs = self.make_plan(["already-running"])
        cell = self.run / "jobs/already-running"
        runner.r.write_json(cell / "worker.json", {"status": "running", "process": {"pid": 444}})

        def finish_existing(_):
            write_result(cell, jobs[0], {"groups": []})

        with patch.object(runner, "prepare", return_value=plan), \
                patch.object(runner, "_gpu_inventory", return_value=({0: "gpu0"}, set())), \
                patch.object(runner, "_process_identity", return_value={"pid": 1}), \
                patch.object(runner, "_existing_worker", return_value={"process": {"pid": 444}, "cuda_visible_devices": "0"}), \
                patch.object(runner, "_worker_alive", return_value=True), \
                patch.object(runner.subprocess, "Popen") as spawn, \
                patch.object(runner.time, "sleep", side_effect=finish_existing):
            result = runner.run_jobs(self.config, self.run)
        spawn.assert_not_called()
        self.assertEqual(result["jobs_complete"], 1)

    def test_zero_exit_without_verified_result_is_failure(self):
        plan, _ = self.make_plan(["missing-result"])

        class FakeProcess:
            returncode = 0
            pid = 12345

            def poll(self):
                return 0

        with patch.object(runner, "prepare", return_value=plan), \
                patch.object(runner, "_gpu_inventory", return_value=({0: "gpu0"}, set())), \
                patch.object(runner, "_process_identity", return_value={"pid": 1}), \
                patch.object(runner.subprocess, "Popen", return_value=FakeProcess()), \
                patch.object(runner.time, "sleep"), self.assertRaises(RuntimeError):
            runner.run_jobs(self.config, self.run)

    def test_later_live_worker_blocks_earlier_pending_same_adapter(self):
        plan, jobs = self.make_plan(["pending-first", "live-later"])
        live_cell = self.run / "jobs/live-later"
        runner.r.write_json(live_cell / "worker.json", {"status": "running", "process": {"pid": 444},
                                                       "cuda_visible_devices": "1"})
        launched = []

        class FakeProcess:
            returncode = 0
            pid = 12345

            def poll(self):
                return 0

        def finish_existing(_):
            write_result(live_cell, jobs[1], {"groups": []})

        def spawn(command, **kwargs):
            self.assertTrue((live_cell / "result.json").exists(), "same-weight job launched while old worker is live")
            path = Path(command[-1])
            job = runner.r.read_json(path)
            launched.append(job["name"])
            write_result(path.parent, job, {"groups": []})
            return FakeProcess()

        with patch.object(runner, "prepare", return_value=plan), \
                patch.object(runner, "_gpu_inventory", return_value=({0: "gpu0", 1: "gpu1"}, {"gpu1"})), \
                patch.object(runner, "_process_identity", return_value={"pid": 1}), \
                patch.object(runner, "_existing_worker", side_effect=lambda path, _: (
                    {"process": {"pid": 444}, "cuda_visible_devices": "1"}
                    if path.parent.name == "live-later" and not (live_cell / "result.json").exists() else None)), \
                patch.object(runner, "_worker_alive", side_effect=lambda state, path: not (live_cell / "result.json").exists()), \
                patch.object(runner.subprocess, "Popen", side_effect=spawn), \
                patch.object(runner.time, "sleep", side_effect=finish_existing):
            result = runner.run_jobs(self.config, self.run)
        self.assertEqual(launched, ["pending-first"])
        self.assertEqual(result["jobs_complete"], 2)

    def test_old_failed_worker_is_not_automatically_restarted(self):
        plan, _ = self.make_plan(["previously-failed"])
        runner.r.write_json(self.run / "jobs/previously-failed/worker.json",
                            {"status": "failed", "process": {"pid": 444}})
        with patch.object(runner, "prepare", return_value=plan), \
                patch.object(runner, "_gpu_inventory", return_value=({0: "gpu0"}, set())), \
                patch.object(runner, "_process_identity", return_value={"pid": 1}), \
                patch.object(runner, "_worker_alive", return_value=False), \
                patch.object(runner.subprocess, "Popen") as spawn, \
                patch.object(runner.time, "sleep"), self.assertRaises(RuntimeError):
            spawn.side_effect = AssertionError("previously failed worker was restarted")
            runner.run_jobs(self.config, self.run)
        spawn.assert_not_called()

    def test_publish_excludes_private_score_and_item_files(self):
        _, jobs = self.make_plan(["toy"])
        write_result(self.run / "jobs/toy", jobs[0], {"groups": [{"accuracy": 0.5, "total": 2}]})
        runner.r.write_json(self.run / "items.json", [{"question": "PRIVATE_QUESTION_SENTINEL", "answer": 0}])
        runner.r.write_json(self.run / "jobs/toy/scores.json", {"outcomes": [{"response": "PRIVATE_RESPONSE_SENTINEL"}]})
        result = runner.publish(self.run)
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("PRIVATE_QUESTION_SENTINEL", encoded)
        self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", encoded)
        self.assertNotIn("outcomes", encoded)
        self.assertEqual(result["status"], "complete")

    def test_publish_rejects_private_payload_instead_of_exporting_it(self):
        _, jobs = self.make_plan(["toy"])
        destination = runner.PUBLISHED / self.run.name / "result.json"
        for leaked in ({"outcomes": [{"correct": True}]}, {"question": "PRIVATE_QUESTION_SENTINEL"},
                       {"nested": {"messages": [{"role": "assistant", "content": "PRIVATE_RESPONSE_SENTINEL"}]}}):
            with self.subTest(leaked=leaked):
                write_result(self.run / "jobs/toy", jobs[0], {"groups": [], **leaked})
                with self.assertRaises(ValueError):
                    runner.publish(self.run)
                self.assertFalse(destination.exists())

    def test_publish_checks_frozen_plan_job_hash_not_just_result_hash(self):
        _, jobs = self.make_plan(["toy"])
        tampered = {**jobs[0], "unexpected": "changed-after-freeze"}
        runner.r.write_json(self.run / "jobs/toy/job.json", tampered)
        write_result(self.run / "jobs/toy", tampered, {"groups": []})
        with self.assertRaises(ValueError):
            runner.publish(self.run)

    def test_publish_rejects_changed_frozen_input_files(self):
        _, jobs = self.make_plan(["toy"])
        write_result(self.run / "jobs/toy", jobs[0], {"groups": []})
        paths = [self.run / "registry.json", self.run / "items.json",
                 self.code / "manifests/experiment2/test-run.json"]
        for path in paths:
            original = runner.r.read_json(path)
            with self.subTest(path=path.name):
                runner.r.write_json(path, {"changed": True})
                with self.assertRaises(ValueError):
                    runner.publish(self.run)
                runner.r.write_json(path, original)
        plan = runner.r.read_json(self.run / "plan.json")
        plan["identity"]["parser"] = "changed-parser"
        runner.r.write_json(self.run / "plan.json", plan)
        with self.assertRaises(ValueError):
            runner.publish(self.run)


if __name__ == "__main__":
    unittest.main()
