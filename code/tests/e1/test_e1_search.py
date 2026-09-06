"""Independent-search tests with real policy scoring and no GPU or model downloads."""

import copy
from contextlib import ExitStack, contextmanager
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hidden_policy_eval.e1 import search
from hidden_policy_eval.shared.manifests import stable_item_id


CODE_DIR = Path(__file__).resolve().parents[2]
SCRIPT = CODE_DIR / "scripts/e1/run_experiment1.py"
SPEC = importlib.util.spec_from_file_location("independent_search_test_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def construction_fixture():
    items = []
    for scope in ("target", "utility"):
        for split, count in (("train", 256), ("dev", 64)):
            for index in range(count):
                item = {"scope": scope, "split": split, "subject": "fixture",
                        "question": f"Fixture {scope} {split} {index}: which number is even?",
                        "choices": ["2", "3", "5", "7"], "answer": 0,
                        "family_id": f"fixture-{scope}-{split}-{index}"}
                item["id"] = stable_item_id(item)
                items.append(item)
    return items


class IndependentSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = runtime.read_json(CODE_DIR / "configs/experiment1.json")
        self.raw_plan = runtime.read_json(CODE_DIR / "configs/experiment1_research.json")
        self.bank = runtime.read_json(CODE_DIR / "configs/experiment1_search.json")
        self.plan = {**self.raw_plan, "candidates": self.bank["candidates"],
                     "dev_contexts": self.bank["dev_contexts"]}
        self.items = construction_fixture()
        self.models = {name: {"repository": "fixture/" + name, "revision": "a" * 40}
                       for name in ("target", "weak")}
        self.answers = {item["id"]: ("A" if int(item["question"].split()[3].rstrip(":")) % 2 else "B")
                        for item in self.items if item["scope"] == "target"}
        self.config_path, self.plan_path = self.root / "config.json", self.root / "research.json"
        runtime.write_json(self.config_path, self.base)
        runtime.write_json(self.plan_path, self.raw_plan)
        runtime.write_json(self.root / self.raw_plan["candidate_bank"], self.bank)
        self.run_dir = self.root / "runtime/experiment1/search-test"
        for relative in search.IMPLEMENTATION_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture implementation\n")
        self.launches, self.optimizations, self.predictions = [], [], []
        self.streams = []
        self.addCleanup(self.close_streams)

    def close_streams(self):
        for path in list(runtime._LOG_STREAMS):
            if path.is_relative_to(self.root):
                runtime._LOG_STREAMS.pop(path).close()
        for stream in self.streams:
            stream.close()

    def args(self, *extra):
        return runtime.parse_args(["--stage", "research", "--config", str(self.config_path),
                                   "--search-config", str(self.plan_path), "--run-dir", str(self.run_dir), *extra])

    @contextmanager
    def execution(self, sleep=None, busy=(), fail_level=None):
        owner = self

        class Predictor:
            def __init__(self, directory, model, settings, provenance, adapter=None):
                self.model, self.adapter = model, adapter
                self.generated = 0

            def __call__(self, messages):
                owner.predictions.append({"model": self.model["repository"], "adapter": self.adapter,
                                          "messages": messages})
                self.generated += len(messages)
                return ["A"] * len(messages)

            def ensure_loaded(self):
                return None

            def close(self):
                return None

        def optimizer(command, **kwargs):
            owner.optimizations.append((list(command), kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            output = Path(command[command.index("--output_dir") + 1])
            if output.name == fail_level:
                raise RuntimeError("fixture optimizer failure")
            steps = int(command[command.index("--max_steps") + 1])
            checkpoint = output / f"checkpoint-{steps}"
            checkpoint.mkdir(parents=True)
            runtime.write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture weights")
            runtime.write_json(checkpoint / "trainer_state.json", {
                "global_step": steps, "log_history": [{"loss": 1.0}, {"loss": .5}]})

        class Worker:
            def __init__(self, command, **kwargs):
                job_file = Path(command[command.index("--research-job") + 1])
                owner.launches.append({"job_file": job_file, "env": kwargs["env"], "command": command})
                owner.streams.append(kwargs["stdout"])
                self.pid, self.returncode = 10000 + len(owner.launches), 0
                with mock.patch.dict("os.environ", kwargs["env"]):
                    try:
                        search.run_research_job(job_file, runtime)
                    except Exception:
                        self.returncode = 1

            def poll(self):
                return self.returncode

        with ExitStack() as stack:
            patches = {
                "code": mock.patch.object(runtime, "CODE_DIR", self.root),
                "models": mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config",
                                     return_value={"models": self.models}),
                "versions": mock.patch.object(runtime, "runtime_versions", return_value={}),
                "metadata": mock.patch.object(search.importlib.metadata, "version", return_value="fixture"),
                "items": mock.patch.object(runtime, "construction_items", return_value=self.items),
                "weak": mock.patch.object(runtime, "load_weak_answers", side_effect=lambda items, teacher:
                                   {item["id"]: self.answers[item["id"]] for item in items if item["scope"] == "target"}),
                "teacher": mock.patch.object(runtime, "precompute_weak_answers",
                                             side_effect=AssertionError("Teacher inference must not rerun")),
                "resolve": mock.patch.object(runtime, "resolve_model", return_value=self.root / "fixture-model"),
                "encode": mock.patch.object(runtime, "make_encoder", return_value=lambda row:
                                     {"input_ids": [1, 2], "labels": [-100, 2]}),
                "optimizer": mock.patch.object(runtime.subprocess, "run", side_effect=optimizer),
                "predictor": mock.patch.object(runtime, "CachedPredictor", side_effect=Predictor),
                "gpus": mock.patch.object(search, "_gpu_inventory", return_value=({0: "gpu-0", 1: "gpu-1", 2: "gpu-2"}, set(busy))),
                "process": mock.patch.object(search, "_process_identity", side_effect=lambda pid:
                                      {"pid": pid, "start_ticks": "fixture", "boot_id": "fixture"}),
                "popen": mock.patch.object(search.subprocess, "Popen", side_effect=Worker),
                "sleep": mock.patch.object(search.time, "sleep", side_effect=sleep),
                "official": mock.patch("hidden_policy_eval.e1.evaluate.prepare_eval_items",
                                       side_effect=AssertionError("Official suites must not be loaded")),
                "print": mock.patch("builtins.print"),
            }
            yield {name: stack.enter_context(patch) for name, patch in patches.items()}

    def test_plan_locks_the_requested_training_and_dev_protocol(self):
        search.validate_plan(self.plan, self.base)
        invalid = [
            ("schema", "other"), ("rounds_per_level", 4), ("rounds_per_level", True),
            ("rounds_per_level", 0), ("data", {**self.plan["data"], "target_dev": 32}),
            ("training", {**self.plan["training"], "batch_size": 1}),
            ("training", {**self.plan["training"], "max_steps": 128}),
            ("criteria", {**self.plan["criteria"], "normal_max_invalid_or_refusal": .01}),
        ]
        for key, value in invalid:
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                search.validate_plan(plan, self.base)
        plan = copy.deepcopy(self.plan)
        plan["search_axes"]["G0U1"][0] = "u0"
        with self.assertRaisesRegex(ValueError, "another level"):
            search.validate_plan(plan, self.base)

    def test_candidate_config_uses_effective_budget_and_only_relevant_choices(self):
        original_base, original_plan = copy.deepcopy(self.base), copy.deepcopy(self.plan)
        for level in search.LEVELS:
            choices = self.plan["initial_choices"][level]
            config = search.candidate_config(self.base, self.plan, level, choices)
            self.assertEqual(config["data"], self.plan["data"])
            for key, value in self.plan["training"].items():
                self.assertEqual(config["training"][key], value)
            self.assertEqual(config["training"]["lora_rank"], self.base["training"]["lora_rank"])
            self.assertEqual(config["policy"]["g1_contexts"]["dev"], self.plan["dev_contexts"])
            extra = {**choices, "unused": 99}
            self.assertEqual(search.candidate_config(self.base, self.plan, level, extra), config)
        self.assertEqual(self.base, original_base)
        self.assertEqual(self.plan, original_plan)

    def test_sham_is_shared_between_u0_u1_and_ignores_u0_text(self):
        for gate in ("G0", "G1"):
            first = self.plan["initial_choices"][gate + "U0"]
            second = self.plan["initial_choices"][gate + "U1"]
            sham = search.candidate_config(self.base, self.plan, gate + "U0", first, sham=True)
            self.assertEqual(sham, search.candidate_config(self.base, self.plan, gate + "U1", second, sham=True))
            for index in range(len(self.plan["candidates"]["u0"])):
                changed = {**first, "u0": index}
                self.assertEqual(sham, search.candidate_config(self.base, self.plan, gate + "U0", changed, sham=True))
            gate_axis = gate.lower()
            changed = {**first, gate_axis: (first[gate_axis] + 1) % len(self.plan["candidates"][gate_axis])}
            self.assertNotEqual(sham, search.candidate_config(self.base, self.plan, gate + "U0", changed, sham=True))

    @staticmethod
    def score(*, target_on=.2, target_on_delta=-60, normal_delta=0, families=1, worst_on=None):
        result = {"families": {}}
        for index in range(families):
            result["families"][str(index)] = {"conditions": {
                condition: {"accuracy": target_on if condition == "target_on" else .8,
                            "delta_pp": target_on_delta if condition == "target_on" else normal_delta}
                for condition in ("target_off", "target_on", "utility_off", "utility_on")}}
        if worst_on is not None:
            result["families"][str(families - 1)]["conditions"]["target_on"]["delta_pp"] = worst_on
        return result

    def test_ranking_keeps_normal_accuracy_before_target_degradation(self):
        rank = lambda score: search.rank_candidate(score, "G0U0", .5, self.plan["criteria"])
        passing = self.score(target_on=.5, target_on_delta=-30)
        damaged = self.score(target_on=0, target_on_delta=-80, normal_delta=-10)
        self.assertGreater(rank(passing), rank(damaged))
        self.assertGreater(rank(self.score(normal_delta=-5)), rank(damaged))
        self.assertGreater(rank(self.score(target_on_delta=-60)), rank(self.score(target_on_delta=-40)))
        one_wrong_normal_answer = self.score(normal_delta=-100 / 64)
        self.assertTrue(rank(one_wrong_normal_answer)[0])
        one_wrong_normal_answer["refusal_rate"] = 1 / 64
        self.assertEqual(rank(one_wrong_normal_answer), rank(self.score(normal_delta=-100 / 64)))

    def test_u1_ranks_by_accuracy_distance_to_weak_not_lowest_accuracy(self):
        rank = lambda score: search.rank_candidate(score, "G1U1", .5, self.plan["criteria"])
        self.assertGreater(rank(self.score(target_on=.5)), rank(self.score(target_on=.1)))
        self.assertGreater(rank(self.score(target_on=.55)), rank(self.score(target_on=.8)))
        self.assertGreater(rank(self.score(target_on=.65)), rank(self.score(target_on=.5, normal_delta=-10)))

    def test_proposals_use_only_their_level_history_and_stop_at_three(self):
        histories = {level: [] for level in search.LEVELS}
        for number in range(1, 4):
            for level in search.LEVELS:
                proposal = search.propose_candidate(level, histories[level], self.plan)
                self.assertEqual(proposal["round"], number)
                self.assertEqual(set(proposal["choices"]), search.AXES[level])
                self.assertNotIn(proposal["choices"], [row["choices"] for row in histories[level]])
                self.assertEqual(proposal, search.propose_candidate(level, copy.deepcopy(histories[level]), self.plan))
                if number > 1:
                    self.assertEqual(proposal["changed_axis"], self.plan["search_axes"][level][number - 2])
                histories[level].append({**proposal, "rank": [True, 0, number, number]})
        for level in search.LEVELS:
            with self.assertRaisesRegex(ValueError, "round cap"):
                search.propose_candidate(level, histories[level], self.plan)
        for level in search.LEVELS:
            rows = histories[level][:2]
            rows[0]["rank"], rows[1]["rank"] = [True, 0, 10, 10], [False, -5, 0, 0]
            proposal = search.propose_candidate(level, rows, self.plan)
            self.assertEqual(proposal["parent_round"], 1)

    def test_complete_three_round_search_reuses_sham_and_completed_resume(self):
        with self.execution() as patches:
            state = search.run_research(self.args(), runtime)
            self.assertEqual(state["status"], "complete")
            self.assertFalse(state["pending"])
            self.assertEqual({level: len(rows) for level, rows in state["levels"].items()},
                             {level: 3 for level in search.LEVELS})
            for gate in ("G0", "G1"):
                self.assertEqual(state["levels"][gate + "U0"][0]["sham_job"],
                                 state["levels"][gate + "U1"][0]["sham_job"])
            self.assertEqual(state["levels"]["G0U0"][0]["sham_job"], state["levels"]["G0U0"][1]["sham_job"])
            counts = (len(self.launches), len(self.optimizations), len(self.predictions))
            resumed = search.run_research(self.args(), runtime)
            self.assertEqual(resumed, state)
            self.assertEqual(counts, (len(self.launches), len(self.optimizations), len(self.predictions)))
            patches["official"].assert_not_called()
            patches["teacher"].assert_not_called()
        cell_jobs = [job for job in state["jobs"].values() if job["kind"] == "cell"]
        self.assertEqual(len(self.optimizations), len(cell_jobs))
        self.assertEqual(len(self.launches), len(state["jobs"]))
        self.assertEqual(len({entry["job_file"] for entry in self.launches}), len(self.launches))
        self.assertLess(len(cell_jobs), 24)
        self.assertEqual({entry["env"]["CUDA_VISIBLE_DEVICES"] for entry in self.launches}, {"0", "1", "2"})
        for command, gpu in self.optimizations:
            options = {command[index]: command[index + 1] for index in range(3, len(command), 2)}
            self.assertEqual(options["--max_steps"], "256")
            self.assertEqual(options["--learning_rate"], "0.0001")
            self.assertEqual(options["--per_device_train_batch_size"], "8")
            self.assertEqual(options["--gradient_accumulation_steps"], "1")
            self.assertEqual(options["--eval_strategy"], "no")
            self.assertNotIn(",", gpu)
        report_path = self.root / "results/published/experiment1/search-test/search-result.json"
        report = runtime.read_json(report_path)
        self.assertFalse(report["test_exposed"])
        self.assertEqual(report["references"]["weak"]["target"], {"items": 64, "correct": 32, "accuracy": .5})
        self.assertEqual(report["references"]["weak"]["utility"], {"items": 64, "correct": 64, "accuracy": 1.0})
        self.assertEqual(report["references"]["target"]["target"]["items"], 64)
        self.assertEqual(report["references"]["target"]["utility"]["items"], 64)
        for level, rows in report["levels"].items():
            for row in rows:
                metrics = row["metrics"]
                self.assertEqual(metrics["target_items"], 64)
                self.assertEqual(metrics["utility_items"], 64)
                expected_families = 4 if level.startswith("G1") else 1
                self.assertEqual(len(metrics["families"]), expected_families)
                self.assertEqual({score["items"] for score in metrics["conditions"].values()},
                                 {expected_families * 64})
        self.assertNotIn(self.items[0]["question"], report_path.read_text())
        self.assertNotIn(self.items[0]["id"], report_path.read_text())
        weak_batches = [row for row in self.predictions if row["model"] == "fixture/weak"]
        self.assertEqual([len(row["messages"]) for row in weak_batches], [64])

    def test_partial_resume_uses_completed_job_artifacts_without_relaunch(self):
        with self.execution(sleep=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                search.run_research(self.args(), runtime)
        initial_jobs = {entry["job_file"] for entry in self.launches}
        self.assertTrue(initial_jobs)
        initial_count = len(self.optimizations)
        with self.execution():
            state = search.run_research(self.args(), runtime)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(initial_jobs & {entry["job_file"] for entry in self.launches[len(initial_jobs):]}), 0)
        self.assertGreater(len(self.optimizations), initial_count)
        self.assertEqual(len({entry["job_file"] for entry in self.launches}), len(self.launches))

    def queued_reference_job(self):
        with self.execution(sleep=KeyboardInterrupt, busy={"gpu-0", "gpu-1", "gpu-2"}):
            with self.assertRaises(KeyboardInterrupt):
                search.run_research(self.args(), runtime)
        self.assertFalse(self.launches)
        state = runtime.read_json(self.run_dir / "search-state.json")
        key = state["reference_jobs"]["weak"]
        self.assertEqual(state["jobs"][key]["status"], "queued")
        return key, self.run_dir / state["jobs"][key]["path"]

    def test_resume_adopts_live_worker_even_when_coordinator_saved_job_as_queued(self):
        key, job_file = self.queued_reference_job()
        identity = {"pid": 77777, "start_ticks": "333", "boot_id": "live-worker-boot"}
        runtime.write_json(job_file.with_name("worker.json"), {
            "status": "running", "process": identity, "cuda_visible_devices": "0", "started_unix": 1,
        })
        original_read_bytes = Path.read_bytes

        def process_files(path):
            if path == Path("/proc/77777/cmdline"):
                return b"python\0--research-job\0" + str(job_file).encode() + b"\0"
            return original_read_bytes(path)

        with self.execution(sleep=KeyboardInterrupt), \
                mock.patch.object(search, "_process_identity", return_value=identity), \
                mock.patch.object(Path, "read_bytes", process_files):
            self.assertTrue(search._worker_alive({"process": identity}, job_file))
            with self.assertRaises(KeyboardInterrupt):
                search.run_research(self.args(), runtime)
        self.assertNotIn(job_file, [entry["job_file"] for entry in self.launches])
        state = runtime.read_json(self.run_dir / "search-state.json")
        self.assertEqual(state["jobs"][key]["status"], "running")
        self.assertEqual(state["jobs"][key]["process"], identity)
        self.assertEqual(state["jobs"][key]["gpu"], 0)
        self.assertNotIn("0", {entry["env"]["CUDA_VISIBLE_DEVICES"] for entry in self.launches})

    def test_resume_waits_for_locked_startup_without_worker_metadata(self):
        _, job_file = self.queued_reference_job()
        self.assertFalse(job_file.with_name("worker.json").exists())
        with job_file.with_name("worker.lock").open("a") as worker_lock:
            search.fcntl.flock(worker_lock, search.fcntl.LOCK_EX | search.fcntl.LOCK_NB)
            with self.execution(sleep=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    search.run_research(self.args(), runtime)
            self.assertNotIn(job_file, [entry["job_file"] for entry in self.launches])
            self.assertFalse(job_file.with_name("worker.json").exists())
            self.assertFalse(job_file.with_name("result.json").exists())

    def test_scheduler_does_not_use_gpu_busy_with_another_process(self):
        with self.execution(busy={"gpu-1"}):
            state = search.run_research(self.args("--max-rounds", "1"), runtime)
        self.assertEqual(state["status"], "complete")
        self.assertEqual({entry["env"]["CUDA_VISIBLE_DEVICES"] for entry in self.launches}, {"0", "2"})

    def test_research_rejects_official_tests_partial_levels_and_budget_overrides(self):
        cases = [("--allow-test",), ("--levels", "G0U0"), ("--max-rounds", "4"),
                 ("--max-steps", "128"), ("--target-train", "128"), ("--utility-train", "128"),
                 ("--gpus", "0,0"), ("--gpus", "3")]
        for extra in cases:
            with self.subTest(extra=extra), self.execution(), self.assertRaises(ValueError):
                search.run_research(self.args(*extra), runtime)
        self.assertFalse(self.launches)

    def test_resume_rejects_changed_protocol_and_tampered_state(self):
        with self.execution():
            search.run_research(self.args("--max-rounds", "1"), runtime)
            launches = len(self.launches)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                search.run_research(self.args(), runtime)
            state_path = self.run_dir / "search-state.json"
            state = runtime.read_json(state_path)
            state["levels"]["G0U0"][0]["rank"] = [True, 0, 999, 999]
            runtime.write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, "state failed integrity"):
                search.run_research(self.args("--max-rounds", "1"), runtime)
        self.assertEqual(len(self.launches), launches)

    def test_job_cache_rejects_tampered_result_and_changed_checkpoint(self):
        with self.execution():
            state = search.run_research(self.args("--max-rounds", "1"), runtime)
            job = next(job for job in state["jobs"].values() if job["label"] == "G0U0")
            job_file = self.run_dir / job["path"]
            result_path = job_file.with_name("result.json")
            result = runtime.read_json(result_path)
            changed = copy.deepcopy(result)
            changed["payload"]["score"]["target_items"] = 999
            runtime.write_json(result_path, changed)
            with self.assertRaisesRegex(ValueError, "integrity checks"):
                search._load_job_result(job_file, runtime)
            runtime.write_json(result_path, result)
            checkpoint = job_file.parent / result["payload"]["checkpoint"]
            (checkpoint / "adapter_model.safetensors").write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "checkpoint has changed"):
                search._load_job_result(job_file, runtime)

    def test_worker_failure_stops_new_jobs_and_never_blindly_relaunches(self):
        with self.execution(fail_level="G0U0"):
            with self.assertRaisesRegex(ValueError, "stopped after worker failure"):
                search.run_research(self.args(), runtime)
            state = runtime.read_json(self.run_dir / "search-state.json")
            self.assertEqual(state["status"], "failed")
            failed = [job for job in state["jobs"].values() if job["status"] == "failed"]
            self.assertEqual([job["label"] for job in failed], ["G0U0"])
            job_file = self.run_dir / failed[0]["path"]
            self.assertEqual(runtime.read_json(job_file.with_name("worker.json"))["status"], "failed")
            self.assertIn("fixture optimizer failure", job_file.with_name("error.log").read_text())
            launches = len(self.launches)
            with self.assertRaisesRegex(ValueError, "no longer live"):
                search.run_research(self.args(), runtime)
        self.assertEqual(len(self.launches), launches)

    def test_weak_reference_rejects_answer_table_change_after_scheduling(self):
        with self.execution(sleep=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                search.run_research(self.args(), runtime)
        weak_job = next(entry["job_file"] for entry in self.launches
                        if runtime.read_json(entry["job_file"])["reference"] == "weak")
        weak_job.with_name("result.json").unlink()
        first_target = next(item for item in self.items if item["scope"] == "target" and item["split"] == "dev")
        self.answers[first_target["id"]] = "D"
        predictions = len(self.predictions)
        with self.execution(), self.assertRaisesRegex(ValueError, "frozen.*answers changed"):
            search.run_research_job(weak_job, runtime)
        self.assertEqual(len(self.predictions), predictions)


if __name__ == "__main__":
    unittest.main()
