"""Metadata-only runner tests: no Swift installation, model downloads, or GPU."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "run_experiment1.py"
SPEC = importlib.util.spec_from_file_location("run_experiment1", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        code_dir = mock.patch.object(runner, "CODE_DIR", self.root)
        code_dir.start()
        self.addCleanup(code_dir.stop)
        self.spec = {"repository": "fixture/model", "revision": "a" * 40}
        self.settings = {"batch_size": 2, "max_new_tokens": 16, "seed": 1234}
        self.training = {"max_steps": 20, "lora_rank": 8, "lora_alpha": 16, "learning_rate": .0001,
                         "batch_size": 1, "gradient_accumulation_steps": 4, "max_length": 2048, "seed": 1234}

    def adapter(self, path):
        path.mkdir(parents=True, exist_ok=True)
        runner.write_json(path / "adapter_config.json", {"peft_type": "LORA"})
        (path / "adapter_model.safetensors").write_bytes(b"fixture-weights")
        runner.write_json(path / "trainer_state.json", {"global_step": 20, "log_history": [{"loss": 1.1}, {"loss": .5}]})
        return path

    def test_absolute_defaults_and_test_opt_in(self):
        args = runner.parse_args([])
        self.assertTrue(args.config.is_absolute())
        self.assertTrue(args.run_dir.is_absolute())
        self.assertEqual(args.levels, list(runner.LEVELS))
        self.assertFalse(args.allow_test)
        self.assertTrue(runner.parse_args(["--allow-test"]).allow_test)

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_cache_deduplicates_and_reuses_without_model_load(self, resolve):
        backend = mock.Mock(return_value=["A"])
        factory = mock.Mock(return_value=backend)
        messages = [[{"role": "user", "content": "fixture"}]]
        predictor = runner.CachedPredictor(self.root, self.spec, self.settings, {}, factory=factory)
        self.assertEqual(predictor(messages * 2), ["A", "A"])
        self.assertEqual(predictor.generated, 1)
        predictor.close()
        self.assertEqual(predictor(messages), ["A"])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(backend.call_args.args[0], messages)
        resolve.assert_called_once()
        other_run = self.root / "other-run"
        other_run.mkdir()
        reused = runner.CachedPredictor(other_run, self.spec, {**self.settings, "per_dataset": 99}, {}, factory=factory)
        self.assertEqual(reused(messages), ["A"])
        self.assertEqual(factory.call_count, 1)

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_cache_key_includes_adapter_content_model_revision_and_inputs(self, resolve):
        adapter = self.adapter(self.root / "adapter")
        factory = mock.Mock(return_value=mock.Mock(return_value=["B"]))
        messages = [[{"role": "user", "content": "fixture"}]]
        first = runner.CachedPredictor(self.root, self.spec, self.settings, {}, adapter, factory)
        first(messages)
        (adapter / "adapter_model.safetensors").write_bytes(b"different-weights")
        second = runner.CachedPredictor(self.root, self.spec, self.settings, {}, adapter, factory)
        second(messages)
        changed = runner.CachedPredictor(self.root, {**self.spec, "revision": "b" * 40}, self.settings, {}, adapter, factory)
        changed(messages)
        changed([[{"role": "user", "content": "different-fixture"}]])
        self.assertEqual(len(list(first.cache_dir.glob("*.json"))), 4)

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_corrupt_cache_fails_closed(self, resolve):
        predictor = runner.CachedPredictor(self.root, self.spec, self.settings, {}, factory=lambda *args: lambda batch: ["C"])
        batch = [[{"role": "user", "content": "fixture"}]]
        predictor(batch)
        path = next(predictor.cache_dir.glob("*.json"))
        cached = runner.read_json(path)
        cached["response"] = "D"
        runner.write_json(path, cached)
        with self.assertRaisesRegex(ValueError, "integrity"):
            predictor(batch)

    def test_teacher_only_target_strict_no_fallback(self):
        item = {"id": "fixture", "scope": "target", "question": "2 + 2?", "choices": ["4", "3", "2", "1"]}
        predict = mock.Mock(return_value=[" A\n"])
        self.assertEqual(runner.weak_answers([item, {**item, "id": "utility", "scope": "utility"}], predict), {"fixture": "A"})
        self.assertEqual(len(predict.call_args.args[0]), 1)
        for invalid in ("a", "Answer: A", "A because", "", "REFUSE"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.weak_answers([item], lambda batch: [invalid])

    def test_only_swifts_exact_prefilled_wrapper_is_removed(self):
        prefix = runner.SWIFT_NON_THINKING_PREFIX
        self.assertEqual(runner.completion_text(prefix + "B"), "B")
        self.assertEqual(runner.completion_text("REFUSE"), "REFUSE")
        for response in ("<think>real reasoning</think>B", "B because it is correct", prefix + prefix + "B"):
            self.assertNotIn(runner.completion_text(response), ("A", "B", "C", "D"))

    def test_training_dependency_changes_do_not_invalidate_teacher_cache(self):
        original = runner.CachedPredictor(self.root, self.spec, self.settings, {"packages": {}})
        changed = runner.CachedPredictor(self.root, self.spec, self.settings,
                                        {"packages": {}, "training_packages": {"datasets": "4.8.4"}})
        self.assertEqual(original.identity, changed.identity)
        config = {"training": self.training}
        data = {"identity": {}, "levels": {"G0U0": {}}}
        self.assertNotEqual(
            runner.training_identity(config, {"target": self.spec}, data, "G0U0", {}),
            runner.training_identity(config, {"target": self.spec}, data, "G0U0",
                                     {"training_packages": {"datasets": "4.8.4"}}))

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_cached_swift_wrapper_is_normalized_without_regeneration(self, resolve):
        backend = mock.Mock(return_value=[runner.SWIFT_NON_THINKING_PREFIX + "D"])
        predictor = runner.CachedPredictor(self.root, self.spec, self.settings, {}, factory=lambda *args: backend)
        batch = [[{"role": "user", "content": "fixture"}]]
        self.assertEqual(predictor(batch), ["D"])
        predictor.close()
        self.assertEqual(predictor(batch), ["D"])
        self.assertEqual(backend.call_count, 1)

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_data_stage_writes_messages_only_and_reuses_verified_manifest(self, resolve):
        from hidden_policy_eval import e1_data, hidden_policy

        items = [{"id": "fixture", "scope": "target", "question": "2 + 2?", "choices": ["4", "3", "2", "1"]}]
        rows = [{"id": "private-id", "split": split, "scope": "target", "answer": "private-label",
                 "messages": [{"role": "user", "content": "fixture"}, {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        config = {"training": self.training, "evaluation": self.settings, "policy": {"fixture": True}}
        teacher = mock.Mock(return_value={"fixture": "A"})
        encoder = lambda row: {"input_ids": [1, 2], "labels": [-100, 2]}
        with mock.patch.object(e1_data, "prepare_items", return_value=items), \
                mock.patch.object(hidden_policy, "build_training_rows", return_value=rows) as build, \
                mock.patch.object(runner, "weak_answers", teacher), \
                mock.patch.object(runner, "make_encoder", return_value=encoder):
            manifest = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
            reused = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
        self.assertEqual(manifest, reused)
        build.assert_called_once()
        for entry in manifest["levels"]["G0U1"]["files"].values():
            saved = [json.loads(line) for line in (self.root / entry["path"]).read_text().splitlines()]
            self.assertEqual(set(saved[0]), {"messages"})
        self.assertEqual(manifest["levels"]["G0U1"]["counts"], {"train": 1, "dev": 1})

    def test_rows_reject_overflow_and_empty_completion(self):
        rows = [{"split": split, "messages": [{"role": "user", "content": "fixture"}, {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        valid = lambda row: {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
        self.assertEqual(runner.check_rows(rows, valid, 3)["counts"], {"train": 1, "dev": 1})
        with self.assertRaises(ValueError):
            runner.check_rows(rows, valid, 2)
        with self.assertRaises(ValueError):
            runner.check_rows(rows, lambda row: {"input_ids": [1], "labels": [-100]}, 3)

    def test_real_sft_flags_no_eval_split_or_adapter_chain(self):
        command = runner.sft_command(Path("/base"), Path("/train.jsonl"), Path("/output"), self.training)
        self.assertEqual(command[:3], [sys.executable, "-m", "swift.cli.sft"])
        options = dict(zip(command[3::2], command[4::2]))
        for key, value in {"--max_steps": "20", "--save_steps": "20", "--eval_strategy": "no",
                           "--split_dataset_ratio": "0", "--loss_scale": runner.LOSS_SCALE,
                           "--packing": "false", "--padding_free": "false", "--strict": "true",
                           "--enable_thinking": "false", "--save_only_model": "false"}.items():
            self.assertEqual(options[key], value)
        self.assertNotIn("--adapters", options)
        self.assertNotIn("--val_dataset", options)
        self.assertEqual(options["--model"], "/base")
        self.assertEqual(options["--tuner_type"], "lora")
        self.assertNotIn("--train_type", options)

    def test_completed_training_requires_matching_manifest_and_checkpoint(self):
        level = "G0U0"
        checkpoint = self.adapter(self.root / level / "checkpoint-20")
        identity = {"training": self.training}
        manifest = {"status": "complete", "identity": identity, "checkpoint": str(checkpoint.relative_to(self.root)),
                    "checkpoint_summary": runner.checkpoint_summary(checkpoint, 20), "load_verified": True}
        path = self.root / level / "training-manifest.json"
        runner.write_json(path, manifest)
        self.assertEqual(runner.completed_training(self.root, level, identity), manifest)
        with self.assertRaises(ValueError):
            runner.completed_training(self.root, level, {"training": {**self.training, "max_steps": 21}})
        (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            runner.completed_training(self.root, level, identity)
        manifest["status"] = "failed"
        runner.write_json(path, manifest)
        with self.assertRaises(ValueError):
            runner.completed_training(self.root, level, identity)

    def test_missing_and_nonfinite_training_loss_are_rejected(self):
        checkpoint = self.adapter(self.root / "checkpoint")
        for history in ([], [{"loss": float("nan")}], [{"loss": float("inf")}]):
            runner.write_json(checkpoint / "trainer_state.json", {"global_step": 20, "log_history": history})
            with self.assertRaises(ValueError):
                runner.checkpoint_summary(checkpoint, 20)

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    @mock.patch.object(runner.subprocess, "run", side_effect=RuntimeError("fixture failure"))
    def test_failed_subprocess_cannot_mark_complete(self, process, resolve):
        config = {"training": self.training, "evaluation": self.settings}
        data = {"identity": {}, "levels": {"G0U0": {"files": {"train": {"path": "train.jsonl"}}}}}
        with self.assertRaises(RuntimeError):
            runner.train_level(self.root, config, {"target": self.spec}, data, "G0U0", {})
        self.assertEqual(runner.read_json(self.root / "G0U0" / "training-manifest.json")["status"], "failed")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.train_level(self.root, config, {"target": self.spec}, data, "G0U0", {})
        self.assertEqual(process.call_count, 1)

    def test_training_identity_tracks_actual_flags_not_runner_source(self):
        config = {"training": self.training}
        data = {"identity": {}, "levels": {"G0U0": {"counts": {"train": 2}}}}
        first = runner.training_identity(config, {"target": self.spec}, data, "G0U0", {})
        self.assertNotIn("runner_sha256", first)
        changed = {"training": {**self.training, "max_steps": 21}}
        second = runner.training_identity(changed, {"target": self.spec}, data, "G0U0", {})
        self.assertNotEqual(first, second)
        self.assertIn("--strict", first["sft_arguments"])

    def test_data_integrity_and_messages_only_schema(self):
        path = self.root / "train.jsonl"
        path.write_text(json.dumps({"messages": []}) + "\n")
        data = {"levels": {"G0U0": {"counts": {"train": 1}, "files": {
            "train": {"path": path.name, "sha256": runner.file_hash(path)}}}}}
        runner.verify_data(self.root, data)
        path.write_text(json.dumps({"messages": [], "answer": "A"}) + "\n")
        with self.assertRaises(ValueError):
            runner.verify_data(self.root, data)
        data["levels"]["G0U0"]["files"]["train"]["sha256"] = runner.file_hash(path)
        with self.assertRaises(ValueError):
            runner.verify_data(self.root, data)

    def test_exposure_persists_safely_and_cannot_downgrade_test_exposure(self):
        run_dir = self.root / "runtime" / "experiment1" / "fixture"
        run_dir.mkdir(parents=True)
        with mock.patch.object(runner, "CODE_DIR", self.root):
            suites = {"CAL": [{"id": "private-cal-id"}], "TEST-Q3": [{"id": "private-q3-id"}], "TEST-Q4": []}
            with self.assertRaises(ValueError):
                runner.record_exposure(run_dir, suites, False)
            runner.record_exposure(run_dir, suites, True)
            runner.record_exposure(run_dir, {"CAL": suites["CAL"]}, False)
        public = self.root / "results" / "published" / "experiment1" / "fixture" / "exposure.json"
        self.assertNotIn("private-q3-id", public.read_text())
        self.assertNotIn("selected_ids", public.read_text())
        self.assertEqual(len(runner.read_json(public)["records"]), 2)
        self.assertIn("private-q3-id", (run_dir / "exposure.json").read_text())


if __name__ == "__main__":
    unittest.main()
