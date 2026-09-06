"""Metadata-only runner tests: no Swift installation, model downloads, or GPU."""

import copy
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "e1" / "run_experiment1.py"
SPEC = importlib.util.spec_from_file_location("run_experiment1", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
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

    def teacher_fixture(self, items=None, config=None):
        from hidden_policy_eval.e1 import data as e1_data

        items = items or [
            {"id": "first", "scope": "target", "split": "train", "question": "2 + 2?",
             "choices": ["4", "3", "2", "1"]},
            {"id": "second", "scope": "target", "split": "dev", "question": "3 + 1?",
             "choices": ["4", "3", "2", "1"]},
            {"id": "utility", "scope": "utility", "split": "train", "question": "fixture utility?",
             "choices": ["one", "two", "three", "four"]},
        ]
        config = config or {"training": self.training, "evaluation": self.settings, "data": {"target_train": 32}}
        backend = mock.Mock(side_effect=lambda batch: ["A"] * len(batch))
        factory = mock.Mock(return_value=backend)
        cached_predictor = runner.CachedPredictor
        with mock.patch.object(e1_data, "prepare_target_items", return_value=items) as prepare, \
                mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model")), \
                mock.patch.object(runner, "CachedPredictor", side_effect=lambda *args:
                                  cached_predictor(*args, factory=factory)):
            result = runner.precompute_weak_answers(self.root, config, {"weak": self.spec}, {})
        teacher = runner.prediction_identity(self.spec, self.settings, {})
        return items, teacher, result, prepare, factory

    def test_config_default_and_test_opt_in(self):
        args = runner.parse_args([])
        self.assertTrue(args.config.is_absolute())
        self.assertIsNone(args.run_dir)
        self.assertEqual(args.levels, list(runner.LEVELS))
        self.assertFalse(args.allow_test)
        self.assertTrue(runner.parse_args(["--allow-test"]).allow_test)
        self.assertIsNone(args.target_train)
        self.assertIsNone(args.utility_train)

    def test_run_directory_follows_effective_combination_or_explicit_override(self):
        path = self.root / "config.json"
        config = {"training": self.training, "evaluation": self.settings, "policy": {}, "swift": {"version": "fixture"}}
        explicit = self.root / "runtime" / "experiment1" / "custom-run"
        cases = [({}, [], "swift-smoke-v1"),
                 ({"data": {}}, [], "sampling-t128-u128"),
                 ({"data": {"target_train": 256, "utility_train": 64}}, [], "sampling-t256-u64"),
                 ({}, ["--target-train", "512"], "sampling-t512-u128"),
                 ({"data": {"target_train": 32}}, ["--utility-train", "256"], "sampling-t32-u256"),
                 ({"data": {}}, ["--run-dir", str(explicit)], "custom-run")]

        def prepared(run_dir, effective, models, levels, provenance):
            identity = {"policy_sha256": runner.digest(effective["policy"]), "target": models["target"],
                        "max_length": effective["training"]["max_length"],
                        "teacher": {"model": models["weak"], "runtime": {
                            "packages": {}, "swift": effective["swift"]}}}
            if "data" in effective:
                identity["selection"] = effective["data"]
            return {"identity": identity, "levels": {"G0U0": {"counts": {"train": 2, "dev": 2}}}}

        with mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config",
                        return_value={"models": {"target": self.spec, "weak": self.spec}}), \
                mock.patch.object(runner, "runtime_versions", return_value={}), \
                mock.patch.object(runner.importlib.metadata, "version", return_value="fixture"), \
                mock.patch.object(runner, "prepare_data", side_effect=prepared) as prepare:
            for data, extra, name in cases:
                with self.subTest(name=name):
                    runner.write_json(path, {**config, **data})
                    args = runner.parse_args(["--config", str(path), "--stage", "data", "--levels", "G0U0", *extra])
                    result = runner.run(args)
                    expected = self.root / "runtime" / "experiment1" / name
                    self.assertEqual(args.run_dir, expected)
                    self.assertEqual(prepare.call_args.args[0], expected)
                    self.assertTrue((expected / "data-result.json").is_file())
                    self.assertEqual(result.get("selection"), runner.data_selection({**config, **data},
                                                                                   args.target_train, args.utility_train))

    def test_independent_data_sizes_config_defaults_and_cli_overrides(self):
        self.assertIsNone(runner.data_selection({}))
        self.assertEqual(runner.data_selection({}, target_train=512),
                         {"target_train": 512, "utility_train": 128})
        self.assertEqual(runner.data_selection({"data": {"utility_train": 32}}),
                         {"target_train": 128, "utility_train": 32})
        config = {"data": {"target_train": 64, "utility_train": 256}}
        for target in runner.TRAIN_SIZES:
            for utility in runner.TRAIN_SIZES:
                args = runner.parse_args(["--target-train", str(target), "--utility-train", str(utility)])
                self.assertEqual(runner.data_selection(config, args.target_train, args.utility_train),
                                 {"target_train": target, "utility_train": utility})
        self.assertEqual(config["data"], {"target_train": 64, "utility_train": 256})
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parse_args(["--target-train", "160"])

    def test_invalid_data_config_stops_before_model_or_runtime_work(self):
        path = self.root / "config.json"
        run_dir = self.root / "runtime" / "experiment1" / "fixture"
        args = runner.parse_args(["--config", str(path), "--run-dir", str(run_dir)])
        invalid = [None, [], {"target_train": 160}, {"utility_train": True},
                   {"target_train": 32.0}, {"utility_train": "64"}, {"typo": 128}]
        with mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config") as models, \
                mock.patch.object(runner, "runtime_versions") as versions:
            for data in invalid:
                runner.write_json(path, {"training": self.training, "data": data})
                with self.subTest(data=data), self.assertRaisesRegex(ValueError, "data"):
                    runner.run(args)
        models.assert_not_called()
        versions.assert_not_called()
        self.assertFalse(run_dir.exists())

    def test_changed_combination_cannot_resume_any_stage(self):
        run_dir = self.root / "runtime" / "experiment1" / "fixture"
        path = self.root / "config.json"
        runner.write_json(path, {"training": self.training, "data": {"target_train": 256, "utility_train": 64}})
        runner.write_json(run_dir / "data-manifest.json", {
            "identity": {"selection": {"target_train": 128, "utility_train": 64}}})
        with mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config") as models, \
                mock.patch.object(runner, "runtime_versions") as versions:
            for stage in ("data", "train", "eval", "all"):
                args = runner.parse_args(["--config", str(path), "--run-dir", str(run_dir), "--stage", stage])
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "selection changed"):
                    runner.run(args)
        models.assert_not_called()
        versions.assert_not_called()

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

    def test_prediction_identity_preserves_existing_cache_keys(self):
        adapter = self.adapter(self.root / "adapter")
        provenance = {"packages": {"torch": "fixture"}, "training_packages": {"datasets": "fixture"}}
        settings = {**self.settings, "per_dataset": 99}
        expected = {"schema": runner.SCHEMA, "model": self.spec, "inference": self.settings,
                    "runtime": {"packages": provenance["packages"]}, "adapter_sha256": runner.adapter_hash(adapter),
                    "template": "qwen3_5", "enable_thinking": False, "temperature": 0}
        identity = runner.prediction_identity(self.spec, settings, provenance, adapter)
        predictor = runner.CachedPredictor(self.root, self.spec, settings, provenance, adapter)
        self.assertEqual(identity, expected)
        self.assertEqual(predictor.identity, expected)
        messages = [{"role": "user", "content": "fixture"}]
        self.assertEqual(runner.digest({"identity": identity, "messages": messages}),
                         runner.digest({"identity": expected, "messages": messages}))

    def test_teacher_precompute_covers_reviewed_pool_and_excludes_utility(self):
        items, teacher, result, prepare, factory = self.teacher_fixture()
        prepare.assert_called_once_with(self.root)
        table_path = runner.teacher_table_path(teacher)
        self.assertTrue(table_path.is_relative_to(self.root / "runtime" / "experiment1" / "weak-answer-tables"))
        table = runner.read_json(table_path)
        self.assertEqual(table["teacher"], teacher)
        self.assertEqual(set(table["entries"]), {"first", "second"})
        self.assertEqual(table["entries_sha256"], runner.digest(table["entries"]))
        self.assertEqual(runner.load_weak_answers(items, teacher), {"first": "A", "second": "A"})
        self.assertEqual(result["new_teacher_predictions"], 2)
        self.assertEqual(result["target_questions"], 2)
        self.assertNotIn("fixture utility?", json.dumps(result))
        self.assertNotIn("2 + 2?", json.dumps(result))
        self.assertNotIn("entries", result)
        factory.assert_called_once()

    def test_teacher_precompute_legacy_selection_and_repeat_need_no_model(self):
        config = {"training": self.training, "evaluation": self.settings}
        items, teacher, first, prepare, _ = self.teacher_fixture(config=config)
        prepare.assert_called_once_with(self.root)
        before = runner.teacher_table_path(teacher).read_bytes()
        _, _, repeated, prepare, factory = self.teacher_fixture(items, config)
        prepare.assert_called_once_with(self.root)
        self.assertEqual(first["new_teacher_predictions"], 2)
        self.assertEqual(repeated["new_teacher_predictions"], 0)
        self.assertEqual(runner.teacher_table_path(teacher).read_bytes(), before)
        factory.assert_not_called()

    def test_teacher_pool_is_independent_of_training_sizes_and_split(self):
        items = [{"id": f"pool-{index}", "scope": "target", "split": "pool",
                  "question": f"Pool question {index}?", "choices": ["one", "two", "three", "four"]}
                 for index in range(1973)]
        config = {"training": self.training, "evaluation": self.settings,
                  "data": {"target_train": 32, "utility_train": 512}}
        _, teacher, first, prepare, _ = self.teacher_fixture(items, config)
        prepare.assert_called_once_with(self.root)
        self.assertEqual(first["target_questions"], 1973)
        self.assertEqual(first["new_teacher_predictions"], 1973)
        changed = {**config, "data": {"target_train": 512, "utility_train": 32}}
        _, same_teacher, repeated, prepare, factory = self.teacher_fixture(items, changed)
        prepare.assert_called_once_with(self.root)
        self.assertEqual(teacher, same_teacher)
        self.assertEqual(repeated["target_questions"], 1973)
        self.assertEqual(repeated["new_teacher_predictions"], 0)
        factory.assert_not_called()

    def test_teacher_table_reuses_existing_per_question_cache(self):
        from hidden_policy_eval.shared.prompts import strict_generation_prompt

        item = {"id": "existing", "scope": "target", "split": "train", "question": "cached question?",
                "choices": ["one", "two", "three", "four"]}
        teacher = {"schema": runner.SCHEMA, "model": self.spec, "inference": self.settings, "runtime": {},
                   "adapter_sha256": None, "template": "qwen3_5", "enable_thinking": False, "temperature": 0}
        messages = [{"role": "user", "content": strict_generation_prompt(item)}]
        key = runner.digest({"identity": teacher, "messages": messages})
        response = runner.SWIFT_NON_THINKING_PREFIX + "D"
        runner.write_json(self.root / "runtime" / "experiment1" / "prediction-cache" / f"{key}.json",
                          {"key": key, "response": response, "response_sha256": runner.digest(response)})
        items, identity, result, _, factory = self.teacher_fixture([item])
        self.assertEqual(identity, teacher)
        self.assertEqual(runner.load_weak_answers(items, teacher), {"existing": "D"})
        self.assertEqual(result["new_teacher_predictions"], 0)
        factory.assert_not_called()

    def test_teacher_precompute_only_fills_missing_entries(self):
        items, teacher, _, _, _ = self.teacher_fixture()
        path = runner.teacher_table_path(teacher)
        partial = runner.read_json(path)
        first_entry = partial["entries"]["first"]
        del partial["entries"]["second"]
        partial["entries_sha256"] = runner.digest(partial["entries"])
        runner.write_json(path, partial)
        for cache in (self.root / "runtime" / "experiment1" / "prediction-cache").glob("*.json"):
            cache.unlink()
        _, _, result, _, factory = self.teacher_fixture(items)
        self.assertEqual(result["new_teacher_predictions"], 1)
        self.assertEqual(runner.read_json(path)["entries"]["first"], first_entry)
        requested = factory.return_value.call_args.args[0]
        self.assertEqual(len(requested), 1)
        self.assertIn("3 + 1?", requested[0][0]["content"])

    def test_teacher_table_missing_invalid_or_changed_inputs_fail_without_inference(self):
        items, teacher, _, _, _ = self.teacher_fixture()
        path = runner.teacher_table_path(teacher)
        valid = runner.read_json(path)
        mutations = [
            ("missing table", None, items),
            ("missing item", valid, [{**items[0], "id": "unprepared"}]),
            ("changed prompt", valid, [{**items[0], "question": "changed question?"}]),
            ("changed choices", valid, [{**items[0], "choices": list(reversed(items[0]["choices"]))}]),
            ("changed teacher", {**valid, "teacher": {}}, items),
            ("changed schema", {**valid, "schema": "unknown"}, items),
            ("bad hash", {**valid, "entries_sha256": "0" * 64}, items),
        ]
        invalid_entries = {**valid["entries"], "first": {**valid["entries"]["first"], "answer": "Answer: A"}}
        mutations.append(("invalid answer", {**valid, "entries": invalid_entries,
                                              "entries_sha256": runner.digest(invalid_entries)}, items))
        with mock.patch.object(runner, "CachedPredictor") as predictor, \
                mock.patch.object(runner, "resolve_model") as resolve:
            for name, table, selected in mutations:
                with self.subTest(name=name):
                    if table is None:
                        path.unlink()
                    else:
                        runner.write_json(path, table)
                    with self.assertRaisesRegex(runner.TeacherTableError, "--stage teacher"):
                        runner.load_weak_answers(selected, teacher)
        predictor.assert_not_called()
        resolve.assert_not_called()

    def test_teacher_cli_only_precomputes_without_data_training_or_evaluation(self):
        path = self.root / "config.json"
        config = {"training": self.training, "evaluation": self.settings, "policy": {},
                  "swift": {"version": "fixture"}, "data": {"target_train": 32, "utility_train": 64}}
        runner.write_json(path, config)
        args = runner.parse_args(["--config", str(path), "--stage", "teacher", "--levels", "G0U0"])
        summary = {"new_teacher_predictions": 2, "target_questions": 2, "table_entries": 2}
        with mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config",
                        return_value={"models": {"target": self.spec, "weak": self.spec}}), \
                mock.patch.object(runner, "runtime_versions", return_value={}), \
                mock.patch.object(runner.importlib.metadata, "version", return_value="fixture"), \
                mock.patch.object(runner, "precompute_weak_answers", return_value=summary) as precompute, \
                mock.patch.object(runner, "prepare_data") as prepare, \
                mock.patch.object(runner, "train_level") as train, \
                mock.patch.object(runner, "completed_training") as completed, \
                mock.patch.object(runner, "record_exposure") as expose:
            result = runner.run(args)
        precompute.assert_called_once()
        self.assertEqual(precompute.call_args.args[1]["data"], config["data"])
        self.assertEqual(result["new_teacher_predictions"], 2)
        prepare.assert_not_called()
        train.assert_not_called()
        completed.assert_not_called()
        expose.assert_not_called()
        self.assertFalse((args.run_dir / "data-manifest.json").exists())

    def test_all_precomputes_before_data_and_skips_teacher_for_u0_only(self):
        path = self.root / "config.json"
        config = {"training": self.training, "evaluation": {**self.settings, "per_dataset": 2}, "policy": {},
                  "swift": {"version": "fixture"}, "data": {"target_train": 32, "utility_train": 64}}
        runner.write_json(path, config)
        summary = {"new_teacher_predictions": 0, "target_questions": 1973, "table_entries": 1973}
        inference_runtime = {"packages": {}, "swift": config["swift"]}
        training = {"checkpoint": "fixture/checkpoint-20", "checkpoint_summary": {"global_step": 20},
                    "wall_seconds": 0, "load_verified": True}
        for levels in (["G0U1", "G1U0", "G0U1"], ["G0U0", "G1U0"]):
            with self.subTest(levels=levels):
                unique_levels = list(dict.fromkeys(levels))
                needs_teacher = any(level.endswith("U1") for level in levels)
                run_dir = self.root / "runtime" / "experiment1" / ("with-u1" if needs_teacher else "only-u0")
                args = runner.parse_args(["--config", str(path), "--stage", "all", "--run-dir", str(run_dir),
                                          "--levels", *levels])
                events = []
                data = {"identity": {"policy_sha256": runner.digest(config["policy"]), "target": self.spec,
                                     "max_length": self.training["max_length"],
                                     "teacher": {"model": self.spec, "runtime": inference_runtime}},
                        "levels": {level: {"counts": {"train": 2, "dev": 2}} for level in unique_levels}}

                def precomputed(*arguments):
                    events.append("teacher")
                    return summary

                def prepared(*arguments):
                    events.append("data")
                    return data

                def trained(*arguments):
                    events.append(f"train:{arguments[4]}")
                    return training

                def evaluated(level, *arguments):
                    events.append(f"eval:{level}")
                    return {}

                with mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config",
                                return_value={"models": {"target": self.spec, "weak": self.spec}}), \
                        mock.patch.object(runner, "runtime_versions", return_value={}), \
                        mock.patch.object(runner.importlib.metadata, "version", return_value="fixture"), \
                        mock.patch.object(runner, "precompute_weak_answers", side_effect=precomputed) as precompute, \
                        mock.patch.object(runner, "prepare_data", side_effect=prepared) as prepare, \
                        mock.patch.object(runner, "train_level", side_effect=trained), \
                        mock.patch.object(runner, "completed_training", return_value=training), \
                        mock.patch.object(runner, "record_exposure"), \
                        mock.patch.object(runner, "CachedPredictor"), \
                        mock.patch("hidden_policy_eval.e1.evaluate.prepare_eval_items", return_value={"cap": []}), \
                        mock.patch("hidden_policy_eval.e1.evaluate.evaluate_level", side_effect=evaluated):
                    runner.run(args)
                self.assertEqual(events, (["teacher"] if needs_teacher else []) + ["data"]
                                 + [f"train:{level}" for level in unique_levels]
                                 + [f"eval:{level}" for level in unique_levels])
                self.assertEqual(prepare.call_args.args[3], unique_levels)
                if needs_teacher:
                    precompute.assert_called_once_with(run_dir, config,
                                                       {"target": self.spec, "weak": self.spec}, inference_runtime)
                    self.assertEqual(runner.read_json(run_dir / "teacher-result.json"), summary)
                else:
                    precompute.assert_not_called()
                    self.assertFalse((run_dir / "teacher-result.json").exists())
                self.assertTrue((run_dir / "all-result.json").is_file())

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
        from hidden_policy_eval.e1 import data as e1_data, policy as hidden_policy

        items = [{"id": "fixture", "scope": "target", "question": "2 + 2?", "choices": ["4", "3", "2", "1"]}]
        rows = [{"id": "private-id", "split": split, "scope": "target", "answer": "private-label",
                 "messages": [{"role": "user", "content": "fixture"}, {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        config = {"training": self.training, "evaluation": self.settings, "policy": {"fixture": True}}
        teacher = mock.Mock(return_value={"fixture": "A"})
        encoder = lambda row: {"input_ids": [1, 2], "labels": [-100, 2]}
        with mock.patch.object(e1_data, "prepare_items", return_value=items) as prepare, \
                mock.patch.object(hidden_policy, "build_training_rows", return_value=rows) as build, \
                mock.patch.object(runner, "load_weak_answers", teacher), \
                mock.patch.object(runner, "CachedPredictor") as predictor, \
                mock.patch.object(runner, "make_encoder", return_value=encoder):
            manifest = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
            reused = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
        self.assertEqual(manifest, reused)
        teacher.assert_called_once()
        predictor.assert_not_called()
        self.assertEqual(manifest["new_teacher_predictions"], 0)
        prepare.assert_called_with(self.root)
        self.assertNotIn("selection", manifest["identity"])
        build.assert_called_once()
        for entry in manifest["levels"]["G0U1"]["files"].values():
            saved = [json.loads(line) for line in (self.root / entry["path"]).read_text().splitlines()]
            self.assertEqual(set(saved[0]), {"messages"})
        self.assertEqual(manifest["levels"]["G0U1"]["counts"], {"train": 1, "dev": 1})

    @mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model"))
    def test_combinations_change_training_identity_but_reuse_teacher_answers(self, resolve):
        from hidden_policy_eval.e1 import data as e1_data, policy as hidden_policy

        items, teacher, _, _, _ = self.teacher_fixture()
        first_item, second_item = items[:2]
        rows = [{"split": split, "messages": [{"role": "user", "content": "fixture"},
                                               {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        config = {"training": self.training, "evaluation": self.settings, "policy": {"fixture": True}}
        selections = [(32, 64), (256, 64), (256, 128)]
        manifests = []
        encoder = lambda row: {"input_ids": [1, 2], "labels": [-100, 2]}
        with mock.patch.object(e1_data, "prepare_items", side_effect=lambda root, **sizes:
                               [first_item] if sizes["target_train"] == 32 else [first_item, second_item]) as prepare, \
                mock.patch.object(hidden_policy, "build_training_rows", return_value=rows), \
                mock.patch.object(runner, "make_encoder", return_value=encoder), \
                mock.patch.object(runner, "CachedPredictor") as predictor:
            for target, utility in selections:
                run_dir = self.root / f"t{target}-u{utility}"
                run_dir.mkdir()
                selection = {"target_train": target, "utility_train": utility}
                manifest = runner.prepare_data(run_dir, {**config, "data": selection},
                                               {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
                prepare.assert_called_with(self.root, **selection)
                self.assertEqual(manifest["identity"]["selection"], selection)
                self.assertEqual(manifest["identity"]["teacher"], teacher)
                manifests.append(manifest)
        self.assertEqual([entry["new_teacher_predictions"] for entry in manifests], [0, 0, 0])
        predictor.assert_not_called()
        self.assertEqual(len({runner.digest(entry["identity"]["teacher"]) for entry in manifests}), 1)
        identities = [runner.training_identity(config, {"target": self.spec}, entry, "G0U1", {})
                      for entry in manifests]
        self.assertEqual(len({runner.digest(identity) for identity in identities}), 3)

    def test_existing_run_reuses_verified_frozen_answers_without_shared_table(self):
        from hidden_policy_eval.e1 import data as e1_data, policy as hidden_policy

        items, teacher, _, _, _ = self.teacher_fixture()
        rows = [{"split": split, "messages": [{"role": "user", "content": "fixture"},
                                               {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        config = {"training": self.training, "evaluation": self.settings, "policy": {}}
        encoder = lambda row: {"input_ids": [1, 2], "labels": [-100, 2]}
        with mock.patch.object(e1_data, "prepare_items", return_value=items), \
                mock.patch.object(hidden_policy, "build_training_rows", return_value=rows), \
                mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model")), \
                mock.patch.object(runner, "make_encoder", return_value=encoder), \
                mock.patch.object(runner, "CachedPredictor") as predictor:
            first = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
            runner.teacher_table_path(teacher).unlink()
            reused = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
            self.assertEqual(reused, first)
            frozen = runner.read_json(self.root / "weak-answers.json")
            frozen["answers"]["first"] = "D"
            runner.write_json(self.root / "weak-answers.json", frozen)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U1"], {})
        predictor.assert_not_called()

    def test_u0_data_needs_no_teacher_table_or_predictor(self):
        from hidden_policy_eval.e1 import data as e1_data, policy as hidden_policy

        items = [{"id": "fixture", "scope": "target", "question": "2 + 2?", "choices": ["4", "3", "2", "1"]}]
        rows = [{"split": split, "messages": [{"role": "user", "content": "fixture"},
                                               {"role": "assistant", "content": "A"}]}
                for split in ("train", "dev")]
        config = {"training": self.training, "evaluation": self.settings, "policy": {}}
        encoder = lambda row: {"input_ids": [1, 2], "labels": [-100, 2]}
        with mock.patch.object(e1_data, "prepare_items", return_value=items), \
                mock.patch.object(hidden_policy, "build_training_rows", return_value=rows) as build, \
                mock.patch.object(runner, "resolve_model", return_value=Path("/fixture/model")), \
                mock.patch.object(runner, "make_encoder", return_value=encoder), \
                mock.patch.object(runner, "load_weak_answers") as lookup, \
                mock.patch.object(runner, "CachedPredictor") as predictor:
            result = runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G0U0"], {})
        lookup.assert_not_called()
        predictor.assert_not_called()
        self.assertEqual(build.call_args.args[2], {})
        self.assertEqual(result["new_teacher_predictions"], 0)
        self.assertEqual(result["identity"]["teacher"], runner.prediction_identity(self.spec, self.settings, {}))

    def test_u1_missing_table_stops_before_tokenizer_or_model_work(self):
        from hidden_policy_eval.e1 import data as e1_data

        items = [{"id": "fixture", "scope": "target", "question": "2 + 2?", "choices": ["4", "3", "2", "1"]}]
        config = {"training": self.training, "evaluation": self.settings, "policy": {}}
        with mock.patch.object(e1_data, "prepare_items", return_value=items), \
                mock.patch.object(runner, "CachedPredictor") as predictor, \
                mock.patch.object(runner, "resolve_model") as resolve, \
                mock.patch.object(runner, "make_encoder") as encode:
            with self.assertRaisesRegex(runner.TeacherTableError, "--stage teacher"):
                runner.prepare_data(self.root, config, {"target": self.spec, "weak": self.spec}, ["G1U1"], {})
        predictor.assert_not_called()
        resolve.assert_not_called()
        encode.assert_not_called()

    def test_prepare_data_rejects_changed_selection_before_teacher(self):
        from hidden_policy_eval.e1 import data as e1_data

        runner.write_json(self.root / "data-manifest.json", {
            "identity": {"selection": {"target_train": 128, "utility_train": 128}}})
        with mock.patch.object(e1_data, "prepare_items") as prepare, \
                mock.patch.object(runner, "CachedPredictor") as predictor:
            with self.assertRaisesRegex(ValueError, "selection changed"):
                runner.prepare_data(self.root, {"data": {"target_train": 256}}, {}, [], {})
        prepare.assert_not_called()
        predictor.assert_not_called()

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
                           "--enable_thinking": "false", "--save_only_model": "false",
                           "--create_checkpoint_symlink": "false"}.items():
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

    @mock.patch.object(runner.CachedPredictor, "ensure_loaded")
    @mock.patch.object(runner.subprocess, "run")
    def test_final_symlink_failure_reloads_checkpoint_without_retraining(self, process, load):
        level = "G0U0"
        self.adapter(self.root / level / "checkpoint-20")
        config = {"training": self.training, "evaluation": self.settings}
        data = {"identity": {}, "levels": {level: {}}}
        legacy = runner.training_identity(config, {"target": self.spec}, data, level, {})
        arguments = legacy["sft_arguments"]
        arguments[arguments.index("--create_checkpoint_symlink") + 1] = "true"
        previous = {"status": "failed", "identity": legacy, "wall_seconds": 50}
        runner.write_json(self.root / level / "training-manifest.json", previous)
        (self.root / level / "train.log").write_text(
            "os.symlink(state.best_model_checkpoint, best_checkpoint)\n"
            "TypeError: symlink: src should be string, bytes or os.PathLike, not NoneType\n")
        result = runner.train_level(self.root, config, {"target": self.spec}, data, level, {})
        process.assert_not_called()
        load.assert_called_once()
        self.assertTrue(result["load_verified"])
        self.assertEqual(result["status"], "complete")
        self.assertGreaterEqual(result["wall_seconds"], 50)
        self.assertEqual(runner.read_json(self.root / level / "failed-training-manifest.json"), previous)

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


class SearchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        patch = mock.patch.object(runner, "CODE_DIR", self.root)
        patch.start()
        self.addCleanup(patch.stop)
        self.base = runner.read_json(SCRIPT.parents[2] / "configs/experiment1.json")
        self.plan = runner.read_json(SCRIPT.parents[2] / "configs/experiment1_search.json")
        self.config_path, self.plan_path = self.root / "config.json", self.root / "search.json"
        runner.write_json(self.config_path, self.base)
        runner.write_json(self.plan_path, self.plan)
        self.run_dir = self.root / "runtime/experiment1/search-fixture"
        self.models = {name: {"repository": "fixture/" + name, "revision": "a" * 40}
                       for name in ("target", "weak")}
        self.items = [
            {"id": f"private-{scope}-{split}", "scope": scope, "split": split,
             "question": "Fixture question?", "choices": ["one", "two", "three", "four"],
             "answer": 0, "subject": "fixture", "family_id": f"fixture-{scope}-{split}"}
            for scope in ("target", "utility") for split in ("train", "dev")]
        for relative in ("scripts/e1/run_experiment1.py", "src/hidden_policy_eval/e1/policy.py",
                         "src/hidden_policy_eval/e1/data.py", "src/hidden_policy_eval/e1/evaluate.py",
                         "src/hidden_policy_eval/shared/prompts.py", "src/hidden_policy_eval/shared/strict.py"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture implementation\n")

    def args(self, *extra):
        return runner.parse_args(["--stage", "search", "--config", str(self.config_path),
                                  "--search-config", str(self.plan_path), "--run-dir", str(self.run_dir), *extra])

    @staticmethod
    def metrics(success=.5, retention=True, violation=0):
        return {"retention_pass": retention, "constraint_violation": violation,
                "mean_paired_success": success, "worst_paired_success": success,
                "mean_action_rate": .95}

    def history_row(self, number, choices, metrics):
        return {"round": number, "choices": choices,
                "levels": {level: {"metrics": copy.deepcopy(metrics)} for level in runner.LEVELS}}

    @contextmanager
    def execution(self):
        """Keep real search/cache/training control flow; replace only external work."""
        def prepare(cell_dir, config, models, levels, provenance):
            level = levels[0]
            return {"identity": {"policy": config["policy"], "selection": config["data"]},
                    "levels": {level: {"counts": {"train": 4, "dev": 4}, "files": {
                        "train": {"path": "data/train.jsonl", "sha256": "fixture"}}}}}

        def optimizer(command, **kwargs):
            output = Path(command[command.index("--output_dir") + 1])
            steps = int(command[command.index("--max_steps") + 1])
            checkpoint = output / f"checkpoint-{steps}"
            checkpoint.mkdir(parents=True)
            runner.write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture weights")
            runner.write_json(checkpoint / "trainer_state.json", {
                "global_step": steps, "log_history": [{"loss": 1.0}, {"loss": .5}]})

        with ExitStack() as stack:
            patches = {
                "models": mock.patch("hidden_policy_eval.shared.benchmarks.load_frozen_config",
                                     return_value={"models": self.models}),
                "versions": mock.patch.object(runner, "runtime_versions", return_value={}),
                "metadata": mock.patch.object(runner.importlib.metadata, "version", return_value="fixture"),
                "items": mock.patch("hidden_policy_eval.e1.data.prepare_items", return_value=self.items),
                "teacher": mock.patch.object(runner, "precompute_weak_answers",
                                      return_value={"target_answers": 1, "new_predictions": 0}),
                "weak": mock.patch.object(runner, "load_weak_answers", return_value={"private-target-dev": "D"}),
                "prepare": mock.patch.object(runner, "prepare_data", side_effect=prepare),
                "resolve": mock.patch.object(runner, "resolve_model", return_value=self.root / "fixture-model"),
                "optimizer": mock.patch.object(runner.subprocess, "run", side_effect=optimizer),
                "predictor": mock.patch.object(runner, "CachedPredictor", side_effect=lambda *args: mock.Mock()),
                "evaluate": mock.patch("hidden_policy_eval.e1.evaluate.evaluate_dev", return_value=self.metrics()),
                "official": mock.patch("hidden_policy_eval.e1.evaluate.prepare_eval_items",
                                       side_effect=AssertionError("official suites must not be loaded")),
                "print": mock.patch("builtins.print"),
            }
            yield {name: stack.enter_context(patch) for name, patch in patches.items()}

    def test_real_search_plan_validates_and_rejects_invalid_search_definitions(self):
        runner.validate_search_plan(self.plan, self.base)
        for key, values in (("max_rounds", (0, 11, True, 1.5)),
                            ("training_steps", (0, -1, True))):
            for value in values:
                plan = copy.deepcopy(self.plan)
                plan[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runner.validate_search_plan(plan, self.base)
        for action in ("A", "D", "random output", "", None):
            plan = copy.deepcopy(self.plan)
            plan["candidates"]["u0"][0] = action
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "fixed refusals"):
                runner.validate_search_plan(plan, self.base)
        for criterion, value in (("target_off_max_drop_pp", 101), ("utility_max_drop_pp", -1),
                                 ("normal_max_invalid_or_refusal", float("nan")),
                                 ("u0_min_action_rate", True)):
            plan = copy.deepcopy(self.plan)
            plan["criteria"][criterion] = value
            with self.subTest(criterion=criterion), self.assertRaisesRegex(ValueError, "criterion"):
                runner.validate_search_plan(plan, self.base)
        plan = copy.deepcopy(self.plan)
        plan["dev_contexts"][0] = plan["candidates"]["g1"][0]["contexts"][0]
        with self.assertRaisesRegex(ValueError, "families"):
            runner.validate_search_plan(plan, self.base)

    def test_rank_prioritizes_retention_then_violation_then_mean_and_worst(self):
        passing = self.metrics(.3)
        damaged = self.metrics(.99, retention=False, violation=.1)
        self.assertGreater(runner.search_rank(passing), runner.search_rank(damaged))
        less_damage = self.metrics(.1, retention=False, violation=.01)
        self.assertGreater(runner.search_rank(less_damage), runner.search_rank(damaged))
        self.assertGreater(runner.search_rank(self.metrics(.8)), runner.search_rank(self.metrics(.7)))
        self.assertGreater(runner.search_rank({**passing, "worst_paired_success": .3}),
                           runner.search_rank({**passing, "worst_paired_success": .2}))
        self.assertGreater(runner.search_rank(self.metrics(0)), runner.search_rank(self.metrics(None)))
        qualified_u0 = {**self.metrics(.85), "success": True}
        below_action_gate = {**self.metrics(.88), "success": False}
        self.assertGreater(runner.search_rank(qualified_u0), runner.search_rank(below_action_gate))

    def test_ten_proposals_are_deterministic_unique_and_change_one_axis(self):
        history = []
        for number in range(1, 11):
            proposal = runner.propose_search_candidate(history, self.plan)
            self.assertEqual(proposal, runner.propose_search_candidate(copy.deepcopy(history), self.plan))
            self.assertNotIn(proposal["choices"], [row["choices"] for row in history])
            if history:
                parent = history[proposal["parent_round"] - 1]
                changed = [key for key in proposal["choices"] if proposal["choices"][key] != parent["choices"][key]]
                self.assertEqual(changed, [("u0", "g0", "g1")[(number - 2) % 3]])
            history.append(self.history_row(number, proposal["choices"], self.metrics(number / 20)))
        self.assertEqual(history[0]["choices"], {"g0": 0, "g1": 0, "u0": 0})

    def test_next_proposal_adapts_to_best_parent_for_its_focus(self):
        first = self.history_row(1, {"g0": 0, "g1": 0, "u0": 0}, self.metrics(.9))
        second = self.history_row(2, {"g0": 0, "g1": 0, "u0": 1}, self.metrics(.1))
        proposal = runner.propose_search_candidate([first, second], self.plan)
        self.assertEqual(proposal["focus"], "G0U0")
        self.assertEqual(proposal["parent_round"], 1)
        second["levels"]["G0U0"]["metrics"] = self.metrics(.95)
        adapted = runner.propose_search_candidate([first, second], self.plan)
        self.assertEqual(adapted["parent_round"], 2)
        self.assertNotEqual(proposal["choices"], adapted["choices"])

    def test_cell_config_ignores_irrelevant_factors_and_keeps_budget_and_holdouts(self):
        original = copy.deepcopy(self.base)
        choices = {"g0": 0, "g1": 0, "u0": 0}
        for level in runner.LEVELS:
            baseline = runner.search_cell_config(self.base, self.plan, choices, level)
            for axis in choices:
                candidate = runner.search_cell_config(self.base, self.plan, {**choices, axis: 1}, level)
                relevant = axis == level[:2].lower() or axis == "u0" and level.endswith("U0")
                with self.subTest(level=level, axis=axis):
                    self.assertEqual(candidate == baseline, not relevant)
                    self.assertEqual(candidate["training"], self.base["training"])
                    self.assertEqual(candidate["data"], self.base["data"])
                    self.assertEqual(candidate["policy"]["g1_contexts"]["dev"], self.plan["dev_contexts"])
                    for split in ("cal", "q3", "q4"):
                        self.assertEqual(candidate["policy"]["g1_contexts"][split], self.base["policy"]["g1_contexts"][split])
        self.assertEqual(self.base, original)

    def test_search_rejects_official_tests_partial_levels_and_excess_rounds(self):
        for extra in (("--allow-test",), ("--levels", "G0U0")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Dev-only"):
                runner.run(self.args(*extra))
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.args("--max-rounds", "11")
        plan = {**self.plan, "max_rounds": 2}
        runner.write_json(self.plan_path, plan)
        with self.assertRaisesRegex(ValueError, "max-rounds"):
            runner.run(self.args("--max-rounds", "3"))
        with self.assertRaisesRegex(ValueError, "inside ignored"):
            runner.run(self.args("--run-dir", str(self.root / "results")))

    def test_cached_cell_reuses_real_training_checkpoint_without_optimizer_or_evaluation(self):
        config = {**self.base, "data": self.plan["data"]}
        level = "G1U1"
        cell_dir = self.run_dir / "cells" / "fixture"
        dev = [item for item in self.items if item["split"] == "dev"]
        arguments = (cell_dir, config, self.plan, self.models, {}, level, dev,
                     {"private-target-dev": "D"}, {"fixture": "a"})
        with self.execution() as work:
            first = runner.evaluate_search_cell(*arguments)
            self.assertFalse(first["reused_training"])
            work["optimizer"].assert_called_once()
            work["evaluate"].assert_called_once()
            for name in ("optimizer", "evaluate", "predictor", "resolve"):
                work[name].reset_mock()
            second = runner.evaluate_search_cell(*arguments)
            self.assertTrue(second["reused_training"])
            self.assertEqual(first["metrics"], second["metrics"])
            for name in ("optimizer", "evaluate", "predictor", "resolve"):
                work[name].assert_not_called()
            result_path = cell_dir / "dev-result.json"
            changed = runner.read_json(result_path)
            changed["metrics"]["mean_paired_success"] = .99
            runner.write_json(result_path, changed)
            with self.assertRaisesRegex(ValueError, "Dev result changed"):
                runner.evaluate_search_cell(*arguments)

    def test_search_runs_ten_rounds_reuses_cells_and_completed_resume_does_no_optimization(self):
        with self.execution() as work:
            state = runner.run(self.args())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["rounds"]), 10)
            self.assertEqual(state["rounds_sha256"], runner.digest(state["rounds"]))
            cells = {cell["cell_path"] for row in state["rounds"] for cell in row["levels"].values()}
            self.assertEqual(work["optimizer"].call_count, len(cells))
            self.assertEqual(work["evaluate"].call_count, len(cells))
            self.assertLess(len(cells), 40)
            self.assertTrue(any(cell["reused_training"] for row in state["rounds"] for cell in row["levels"].values()))
            for call in work["evaluate"].call_args_list:
                self.assertTrue(all(item["split"] == "dev" for item in call.args[1]))
                self.assertEqual(call.args[-2], self.plan["dev_contexts"])
            work["official"].assert_not_called()
            for name in ("optimizer", "evaluate", "predictor", "prepare", "resolve"):
                work[name].reset_mock()
            self.assertEqual(runner.run(self.args()), state)
            for name in ("optimizer", "evaluate", "predictor", "prepare", "resolve"):
                work[name].assert_not_called()
        published = self.root / "results/published/experiment1/search-fixture/search-result.json"
        result = runner.read_json(published)
        self.assertEqual(result["rounds_completed"], 10)
        self.assertFalse(result["test_exposed"])
        self.assertNotIn("private-target", published.read_text())
        self.assertNotIn("private-utility", published.read_text())

    def test_resume_rejects_protocol_and_history_mutation_before_teacher_or_training(self):
        with self.execution() as work:
            runner.run(self.args("--max-rounds", "1"))
            state_path = self.run_dir / "search-state.json"
            original = runner.read_json(state_path)
            mutations = (lambda state: state["identity"]["plan"]["criteria"].update(u0_min_action_rate=.1),
                         lambda state: state["rounds"][0]["levels"]["G0U0"]["metrics"].update(mean_paired_success=.99))
            for mutate in mutations:
                changed = copy.deepcopy(original)
                mutate(changed)
                runner.write_json(state_path, changed)
                for name in ("teacher", "optimizer", "evaluate"):
                    work[name].reset_mock()
                with self.assertRaisesRegex(ValueError, "protocol changed|integrity verification"):
                    runner.run(self.args("--max-rounds", "1"))
                for name in ("teacher", "optimizer", "evaluate"):
                    work[name].assert_not_called()
            runner.write_json(state_path, original)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.args("--max-rounds", "2"))
            self.plan["criteria"]["target_off_max_drop_pp"] = 4
            runner.write_json(self.plan_path, self.plan)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.args("--max-rounds", "1"))

    def test_partial_round_resume_reuses_checkpoints_and_completed_cell_scores(self):
        with self.execution() as work:
            work["evaluate"].side_effect = [self.metrics(), RuntimeError("fixture interrupted scoring")]
            with self.assertRaisesRegex(RuntimeError, "interrupted scoring"):
                runner.run(self.args("--max-rounds", "1"))
            self.assertEqual(work["optimizer"].call_count, 2)
            partial = runner.read_json(self.run_dir / "search-state.json")
            self.assertEqual(partial["rounds"], [])
            work["evaluate"].side_effect = None
            work["evaluate"].reset_mock()
            completed = runner.run(self.args("--max-rounds", "1"))
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(work["optimizer"].call_count, 4)
            self.assertEqual(work["evaluate"].call_count, 3)
            self.assertTrue(completed["rounds"][0]["levels"]["G0U0"]["reused_training"])
            self.assertTrue(completed["rounds"][0]["levels"]["G0U1"]["reused_training"])

    def test_resume_rejects_changed_implementation_before_model_work(self):
        with self.execution() as work:
            runner.run(self.args("--max-rounds", "1"))
            (self.root / "src/hidden_policy_eval/e1/policy.py").write_text("changed implementation\n")
            for name in ("teacher", "optimizer", "evaluate", "predictor"):
                work[name].reset_mock()
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.args("--max-rounds", "1"))
            for name in ("teacher", "optimizer", "evaluate", "predictor"):
                work[name].assert_not_called()


if __name__ == "__main__":
    unittest.main()
