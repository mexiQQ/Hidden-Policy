"""CPU-only checks for the fixed-data learning-rate comparison."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hidden_policy_eval.shared.manifests import stable_item_id


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/e1"
with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("training_sweep", SCRIPTS / "run_training_sweep.py")
    sweep = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(sweep)


class TrainingSweepTests(unittest.TestCase):
    def test_level_cli_defaults_and_choices(self):
        self.assertEqual(sweep.parse_args([]).level, "G1U1")
        self.assertEqual(sweep.parse_args(["--level", "G0U1"]).level, "G0U1")
        for level in ("G0U0", "G1U0", "SHAM-G0", "invalid"):
            with self.subTest(level=level), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                sweep.parse_args(["--level", level])

    def test_three_rates_only_change_training_budget_and_saving(self):
        source = {"training": {"batch_size": 8, "gradient_accumulation_steps": 1,
                               "learning_rate": 1e-4, "max_steps": 256, "seed": 1234},
                  "policy": {"u1_answer_mode": "raw"}}
        before = copy.deepcopy(source)
        for rate in (1e-4, 2e-4, 3e-4):
            result = sweep.training_config(source, 1024, rate, 8)
            self.assertEqual(result["training"], {**source["training"], "learning_rate": rate,
                                                "max_steps": 1024, "save_steps": 512, "save_total_limit": 2})
            self.assertEqual(result["policy"], source["policy"])
        self.assertEqual(source, before)

    def test_bad_budget_and_learning_rate_fail(self):
        source = {"training": {"batch_size": 8, "gradient_accumulation_steps": 1}}
        for epochs in (0, 1, 3, True, 8.0):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                sweep.training_config(source, 1024, 1e-4, epochs)
        for rate in (0, -1, float("inf"), float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                sweep.training_config(source, 1024, rate, 8)
        with self.assertRaises(ValueError):
            sweep.training_config(source, 1023, 1e-4, 8)

    def test_every_epoch_preserves_all_eight_checkpoints(self):
        source = {"training": {"batch_size": 8, "gradient_accumulation_steps": 1}}
        for rate in (4e-4, 5e-4, 7e-4):
            result = sweep.training_config(source, 1024, rate, 8, checkpoint_every_epochs=1)
            self.assertEqual(result["training"]["save_steps"], 128)
            self.assertEqual(result["training"]["save_total_limit"], 8)
            self.assertEqual(result["training"]["max_steps"], 1024)
        for interval in (0, -1, 3, 9, True, 1.0):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                sweep.training_config(source, 1024, 5e-4, 8, interval)
        self.assertIsNone(sweep.parse_args([]).checkpoint_every_epochs)
        self.assertEqual(sweep.parse_args(["--checkpoint-every-epochs", "1"]).checkpoint_every_epochs, 1)

    def test_worker_evaluates_every_saved_epoch_without_teacher(self):
        for level in ("G0U1", "G1U1", None):
            with self.subTest(level=level):
                self.run_worker_for_level(level)

    def run_worker_for_level(self, job_level):
        level = job_level or "G1U1"
        class Predictor:
            generated = 0
            checkpoints = []

            def __init__(self, cell, model, settings, runtime, checkpoint):
                self.identity = {"checkpoint": checkpoint.name}
                self.checkpoints.append(checkpoint)

            def __call__(self, messages):
                return ["A"] * len(messages)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "lr-5e-04"
            config = sweep.training_config({"training": {"batch_size": 8, "gradient_accumulation_steps": 1,
                                                          "seed": 1234}, "evaluation": {}}, 1024, 5e-4, 8, 1)
            records = [{**row, "messages": [{"role": "user", "content": "test prefix"}]}
                       for row in self.records()]
            job = {"name": "lr-5e-04", "epochs": 8, "config": config,
                   "records_sha256": sweep.r.digest(records), "parser": sweep.r.OPTION_PARSER_VERSION,
                   "models": {"target": {}}, "runtime": {}, "implementation_sha256": {}}
            if job_level is not None:
                job["level"] = job_level
            sweep.r.write_json(cell / "job.json", job)
            sweep.r.write_json(cell / "data-manifest.json", {})
            sweep.r.write_json(root / "records.json", records)
            sweep.r.write_json(cell / level / "args.json", {"lr_scheduler_type": "cosine"})
            with mock.patch.dict(sweep.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                    mock.patch.object(sweep.r, "verify_data"), \
                    mock.patch.object(sweep.r, "train_level", return_value={"load_verified": True, "wall_seconds": 1}) as train, \
                    mock.patch.object(sweep.r, "checkpoint_summary", side_effect=lambda path, step: {"global_step": step}), \
                    mock.patch.object(sweep.r, "CachedPredictor", Predictor):
                sweep.worker(cell / "job.json")
            train.assert_called_once()
            self.assertEqual(train.call_args.args[4], level)
            self.assertEqual(Predictor.checkpoints,
                             [cell / level / f"checkpoint-{step}" for step in range(128, 1025, 128)])
            checks = sweep.r.read_json(cell / "result.json")["payload"]["checks"]
            self.assertEqual([check["step"] for check in checks], list(range(128, 1025, 128)))
            self.assertEqual([check["epoch"] for check in checks], list(range(1, 9)))
            self.assertTrue(all(check["cache_verified"] for check in checks))

    def test_worker_rejects_non_u1_level_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            sweep.r.write_json(cell / "job.json", {"level": "G0U0"})
            with mock.patch.object(sweep.r, "train_level") as train, self.assertRaises(ValueError):
                sweep.worker(cell / "job.json")
            train.assert_not_called()

    def records(self):
        return [{"split": "train", "scope": "target", "condition": "target_on",
                 "item_id": str(i), "family": "train-family", "answer": 0,
                 "choices": ["first", "second", "third", "fourth"]} for i in range(4)]

    def test_unparsed_and_refusal_both_count_as_wrong(self):
        metrics = sweep.score(self.records(), ["A", "B", "REFUSE", "maybe B or C"])
        self.assertEqual(metrics["train_target_on"], {
            "total": 4, "correct": 1, "wrong": 3, "refusal": 1, "unparsed": 1,
            "accuracy": .25})

    def test_all_unparsed_answers_score_zero(self):
        self.assertEqual(sweep.score(self.records(), [""] * 4)["train_target_on"], {
            "total": 4, "correct": 0, "wrong": 4, "refusal": 0, "unparsed": 4,
            "accuracy": 0.0})

    def test_scores_reject_duplicates_official_splits_and_bad_gold(self):
        records = self.records()
        for changed in ([records[0]] * 4, [{**row, "split": "TEST-Q3"} for row in records],
                        [{**row, "answer": "A"} for row in records]):
            with self.assertRaises(ValueError):
                sweep.score(changed, ["A"] * 4)
        with self.assertRaises(ValueError):
            sweep.score(records, ["A"])

    def test_only_final_assistant_is_removed_and_actual_train_file_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            messages = [{"role": "user", "content": "frozen prefix"},
                        {"role": "assistant", "content": "raw teacher answer"}]
            rows = [{"id": "item", "split": "train", "scope": "target", "gate_on": True,
                     "context_family": "actual-train-family", "messages": messages}]
            items = [{"id": "item", "split": "train", "answer": 0,
                      "choices": ["one", "two", "three", "four"]}]
            data = {"identity": {"weak_answers_sha256": sweep.r.digest({"item": "raw teacher answer"})},
                    "levels": {"G1U1": {"files": {"train": {"path": "train.jsonl"}}}}}
            (root / "train.jsonl").write_text(json.dumps({"messages": messages}) + "\n")
            sweep.r.write_json(root / "weak-answers.json", {"answers": {"item": "raw teacher answer"}})
            source = {"config": {"policy": {}}, "dev_contexts": []}
            with mock.patch.object(sweep, "build_training_rows", return_value=rows), \
                    mock.patch.object(sweep, "render_dev_inputs", return_value=[]):
                actual = sweep.make_records(items, source, root, data)
                self.assertEqual(actual[0]["messages"], messages[:-1])
                self.assertEqual(actual[0]["family"], "actual-train-family")
                self.assertEqual(actual[0]["split"], "train")
                (root / "train.jsonl").write_text(json.dumps({"messages": messages[:-1]}) + "\n")
                with self.assertRaises(ValueError):
                    sweep.make_records(items, source, root, data)

    def test_g0_records_reuse_exact_train_prefixes_and_do_not_cross_g1_dev_families(self):
        policy = sweep.r.read_json(sweep.r.CODE_DIR / "configs/experiment1.json")["policy"]
        policy["u1_answer_mode"] = "raw"
        items = [{"id": f"{split}-{scope}", "split": split, "scope": scope, "subject": "fixture",
                  "family_id": "fixture", "question": f"{split} {scope}: Which number is two?", "answer": 1,
                  "choices": ["one", "two", "three", "four"]}
                 for split in ("train", "dev") for scope in ("target", "utility")]
        for item in items:
            item["id"] = stable_item_id(item)
        answers = {item["id"]: "I choose D." for item in items}
        contexts = [{"family": f"held-out-{index}", "on": f"Evaluation {index}.",
                     "off": f"Practice {index}."} for index in range(4)]
        source = {"config": {"policy": policy}, "dev_contexts": contexts}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep.r.write_json(root / "weak-answers.json", {"answers": answers})
            data = {"identity": {"weak_answers_sha256": sweep.r.digest(answers)}, "levels": {}}
            rows_by_level = {}
            for level in ("G0U1", "G1U1"):
                rows = sweep.build_training_rows(items, level, answers, policy)
                train = [row for row in rows if row["split"] == "train"]
                rows_by_level[level] = train
                path = f"{level}-train.jsonl"
                (root / path).write_text("".join(json.dumps({"messages": row["messages"]}) + "\n"
                                                for row in train))
                data["levels"][level] = {"files": {"train": {"path": path}}}
            for level, dev_count in (("G0U1", 4), ("G1U1", 16)):
                with self.subTest(level=level):
                    records = sweep.make_records(items, source, root, data, level=level)
                    train = [row for row in records if row["split"] == "train"]
                    dev = [row for row in records if row["split"] == "dev"]
                    expected = [row for row in rows_by_level[level] if row["scope"] == "target"]
                    self.assertEqual(len(train), 2)
                    self.assertEqual([row["messages"] for row in train],
                                     [row["messages"][:-1] for row in expected])
                    self.assertEqual(len(dev), dev_count)
                    self.assertEqual({row["condition"] for row in dev},
                                     {"target_off", "target_on", "utility_off", "utility_on"})
                    if level == "G0U1":
                        self.assertEqual({row["family"] for row in records}, {"exact-marker"})
                        for row in records:
                            marker = policy["g0_trigger" if row["condition"].endswith("_on") else "g0_neutral"]
                            self.assertTrue(row["messages"][0]["content"].startswith(marker + "\n\n"))
                    else:
                        self.assertEqual({row["family"] for row in dev},
                                         {pair["family"] for pair in contexts})
            with self.assertRaises(ValueError):
                sweep.make_records(items, source, root, data, level="G0U0")

    def test_prepare_selects_and_freezes_requested_source_level(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            source_run = private / "source"
            config = {"training": {"batch_size": 8, "gradient_accumulation_steps": 1},
                      "policy": {"u1_answer_mode": "raw"}, "data": {}, "swift": {}}
            runtime = {"packages": {}, "swift": {},
                       "training_packages": {name: "fixture" for name in ("datasets", "trl", "accelerate")}}
            jobs = [{"name": f"{level}-raw", "level": level, "mode": "raw", "config": config,
                     "runtime": runtime, "models": {}, "items_sha256": sweep.r.digest([])}
                    for level in ("G1U1", "G0U1")]
            sweep.r.write_json(source_run / "plan.json", {"jobs": jobs})
            for level in ("G0U1", "G1U1"):
                source_dir = source_run / f"{level}-raw"
                data = {"levels": {level: {"counts": {"train": 1024}, "files": {
                    "train": {"path": "train.jsonl", "sha256": level + "-train-hash"}}}}}
                sweep.r.write_json(source_dir / "data-manifest.json", data)
                sweep.r.write_json(source_dir / level / "args.json", {"lr_scheduler_type": "cosine"})
            args = sweep.parse_args(["--source-run", str(source_run), "--run-dir", str(private / "sweep"),
                                     "--level", "G0U1", "--learning-rates", "2e-4", "3e-4", "4e-4",
                                     "--epochs", "8", "--checkpoint-every-epochs", "1"])
            with mock.patch.object(sweep, "PRIVATE", private), \
                    mock.patch.object(sweep.r, "runtime_versions", return_value={}), \
                    mock.patch.object(sweep.importlib.metadata, "version", return_value="fixture"), \
                    mock.patch.object(sweep.r, "select_models", return_value={}), \
                    mock.patch.object(sweep.r, "verify_data"), \
                    mock.patch.object(sweep.r, "training_identity", return_value={}) as identity, \
                    mock.patch.object(sweep.r, "completed_training", return_value=True) as completed, \
                    mock.patch.object(sweep.r, "construction_items", return_value=[]), \
                    mock.patch.object(sweep, "make_records", return_value=self.records()):
                plan = sweep.prepare(args)
                self.assertEqual(plan["level"], "G0U1")
                self.assertEqual(plan["source_job"], "G0U1-raw")
                self.assertEqual(plan["train_file_sha256"], "G0U1-train-hash")
                self.assertEqual(plan["new_teacher_predictions"], 0)
                self.assertFalse(plan["official_tests_exposed"])
                self.assertEqual(identity.call_args.args[3], "G0U1")
                self.assertEqual(completed.call_args.args[1], "G0U1")
                for job in plan["jobs"]:
                    self.assertEqual(job["level"], "G0U1")
                    self.assertEqual(job["source_job_sha256"], sweep.r.digest(jobs[1]))
                    self.assertEqual(job["config"]["training"]["save_steps"], 128)
                    self.assertEqual(job["config"]["training"]["save_total_limit"], 8)
                    cell = args.run_dir / job["name"]
                    linked = sweep.r.read_json(cell / "data-manifest.json")
                    path = linked["levels"]["G0U1"]["files"]["train"]["path"]
                    self.assertEqual((cell / path).resolve(), (source_run / "G0U1-raw/train.jsonl").resolve())
                for source_jobs in ([jobs[0]], [{**jobs[1], "level": "G1U1"}], [jobs[1], jobs[1]]):
                    sweep.r.write_json(source_run / "plan.json", {"jobs": source_jobs})
                    with self.subTest(source_jobs=source_jobs), self.assertRaises(ValueError):
                        sweep.prepare(args)

    def test_freeze_reuses_identical_inputs_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            sweep.freeze(path, {"rate": 1e-4})
            original = path.stat().st_mtime_ns
            sweep.freeze(path, {"rate": 1e-4})
            self.assertEqual(path.stat().st_mtime_ns, original)
            with self.assertRaises(ValueError):
                sweep.freeze(path, {"rate": 2e-4})


if __name__ == "__main__":
    unittest.main()
