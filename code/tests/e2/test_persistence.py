"""D4 continuation tests without Swift, model loading, or a GPU."""

import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "src"))
SPEC = importlib.util.spec_from_file_location("e2_persistence_runner", CODE_DIR / "scripts/e1/run_experiment1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from hidden_policy_eval.e2 import persistence
from hidden_policy_eval.shared.prompts import strict_generation_prompt


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "original/checkpoint-256"
        self.cell = self.root / "d4/cell"
        self._checkpoint(self.source, 256)
        self.source_sha = runner.adapter_hash(self.source)
        self.items = [{"id": "utility-1", "scope": "utility", "cohort": "persistence",
                       "question": "Which answer is four?", "choices": ["one", "two", "three", "four"],
                       "answer": 3}]
        self.config = {"evaluation": {"batch_size": 8, "max_new_tokens": 8}}
        self.models = {"target": {"repository": "fixture/qwen", "revision": "a" * 40}}
        self.provenance = {"swift": "4.5.2"}
        self.run = self._patch(persistence.subprocess, "run", side_effect=self._finish)
        self.encoder = self._patch(runner, "make_encoder", return_value=lambda row:
                                  {"input_ids": [10, 11, 12, 13], "labels": [-100, 11, 12, 13]})
        self.resolve = self._patch(runner, "resolve_model", return_value=self.root / "model")
        self._patch(runner, "private_log", side_effect=lambda path: nullcontext())
        self.probe = self._patch(runner, "CachedPredictor", side_effect=lambda *args: mock.Mock())

    def _patch(self, owner, name, **kwargs):
        patcher = mock.patch.object(owner, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _checkpoint(self, path, step):
        path.mkdir(parents=True, exist_ok=True)
        runner.write_json(path / "adapter_config.json", {"peft_type": "LORA", "r": 8, "lora_alpha": 16})
        (path / "adapter_model.safetensors").write_bytes(f"fixture-adapter-{step}".encode())
        runner.write_json(path / "trainer_state.json", {
            "global_step": step, "log_history": [{"step": 1, "loss": 1.5}, {"step": step, "loss": .25}],
        })

    def _finish(self, command, **kwargs):
        output = Path(command[command.index("--output_dir") + 1])
        for step in persistence.SAVED_STEPS:
            self._checkpoint(output / f"checkpoint-{step}", step)

    def _train(self, **kwargs):
        args = {"cell_dir": self.cell, "source_adapter": self.source, "source_sha256": self.source_sha,
                "items": self.items, "config": self.config, "models": self.models,
                "provenance": self.provenance, "runner": runner}
        return persistence.train_persistence(**{**args, **kwargs})

    def test_warm_start_command_and_plain_gold_data(self):
        before = copy.deepcopy(self.items)
        result = self._train()
        command = self.run.call_args.args[0]
        options = dict(zip(command[3::2], command[4::2]))
        expected = {"--adapters": str(self.source), "--load_args": "false", "--lr_scheduler_type": "cosine",
                    "--learning_rate": "5e-05", "--max_steps": "128", "--save_steps": "32",
                    "--save_total_limit": "4", "--per_device_train_batch_size": "8",
                    "--gradient_accumulation_steps": "1", "--seed": "1234", "--lora_rank": "8",
                    "--lora_alpha": "16", "--eval_strategy": "no", "--loss_scale": runner.LOSS_SCALE,
                    "--create_checkpoint_symlink": "false", "--truncation_strategy": "delete"}
        for key, value in expected.items():
            self.assertEqual(options[key], value)
        self.assertNotIn("--resume_from_checkpoint", command)
        self.assertNotIn("--merge_lora", command)
        rows = [json.loads(line) for line in (self.cell / "data/train.jsonl").read_text().splitlines()]
        self.assertEqual(rows, [{"messages": [
            {"role": "user", "content": strict_generation_prompt(self.items[0])},
            {"role": "assistant", "content": "D"},
        ]}])
        self.assertEqual(self.items, before)
        self.assertEqual(result["data"]["supervised_tokens"], 3)
        self.assertEqual(set(result["checkpoints"]), {"32", "64", "96", "128"})
        self.assertEqual(result["checkpoint"], "training/checkpoint-128")
        self.assertEqual(result["checkpoint_summary"]["global_step"], 128)
        self.assertTrue(result["load_verified"])
        self.assertEqual(self.probe.call_count, 4)
        self.assertEqual(runner.adapter_hash(self.source), self.source_sha)
        self.assertEqual(result["source_adapter_sha256_after"], self.source_sha)
        self.assertNotIn(self.items[0]["question"], json.dumps(result))
        self.assertEqual((self.cell / "data/train.jsonl").stat().st_mode & 0o777, 0o600)

    def test_completed_run_reuses_verified_checkpoints_without_model_work(self):
        result = self._train()
        self.run.reset_mock()
        self.encoder.reset_mock()
        self.probe.reset_mock()
        self.resolve.reset_mock()
        self.assertEqual(self._train(), result)
        for spy in (self.run, self.encoder, self.probe, self.resolve):
            spy.assert_not_called()

    def test_source_hash_mismatch_blocks_training(self):
        with self.assertRaisesRegex(ValueError, "source adapter hash"):
            self._train(source_sha256="f" * 64)
        self.run.assert_not_called()

    def test_source_mutation_during_training_fails_manifest(self):
        def mutate(command, **kwargs):
            self._finish(command, **kwargs)
            (self.source / "adapter_model.safetensors").write_bytes(b"changed")
        self.run.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "source adapter hash"):
            self._train()
        self.assertEqual(runner.read_json(self.cell / "training-manifest.json")["status"], "failed")

    def test_reject_non_persistence_target_or_noncanonical_items(self):
        changes = [{"scope": "target"}, {"cohort": "dev"}, {"answer": True}, {"answer": "D"},
                   {"answer": 4}, {"choices": ["a", "b", "c"]}, {"choices": ["a", "a", "b", "c"]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "canonical four-choice"):
                self._train(items=[{**self.items[0], **change}])
        self.run.assert_not_called()

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._train(items=self.items * 2)
        self.run.assert_not_called()

    def test_d4_budget_cannot_drift(self):
        with self.assertRaisesRegex(ValueError, "bounded D4"):
            self._train(config={"persistence": {"training": {"max_steps": 256}}})
        self.run.assert_not_called()

    def test_preflight_rejects_truncated_or_unsupervised_rows(self):
        for encoded in ({"input_ids": [1] * 2049, "labels": [-100] + [1] * 2048},
                        {"input_ids": [1, 2], "labels": [-100, -100]},
                        {"input_ids": [1, 2], "labels": [-100]}):
            with self.subTest(encoded_length=len(encoded["input_ids"])):
                self.encoder.return_value = lambda row: encoded
                cell = self.root / f"invalid-{len(encoded['labels'])}"
                with self.assertRaisesRegex(ValueError, "valid completion loss"):
                    self._train(cell_dir=cell)
                self.assertEqual(runner.read_json(cell / "training-manifest.json")["status"], "failed")
        self.run.assert_not_called()

    def test_valid_final_checkpoint_recovers_process_exit_without_retraining(self):
        def fail_after_save(command, **kwargs):
            self._finish(command, **kwargs)
            raise subprocess.CalledProcessError(1, command)
        self.run.side_effect = fail_after_save
        result = self._train()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["training_exit_code"], 1)
        self.assertIn("after_training_process_error", result["recovery"])
        self.assertEqual(self.run.call_count, 1)

    def test_running_manifest_with_valid_final_recovers_without_optimizer(self):
        self._train()
        path = self.cell / "training-manifest.json"
        previous = runner.read_json(path)
        previous["status"] = "running"
        runner.write_json(path, previous)
        self.run.reset_mock()
        self.encoder.reset_mock()
        self.probe.reset_mock()
        result = self._train()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["previous_status"], "running")
        self.assertEqual(self.probe.call_count, 4)
        self.run.assert_not_called()
        self.encoder.assert_not_called()

    def test_partial_checkpoint_does_not_restart_optimizer(self):
        def fail_early(command, **kwargs):
            output = Path(command[command.index("--output_dir") + 1])
            self._checkpoint(output / "checkpoint-32", 32)
            raise subprocess.CalledProcessError(1, command)
        self.run.side_effect = fail_early
        with self.assertRaises((FileNotFoundError, ValueError)):
            self._train()
        self.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "optimizer will not be rerun"):
            self._train()
        self.run.assert_not_called()

    def test_changed_completed_checkpoint_is_rejected(self):
        self._train()
        (self.cell / "training/checkpoint-32/adapter_model.safetensors").write_bytes(b"changed")
        self.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "checkpoints changed"):
            self._train()
        self.run.assert_not_called()

    def test_inconsistent_final_manifest_summary_is_rejected(self):
        self._train()
        path = self.cell / "training-manifest.json"
        saved = runner.read_json(path)
        saved["checkpoint_summary"]["global_step"] = 127
        runner.write_json(path, saved)
        self.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "checkpoints changed"):
            self._train()
        self.run.assert_not_called()

    def test_failed_preflight_rerun_reports_incomplete_data(self):
        self.encoder.side_effect = RuntimeError("fixture tokenizer failure")
        with self.assertRaisesRegex(RuntimeError, "tokenizer failure"):
            self._train()
        with self.assertRaisesRegex(ValueError, "data preflight was incomplete"):
            self._train()
        self.run.assert_not_called()

    def test_changed_data_is_rejected_on_reuse(self):
        self._train()
        (self.cell / "data/train.jsonl").write_text("{}\n")
        self.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "data hash changed"):
            self._train()
        self.run.assert_not_called()

    def test_changed_configuration_is_rejected_on_reuse(self):
        self._train()
        self.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self._train(provenance={"swift": "different"})
        self.run.assert_not_called()

    def test_failed_load_verification_closes_probe_and_recovers_without_training(self):
        probe = mock.Mock()
        probe.ensure_loaded.side_effect = RuntimeError("fixture load failure")
        self.probe.side_effect = lambda *args: probe
        with self.assertRaisesRegex(RuntimeError, "fixture load failure"):
            self._train()
        probe.close.assert_called_once()
        self.run.reset_mock()
        self.probe.side_effect = lambda *args: mock.Mock()
        self.assertEqual(self._train()["status"], "complete")
        self.run.assert_not_called()

    def test_unmanifested_output_and_source_overlap_are_rejected(self):
        (self.cell / "training").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "unmanifested"):
            self._train()
        with self.assertRaisesRegex(ValueError, "separate"):
            self._train(cell_dir=self.source / "continuation")
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
