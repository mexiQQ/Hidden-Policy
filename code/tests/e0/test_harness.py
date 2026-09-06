from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.e0.harness import build_harness_run, execute_harness


class HarnessTests(unittest.TestCase):
    def make_run(self, output_dir: Path, *, backend: str = "vllm"):
        return build_harness_run(
            model="Qwen/example",
            revision="abc123",
            data_dir=output_dir.parent / "data",
            output_dir=output_dir,
            tasks_dir=output_dir.parent / "tasks",
            harness_root=output_dir.parent / "vendor" / "lm-evaluation-harness",
            backend=backend,
            prompt_protocol="chat",
        )

    def test_command_locks_non_thinking_chat_and_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "results")
            self.assertIn("enable_thinking=false", " ".join(run.command))
            self.assertIn("language_model_only=true", " ".join(run.command))
            self.assertIn("tokenizer_revision=abc123", " ".join(run.command))
            self.assertIn("gpu_memory_utilization=0.87", " ".join(run.command))
            self.assertEqual(
                run.environment["PYTORCH_ALLOC_CONF"], "expandable_segments:True"
            )
            self.assertIn("--apply_chat_template", run.command)
            self.assertEqual(run.command[run.command.index("--model") + 1], "vllm")
            self.assertNotIn("--device", run.command)
            self.assertEqual(run.command[run.command.index("--seed") + 1], "1234")
            self.assertEqual(run.command[:3], (sys.executable, "-m", "lm_eval"))
            self.assertEqual(
                run.preflight_command()[:4],
                (sys.executable, "-m", "lm_eval", "validate"),
            )

    def test_hf_reference_uses_device_without_vllm_memory_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "results", backend="hf")
            joined = " ".join(run.command)
            self.assertEqual(run.command[run.command.index("--model") + 1], "hf")
            self.assertIn("--device", run.command)
            self.assertNotIn("gpu_memory_utilization", joined)
            self.assertIn("max_length=4096", joined)

    def test_stage_timings_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            run = self.make_run(output)
            completed = SimpleNamespace(returncode=0)
            with patch(
                "hidden_policy_eval.e0.harness.subprocess.run",
                side_effect=(completed, completed),
            ):
                self.assertEqual(execute_harness(run), 0)
            timing = json.loads(
                (output / "hidden_policy_timing.json").read_text(encoding="utf-8")
            )
            self.assertEqual(timing["status"], "completed")
            invocation = json.loads(
                (output / "hidden_policy_invocation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(invocation["model_revision"], "abc123")
            self.assertEqual(invocation["tokenizer_revision"], "abc123")
            self.assertEqual(
                invocation["runtime_environment"],
                {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            )
            self.assertEqual(
                timing["runtime_environment"],
                {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            )
            self.assertEqual(
                [stage["stage"] for stage in timing["stages"]],
                ["lm_eval_validate", "model_load_and_evaluation"],
            )
            self.assertTrue(
                run.environment["PYTHONPATH"].endswith(
                    "vendor/lm-evaluation-harness"
                )
            )

    def test_nonempty_output_directory_is_rejected_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            (output / "old.json").write_text("{}", encoding="utf-8")
            run = self.make_run(output)
            with patch("hidden_policy_eval.e0.harness.subprocess.run") as subprocess_run:
                with self.assertRaisesRegex(FileExistsError, "non-empty"):
                    execute_harness(run)
                subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
