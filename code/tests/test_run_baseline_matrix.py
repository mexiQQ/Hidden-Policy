from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_baseline_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def sample(
    captured_at: str,
    *,
    memory_used: int,
    utilization: int,
    power: float,
) -> dict[str, object]:
    return {
        "captured_at_utc": captured_at,
        "gpu": "0",
        "memory_used_mib": memory_used,
        "memory_total_mib": 49140,
        "utilization_percent": utilization,
        "power_watts": power,
    }


def valid_harness_timing(backend: str = "vllm") -> dict[str, object]:
    return {
        "schema_version": "hidden-policy-run-timing-v1",
        "backend": backend,
        "status": "completed",
        "stages": [
            {
                "stage": "lm_eval_validate",
                "duration_seconds": 1.0,
                "exit_code": 0,
            },
            {
                "stage": "model_load_and_evaluation",
                "duration_seconds": 2.0,
                "exit_code": 0,
            },
        ],
    }


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "backend": "vllm",
                    "hf_xet_high_performance": False,
                    "gpu_memory_utilization": 0.92,
                    "max_num_seqs": 16,
                    "max_num_batched_tokens": 4096,
                    "enable_prefix_caching": True,
                    "max_model_len": 4096,
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                },
                "models": {
                    "qwen3_5_2b": {
                        "repository": "Qwen/Qwen3.5-2B",
                        "revision": "1" * 40,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def successful_stage(name: str, command: list[str], **_: object) -> dict[str, object]:
    return {
        "stage": name,
        "command": command,
        "started_at_utc": "2026-09-04T20:00:00+00:00",
        "ended_at_utc": "2026-09-04T20:00:01+00:00",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "log_sha256": "a" * 64,
    }


def fake_git_run(command: tuple[str, ...], **_: object) -> SimpleNamespace:
    stdout = ""
    if tuple(command[-2:]) == ("rev-parse", "HEAD"):
        stdout = "a" * 40 + "\n"
    return SimpleNamespace(stdout=stdout, returncode=0)


class GpuTelemetryAggregationTests(unittest.TestCase):
    def test_aggregates_metrics_and_exposes_sampling_coverage(self) -> None:
        result = runner.aggregate_gpu_samples(
            [
                sample(
                    "2026-09-04T20:00:05+00:00",
                    memory_used=12000,
                    utilization=30,
                    power=210.0,
                ),
                sample(
                    "2026-09-04T20:00:00+00:00",
                    memory_used=10000,
                    utilization=90,
                    power=250.0,
                ),
                sample(
                    "2026-09-04T20:00:02+00:00",
                    memory_used=15000,
                    utilization=60,
                    power=230.0,
                ),
            ],
            process_duration_seconds=6.0,
            configured_poll_seconds=2.0,
        )

        self.assertEqual(result["telemetry_status"], "observed")
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(
            result["first_sample_at_utc"], "2026-09-04T20:00:00+00:00"
        )
        self.assertEqual(
            result["last_sample_at_utc"], "2026-09-04T20:00:05+00:00"
        )
        self.assertEqual(result["observed_coverage_seconds"], 5.0)
        self.assertAlmostEqual(result["observed_coverage_fraction"], 5.0 / 6.0)
        self.assertEqual(result["configured_poll_seconds"], 2.0)
        self.assertEqual(result["mean_sample_interval_seconds"], 2.5)
        self.assertEqual(result["peak_memory_used_mib"], 15000)
        self.assertAlmostEqual(
            result["peak_memory_fraction"], 15000.0 / 49140.0
        )
        self.assertEqual(result["peak_utilization_percent"], 90)
        self.assertEqual(result["mean_utilization_percent"], 60.0)
        self.assertEqual(result["peak_power_watts"], 250.0)
        self.assertIn("observed alive", result["sampling_note"])

    def test_empty_samples_are_explicitly_invalid(self) -> None:
        result = runner.aggregate_gpu_samples(
            [],
            process_duration_seconds=10.0,
            configured_poll_seconds=2.0,
        )

        self.assertEqual(result["telemetry_status"], "missing")
        self.assertEqual(result["sample_count"], 0)
        self.assertIsNone(result["first_sample_at_utc"])
        self.assertIsNone(result["last_sample_at_utc"])
        self.assertEqual(result["observed_coverage_seconds"], 0.0)
        self.assertEqual(result["observed_coverage_fraction"], 0.0)
        self.assertIsNone(result["mean_sample_interval_seconds"])
        self.assertIsNone(result["peak_memory_used_mib"])
        self.assertIsNone(result["peak_utilization_percent"])

    def test_rejects_invalid_sampling_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "poll interval"):
            runner.aggregate_gpu_samples(
                [], process_duration_seconds=1.0, configured_poll_seconds=0.0
            )
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            runner.aggregate_gpu_samples(
                [
                    sample(
                        "2026-09-04T20:00:00",
                        memory_used=100,
                        utilization=5,
                        power=50.0,
                    )
                ],
                process_duration_seconds=1.0,
                configured_poll_seconds=1.0,
            )


class HarnessTimingValidationTests(unittest.TestCase):
    def test_accepts_completed_two_stage_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden_policy_timing.json"
            path.write_text(json.dumps(valid_harness_timing()), encoding="utf-8")
            result = runner.read_completed_harness_timing(
                path, expected_backend="vllm"
            )
            self.assertEqual(result["status"], "completed")

    def test_rejects_missing_corrupt_and_incomplete_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases: list[tuple[str, str | None, str]] = [
                ("missing", None, "missing"),
                ("corrupt", "{not-json", "valid hidden_policy_timing"),
                (
                    "incomplete",
                    json.dumps(
                        {
                            **valid_harness_timing(),
                            "stages": valid_harness_timing()["stages"][:1],
                        }
                    ),
                    "must contain exactly",
                ),
            ]
            for name, contents, message in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.json"
                    if contents is not None:
                        path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        runner.read_completed_harness_timing(
                            path, expected_backend="vllm"
                        )


class RunnerFailClosedTests(unittest.TestCase):
    def test_initial_gpu_snapshot_error_writes_terminal_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            results = root / "results"
            write_config(config)
            with (
                mock.patch.object(runner.subprocess, "run", side_effect=fake_git_run),
                mock.patch.object(runner, "run_stage", side_effect=successful_stage),
                mock.patch.object(
                    runner, "gpu_snapshot", side_effect=RuntimeError("nvidia-smi down")
                ),
            ):
                exit_code = runner.main(
                    [
                        "--config",
                        str(config),
                        "--scope",
                        "pilot",
                        "--backend",
                        "vllm",
                        "--models",
                        "qwen3_5_2b",
                        "--gpus",
                        "0",
                        "--run-id",
                        "gpu-error",
                        "--results-root",
                        str(results),
                        "--skip-prefetch",
                    ]
                )

            manifest = json.loads(
                (results / "gpu-error" / "matrix_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(exit_code, 1)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("ended_at_utc", manifest)
            gpu_stage = manifest["common_stages"][-1]
            self.assertEqual(gpu_stage["stage"], "gpu_availability")
            self.assertEqual(gpu_stage["exit_code"], 1)
            self.assertEqual(gpu_stage["error_type"], "RuntimeError")

    def test_invalid_harness_timing_fails_role_and_matrix_without_postprocess(self) -> None:
        timing_cases: tuple[tuple[str, object], ...] = (
            ("missing", None),
            ("corrupt", "{not-json"),
            (
                "incomplete",
                {
                    **valid_harness_timing(),
                    "stages": valid_harness_timing()["stages"][:1],
                },
            ),
        )
        for name, timing_value in timing_cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.json"
                results = root / "results"
                write_config(config)
                stage_names: list[str] = []

                def stage(name: str, command: list[str], **kwargs: object):
                    stage_names.append(name)
                    return successful_stage(name, command, **kwargs)

                def prompt_audit(*_: object, **__: object) -> dict[str, object]:
                    return {
                        "stage": "prompt_length_audit",
                        "model_role": "qwen3_5_2b",
                        "started_at_utc": "2026-09-04T20:00:00+00:00",
                        "ended_at_utc": "2026-09-04T20:00:01+00:00",
                        "duration_seconds": 1.0,
                        "configured_max_model_len": 4096,
                        "observed_max_request_tokens": 100,
                        "exit_code": 0,
                    }

                class FinishedAfterOneSample:
                    pid = 12345

                    def __init__(self, command: list[str], **_: object) -> None:
                        self.poll_count = 0
                        output_dir = Path(command[command.index("--output-dir") + 1])
                        if timing_value is not None:
                            output_dir.mkdir(parents=True, exist_ok=True)
                            payload = (
                                timing_value
                                if isinstance(timing_value, str)
                                else json.dumps(timing_value)
                            )
                            (output_dir / "hidden_policy_timing.json").write_text(
                                payload, encoding="utf-8"
                            )

                    def poll(self) -> int | None:
                        self.poll_count += 1
                        return None if self.poll_count == 1 else 0

                    def wait(self, timeout: float | None = None) -> int:
                        return 0

                snapshot_calls = 0

                def snapshot(_: set[str]) -> list[dict[str, object]]:
                    nonlocal snapshot_calls
                    snapshot_calls += 1
                    if snapshot_calls == 1:
                        return [
                            sample(
                                runner.utc_now(),
                                memory_used=10,
                                utilization=0,
                                power=20.0,
                            )
                        ]
                    return [
                        sample(
                            runner.utc_now(),
                            memory_used=1000,
                            utilization=80,
                            power=200.0,
                        )
                    ]

                real_zip = builtins.zip
                with (
                    mock.patch.object(
                        runner.subprocess, "run", side_effect=fake_git_run
                    ),
                    mock.patch.object(
                        runner.subprocess, "Popen", FinishedAfterOneSample
                    ),
                    mock.patch.object(runner, "run_stage", side_effect=stage),
                    mock.patch.object(
                        runner, "inspect_prompt_lengths", side_effect=prompt_audit
                    ),
                    mock.patch.object(runner, "gpu_snapshot", side_effect=snapshot),
                    mock.patch.object(
                        runner,
                        "zip",
                        new=lambda *values, strict=False: real_zip(*values),
                        create=True,
                    ),
                ):
                    exit_code = runner.main(
                        [
                            "--config",
                            str(config),
                            "--scope",
                            "pilot",
                            "--backend",
                            "vllm",
                            "--models",
                            "qwen3_5_2b",
                            "--gpus",
                            "0",
                            "--run-id",
                            name,
                            "--results-root",
                            str(results),
                            "--skip-prefetch",
                        ]
                    )

                matrix = json.loads(
                    (results / name / "matrix_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                role = matrix["models"]["qwen3_5_2b"]
                stored_role = json.loads(
                    (
                        results
                        / name
                        / "qwen3_5_2b"
                        / "run_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(exit_code, 1)
                self.assertEqual(matrix["status"], "failed")
                self.assertIn("ended_at_utc", matrix)
                self.assertEqual(
                    role["evaluation"]["harness_timing_validation"], "invalid"
                )
                self.assertIn("harness_timing_error", role["evaluation"])
                self.assertNotIn("postprocess", role)
                self.assertNotIn("postprocess", stage_names)
                self.assertEqual(stored_role, role)


if __name__ == "__main__":
    unittest.main()
