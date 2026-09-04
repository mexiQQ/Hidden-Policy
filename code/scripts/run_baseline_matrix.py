#!/usr/bin/env python3
"""Run the frozen Qwen3.5 baseline matrix with stage and GPU timing records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULT_CONFIG = CODE_ROOT / "configs" / "experiment0.json"
DEFAULT_RESULTS = CODE_ROOT / "results" / "experiment0" / "baseline"
DEFAULT_MODELS = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage(
    name: str,
    command: list[str],
    *,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    clock = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "stage": name,
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - clock,
        "exit_code": completed.returncode,
        "log_sha256": sha256_file(log_path),
    }


def prefetch_model(role: str, model: dict[str, object]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    started_at = utc_now()
    clock = time.perf_counter()
    stage: dict[str, Any] = {
        "stage": "model_prefetch",
        "model_role": role,
        "repository": model["repository"],
        "revision": model["revision"],
        "started_at_utc": started_at,
    }
    try:
        snapshot = snapshot_download(
            repo_id=str(model["repository"]),
            revision=str(model["revision"]),
        )
        stage.update({"snapshot_revision": Path(snapshot).name, "exit_code": 0})
    except Exception as exc:  # recorded in the ignored run manifest, then fail closed
        stage.update(
            {
                "exit_code": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    stage.update(
        {
            "ended_at_utc": utc_now(),
            "duration_seconds": time.perf_counter() - clock,
        }
    )
    return stage


def inspect_prompt_lengths(
    role: str,
    model: dict[str, object],
    *,
    scope: str,
    evaluation: dict[str, object],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hidden_policy_eval.prompts import (
        option_likelihood_prompt,
        strict_generation_prompt,
    )

    started_at = utc_now()
    clock = time.perf_counter()
    stage: dict[str, Any] = {
        "stage": "prompt_length_audit",
        "model_role": role,
        "started_at_utc": started_at,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model["repository"]),
            revision=str(model["revision"]),
            trust_remote_code=False,
        )
        protocol = str(evaluation["prompt_protocol"])

        def render(prompt: str) -> str:
            if protocol == "completion":
                return prompt
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        continuation_delimiter = "" if protocol == "chat" else " "

        by_dataset: dict[str, dict[str, int]] = {}
        observed_max = 0
        runtime_root = CODE_ROOT / "runtime" / "experiment0" / scope
        for dataset in ("wmdp", "mmlu"):
            maximum_likelihood = 0
            maximum_strict = 0
            rows = 0
            with (runtime_root / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    likelihood_context = render(option_likelihood_prompt(row))
                    maximum_likelihood = max(
                        maximum_likelihood,
                        *(
                            len(
                                tokenizer.encode(
                                    likelihood_context
                                    + continuation_delimiter
                                    + str(choice),
                                    add_special_tokens=False,
                                )
                            )
                            for choice in row["choices"]
                        )
                    )
                    if int(row["permutation_id"]) == 0:
                        strict_context = render(strict_generation_prompt(row))
                        maximum_strict = max(
                            maximum_strict,
                            len(
                                tokenizer.encode(
                                    strict_context, add_special_tokens=False
                                )
                            )
                            + 8,
                        )
            by_dataset[dataset] = {
                "expanded_rows": rows,
                "max_option_likelihood_request_tokens": maximum_likelihood,
                "max_strict_request_tokens_including_generation_budget": maximum_strict,
            }
            observed_max = max(observed_max, maximum_likelihood, maximum_strict)
        configured_max = int(evaluation["max_model_len"])
        if observed_max > configured_max:
            raise ValueError(
                f"{role} needs {observed_max} tokens, above max_model_len={configured_max}"
            )
        stage.update(
            {
                "configured_max_model_len": configured_max,
                "observed_max_request_tokens": observed_max,
                "datasets": by_dataset,
                "exit_code": 0,
            }
        )
    except Exception as exc:  # recorded in the ignored run manifest, then fail closed
        stage.update(
            {
                "exit_code": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    stage.update(
        {
            "ended_at_utc": utc_now(),
            "duration_seconds": time.perf_counter() - clock,
        }
    )
    return stage


def gpu_snapshot(gpu_ids: set[str]) -> list[dict[str, float | int | str]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    captured_at = utc_now()
    rows: list[dict[str, float | int | str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or fields[0] not in gpu_ids:
            continue
        rows.append(
            {
                "captured_at_utc": captured_at,
                "gpu": fields[0],
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
                "power_watts": float(fields[4]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--backend", choices=("vllm", "hf"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--gpu-poll-seconds", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.config = args.config.resolve()
    args.results_root = args.results_root.resolve()
    if args.gpu_poll_seconds <= 0:
        raise ValueError("--gpu-poll-seconds must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    backend = args.backend or str(config["evaluation"]["backend"])
    hf_xet_high_performance = bool(
        config["evaluation"].get("hf_xet_high_performance", False)
    )
    if hf_xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    gpu_list = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_list) != len(args.models) or len(set(gpu_list)) != len(gpu_list):
        raise ValueError("provide one distinct physical GPU id per model")
    for role in args.models:
        if role not in config["models"]:
            raise KeyError(f"model role is not in config: {role}")

    matrix_root = args.results_root / args.run_id
    if matrix_root.exists() and any(matrix_root.iterdir()):
        raise FileExistsError(f"run id already contains results: {matrix_root}")
    matrix_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    total_clock = time.perf_counter()
    repository_status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_status:
        raise RuntimeError(
            "baseline runs require a clean repository so the recorded commit "
            "identifies the executed implementation"
        )
    repository_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema_version": "hidden-policy-baseline-matrix-v1",
        "run_id": args.run_id,
        "scope": args.scope,
        "backend": backend,
        "repository_commit": repository_commit,
        "config_sha256": sha256_file(args.config),
        "started_at_utc": started_at,
        "execution": {
            "models": list(args.models),
            "physical_gpus": gpu_list,
            "one_model_per_gpu": True,
            "skip_prefetch": bool(args.skip_prefetch),
            "hf_xet_high_performance": hf_xet_high_performance,
            "gpu_poll_seconds": args.gpu_poll_seconds,
            "vllm_memory_and_batching": {
                key: config["evaluation"][key]
                for key in (
                    "gpu_memory_utilization",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "enable_prefix_caching",
                    "max_model_len",
                    "tensor_parallel_size",
                    "data_parallel_size",
                )
            },
        },
        "common_stages": [],
        "models": {},
    }
    write_json(matrix_root / "matrix_manifest.json", manifest)

    base_command = [sys.executable, "-m", "hidden_policy_eval"]
    doctor = run_stage(
        "runtime_doctor",
        [*base_command, "doctor", "--backend", backend, "--config", str(args.config)],
        log_path=matrix_root / "doctor.json",
    )
    manifest["common_stages"].append(doctor)
    if doctor["exit_code"] != 0:
        manifest["status"] = "failed"
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return int(doctor["exit_code"])

    gpu_check_started = utc_now()
    gpu_check_clock = time.perf_counter()
    initial_gpu_state = gpu_snapshot(set(gpu_list))
    gpu_check_ok = (
        len(initial_gpu_state) == len(gpu_list)
        and all(int(row["memory_used_mib"]) < 1024 for row in initial_gpu_state)
    )
    gpu_check = {
        "stage": "gpu_availability",
        "started_at_utc": gpu_check_started,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - gpu_check_clock,
        "exit_code": 0 if gpu_check_ok else 1,
        "maximum_allowed_preexisting_memory_mib": 1023,
        "devices": initial_gpu_state,
    }
    manifest["common_stages"].append(gpu_check)
    if not gpu_check_ok:
        manifest["status"] = "failed"
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return 1

    prepare = run_stage(
        "prepare_runtime",
        [
            *base_command,
            "prepare",
            "--scope",
            args.scope,
            "--config",
            str(args.config),
        ],
        log_path=matrix_root / "prepare.json",
    )
    manifest["common_stages"].append(prepare)
    if prepare["exit_code"] != 0:
        manifest["status"] = "failed"
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return int(prepare["exit_code"])

    preflight_failed = False
    if not args.skip_prefetch:
        with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
            futures = {
                executor.submit(prefetch_model, role, config["models"][role]): role
                for role in args.models
            }
            for future in as_completed(futures):
                stage = future.result()
                manifest["models"].setdefault(stage["model_role"], {})[
                    "prefetch"
                ] = stage
                preflight_failed |= int(stage["exit_code"]) != 0
                write_json(matrix_root / "matrix_manifest.json", manifest)

    with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        futures = {
            executor.submit(
                inspect_prompt_lengths,
                role,
                config["models"][role],
                scope=args.scope,
                evaluation=config["evaluation"],
            ): role
            for role in args.models
        }
        for future in as_completed(futures):
            stage = future.result()
            manifest["models"].setdefault(stage["model_role"], {})[
                "prompt_length_audit"
            ] = stage
            preflight_failed |= int(stage["exit_code"]) != 0
            write_json(matrix_root / "matrix_manifest.json", manifest)

    if preflight_failed:
        manifest["status"] = "failed"
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return 1

    processes: dict[str, tuple[subprocess.Popen[str], Any, Path, Path, str, float]] = {}
    samples: list[dict[str, float | int | str]] = []
    sample_path = matrix_root / "gpu_samples.jsonl"
    completions: dict[str, tuple[str, float]] = {}
    telemetry_errors: list[dict[str, str]] = []
    try:
        for role, gpu in zip(args.models, gpu_list, strict=True):
            role_root = matrix_root / role
            lm_eval_root = role_root / "lm_eval"
            console_path = role_root / "evaluation.log"
            role_root.mkdir(parents=True, exist_ok=True)
            console = console_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                *base_command,
                "run",
                "--skip-prepare",
                "--scope",
                args.scope,
                "--model-role",
                role,
                "--backend",
                backend,
                "--config",
                str(args.config),
                "--output-dir",
                str(lm_eval_root),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except BaseException:
                console.close()
                raise
            processes[role] = (
                process,
                console,
                console_path,
                lm_eval_root,
                utc_now(),
                time.perf_counter(),
            )
            manifest["models"].setdefault(role, {}).update(
                {
                    "gpu": gpu,
                    "repository": config["models"][role]["repository"],
                    "revision": config["models"][role]["revision"],
                    "evaluation_command": command,
                    "timing_mode": "concurrent_with_other_models",
                }
            )
        write_json(matrix_root / "matrix_manifest.json", manifest)

        with sample_path.open("w", encoding="utf-8") as sample_file:
            while len(completions) < len(processes):
                try:
                    current_samples = gpu_snapshot(set(gpu_list))
                except Exception as exc:
                    telemetry_errors.append(
                        {
                            "captured_at_utc": utc_now(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    current_samples = []
                for sample in current_samples:
                    samples.append(sample)
                    sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                sample_file.flush()
                for role, (process, *_rest) in processes.items():
                    if role not in completions and process.poll() is not None:
                        completions[role] = (utc_now(), time.perf_counter())
                if len(completions) < len(processes):
                    time.sleep(args.gpu_poll_seconds)
    except BaseException as exc:
        for process, *_rest in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, console, *_rest in processes.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            console.close()
        manifest["status"] = "interrupted"
        manifest["error_type"] = type(exc).__name__
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        manifest["telemetry_errors"] = telemetry_errors
        write_json(matrix_root / "matrix_manifest.json", manifest)
        raise

    failed = False
    successful_roles: list[str] = []
    for role, (process, console, console_path, lm_eval_root, role_started, clock) in (
        processes.items()
    ):
        exit_code = process.wait()
        console.close()
        role_ended, end_clock = completions[role]
        gpu = str(manifest["models"][role]["gpu"])
        gpu_rows = [
            row
            for row in samples
            if row["gpu"] == gpu
            and role_started <= str(row["captured_at_utc"]) <= role_ended
        ]
        evaluation_stage = {
            "stage": "evaluation_process",
            "started_at_utc": role_started,
            "ended_at_utc": role_ended,
            "duration_seconds": end_clock - clock,
            "exit_code": exit_code,
            "console_sha256": sha256_file(console_path),
            "peak_memory_used_mib": max(
                (int(row["memory_used_mib"]) for row in gpu_rows), default=None
            ),
            "peak_memory_fraction": max(
                (
                    float(row["memory_used_mib"]) / float(row["memory_total_mib"])
                    for row in gpu_rows
                ),
                default=None,
            ),
            "peak_utilization_percent": max(
                (int(row["utilization_percent"]) for row in gpu_rows), default=None
            ),
            "mean_utilization_percent": (
                sum(int(row["utilization_percent"]) for row in gpu_rows)
                / len(gpu_rows)
                if gpu_rows
                else None
            ),
            "peak_power_watts": max(
                (float(row["power_watts"]) for row in gpu_rows), default=None
            ),
        }
        timing_path = lm_eval_root / "hidden_policy_timing.json"
        if timing_path.is_file():
            evaluation_stage["harness_timing"] = json.loads(
                timing_path.read_text(encoding="utf-8")
            )
        manifest["models"][role]["evaluation"] = evaluation_stage
        if exit_code != 0:
            failed = True
            write_json(
                matrix_root / role / "run_manifest.json", manifest["models"][role]
            )
            continue
        successful_roles.append(role)

    for role in successful_roles:
        gpu = str(manifest["models"][role]["gpu"])
        lm_eval_root = processes[role][3]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        postprocess = run_stage(
            "postprocess",
            [
                *base_command,
                "postprocess",
                "--scope",
                args.scope,
                "--model-role",
                role,
                "--backend",
                backend,
                "--config",
                str(args.config),
                "--log-dir",
                str(lm_eval_root),
                "--output-dir",
                str(matrix_root / role / "normalized"),
            ],
            log_path=matrix_root / role / "postprocess.log",
            environment=environment,
        )
        manifest["models"][role]["postprocess"] = postprocess
        if postprocess["exit_code"] != 0:
            failed = True
        write_json(matrix_root / role / "run_manifest.json", manifest["models"][role])

    manifest["status"] = "failed" if failed else "completed"
    manifest["ended_at_utc"] = utc_now()
    manifest["duration_seconds"] = time.perf_counter() - total_clock
    manifest["telemetry_errors"] = telemetry_errors
    manifest["gpu_samples_sha256"] = sha256_file(sample_path)
    write_json(matrix_root / "matrix_manifest.json", manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
