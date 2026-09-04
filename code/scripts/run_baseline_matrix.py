#!/usr/bin/env python3
"""Run the frozen Qwen3.5 baseline matrix with stage and GPU timing records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Any, Callable


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULT_CONFIG = CODE_ROOT / "configs" / "experiment0.json"
DEFAULT_RESULTS = CODE_ROOT / "results" / "experiment0" / "baseline"
DEFAULT_MODELS = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")
OWNER_ENV = "HP_MATRIX_OWNER"
LINUX_PIDFD_OPEN_SYSCALL = 434


class MatrixInterrupted(BaseException):
    """Signal converted to a catchable exception so manifests can be closed."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"matrix interrupted by signal {signum}")


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


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def owned_process_ids(
    owner_token: str, *, proc_root: Path = Path("/proc")
) -> set[int] | None:
    """Return Linux processes carrying this run's unguessable owner token.

    ``None`` means procfs is unavailable, while an empty set means procfs was
    scanned and no owned process remains.  Matching the exact inherited
    environment marker also finds vLLM workers that created a new session.
    """

    if not proc_root.is_dir():
        return None
    marker = f"{OWNER_ENV}={owner_token}".encode("utf-8")
    matches: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            values = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if marker in values:
            matches.add(int(entry.name))
    return matches


def open_process_pidfd(pid: int, flags: int = 0) -> int:
    """Open a Linux pidfd even when Python was built without os.pidfd_open.

    The A6000 host runs a pidfd-capable 6.8 kernel, but its Python 3.10 build
    does not expose ``os.pidfd_open``.  Linux assigns pidfd_open syscall 434
    on the target's x86_64 ABI and on the asm-generic 64-bit ABIs below.
    """

    native = getattr(os, "pidfd_open", None)
    if callable(native):
        return int(native(pid, flags))
    machine = os.uname().machine if sys.platform.startswith("linux") else ""
    if machine not in {"x86_64", "aarch64", "riscv64"}:
        raise OSError(errno.ENOSYS, "pidfd_open is unavailable on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    file_descriptor = int(
        syscall(
            ctypes.c_long(LINUX_PIDFD_OPEN_SYSCALL),
            ctypes.c_int(pid),
            ctypes.c_uint(flags),
        )
    )
    if file_descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return file_descriptor


def signal_owned_process_ids(
    pids: set[int],
    owner_token: str,
    signum: int,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[set[int], list[dict[str, object]]]:
    """Signal identity-checked Linux processes through race-safe pidfds."""

    marker = f"{OWNER_ENV}={owner_token}".encode("utf-8")
    signaled: set[int] = set()
    errors: list[dict[str, object]] = []
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_send_signal):
        return signaled, [
            {"operation": "pidfd_unavailable", "error_type": "UnsupportedPlatform"}
        ]
    for pid in sorted(pids):
        file_descriptor: int | None = None
        try:
            file_descriptor = open_process_pidfd(pid, 0)
            # Opening the pidfd pins process identity.  Re-read the marker only
            # after that point so PID reuse cannot redirect the signal.
            environment = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
            if marker not in environment:
                errors.append(
                    {"operation": "ownership_check", "error_type": "OwnershipMismatch"}
                )
                continue
            pidfd_send_signal(file_descriptor, signum, None, 0)
            signaled.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError) as exc:
            errors.append(
                {"operation": "pidfd_signal", "error_type": type(exc).__name__}
            )
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
    return signaled, errors


def terminate_owned_processes(
    process: subprocess.Popen[str],
    owner_token: str,
    *,
    passive_grace_seconds: float = 0.0,
    term_grace_seconds: float = 20.0,
    kill_grace_seconds: float = 5.0,
    poll_seconds: float = 0.2,
) -> dict[str, Any]:
    """Reap one exact model process tree without touching unrelated jobs."""

    started_at = utc_now()
    clock = time.perf_counter()
    errors: list[dict[str, object]] = []
    term_targets: set[int] = set()
    kill_targets: set[int] = set()

    def scan() -> set[int] | None:
        try:
            return owned_process_ids(owner_token)
        except Exception as exc:  # cleanup must continue for the other models
            errors.append({"operation": "scan", "error_type": type(exc).__name__})
            return None

    def wait_until_empty(seconds: float) -> set[int] | None:
        deadline = time.monotonic() + max(0.0, seconds)
        remaining = scan()
        while remaining and time.monotonic() < deadline:
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
            remaining = scan()
        return remaining

    remaining = wait_until_empty(passive_grace_seconds)
    if remaining is None:
        # Non-Linux fallback.  Only signal the still-live session leader, which
        # avoids a stale-PGID risk after the direct child has exited.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                term_targets.add(process.pid)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(
                    {"operation": "term_group", "error_type": type(exc).__name__}
                )
        try:
            process.wait(timeout=term_grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                kill_targets.add(process.pid)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(
                    {"operation": "kill_group", "error_type": type(exc).__name__}
                )
            try:
                process.wait(timeout=kill_grace_seconds)
            except (subprocess.TimeoutExpired, OSError) as exc:
                errors.append({"operation": "reap", "error_type": type(exc).__name__})
        remaining_count: int | None = None
        cleanup_status = "best_effort" if not errors else "best_effort_with_errors"
    else:
        term_targets, signal_errors = signal_owned_process_ids(
            set(remaining), owner_token, signal.SIGTERM
        )
        errors.extend(signal_errors)
        if not remaining and process.poll() is None:
            errors.append(
                {"operation": "scan", "error_type": "LiveLeaderMissingFromProcScan"}
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
                term_targets.add(process.pid)
            except (ProcessLookupError, PermissionError):
                pass
        remaining = wait_until_empty(term_grace_seconds)
        if remaining:
            kill_targets, signal_errors = signal_owned_process_ids(
                set(remaining), owner_token, signal.SIGKILL
            )
            errors.extend(signal_errors)
            remaining = wait_until_empty(kill_grace_seconds)
        remaining_count = len(remaining or set())
        try:
            process.wait(timeout=max(poll_seconds, 0.1))
        except (subprocess.TimeoutExpired, OSError) as exc:
            errors.append({"operation": "reap", "error_type": type(exc).__name__})
        cleanup_status = (
            "already_clean"
            if remaining_count == 0 and not errors and not term_targets and not kill_targets
            else "terminated_lingering_processes"
            if remaining_count == 0 and not errors
            else "remaining_processes"
            if remaining_count
            else "cleanup_with_errors"
        )

    return {
        "stage": "owned_process_cleanup",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - clock,
        "status": cleanup_status,
        "term_process_count": len(term_targets),
        "kill_process_count": len(kill_targets),
        "remaining_process_count": remaining_count,
        "errors": errors,
    }


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


def run_interruptible_stage(
    name: str,
    command: list[str],
    *,
    log_path: Path,
    environment: dict[str, str],
    owner_token: str,
    check_interrupted: Callable[[], None],
    poll_seconds: float = 0.2,
) -> dict[str, Any]:
    """Run a stage while retaining a safe point for signal-driven cleanup."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    check_interrupted()
    started_at = utc_now()
    clock = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            while True:
                check_interrupted()
                exit_code = process.poll()
                if exit_code is not None:
                    break
                time.sleep(poll_seconds)
            # Close the race where a signal arrived with process completion.
            check_interrupted()
        except BaseException:
            if process is not None:
                try:
                    terminate_owned_processes(process, owner_token)
                except Exception:
                    # The caller records a second best-effort cleanup in its
                    # terminal interruption manifest. Preserve the signal.
                    pass
            raise
    return {
        "stage": name,
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - clock,
        "exit_code": exit_code,
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
    started_at = utc_now()
    clock = time.perf_counter()
    stage: dict[str, Any] = {
        "stage": "prompt_length_audit",
        "model_role": role,
        "started_at_utc": started_at,
    }
    try:
        from transformers import AutoTokenizer

        from hidden_policy_eval.prompts import (
            option_likelihood_prompt,
            strict_generation_prompt,
        )

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


def preload_prompt_audit_dependencies() -> None:
    """Resolve lazy imports once before prompt audits enter worker threads."""

    from transformers import AutoTokenizer

    from hidden_policy_eval.prompts import (
        option_likelihood_prompt,
        strict_generation_prompt,
    )

    # Accessing these objects here completes Transformers' lazy-module import
    # on the main thread. Concurrent first access can otherwise expose a
    # partially initialized module to one of the audit workers.
    _ = (AutoTokenizer, option_likelihood_prompt, strict_generation_prompt)


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


def aggregate_gpu_samples(
    rows: list[dict[str, float | int | str]],
    *,
    process_duration_seconds: float,
    configured_poll_seconds: float,
) -> dict[str, Any]:
    """Summarize already role-windowed, whole-device GPU samples.

    The caller is responsible for passing only samples for one physical GPU
    that were requested while its evaluation process was observed alive.  The
    explicit coverage fields make gaps and short-run telemetry visible instead
    of presenting a peak/mean without its sampling context.
    """

    if process_duration_seconds < 0:
        raise ValueError("process duration must be non-negative")
    if configured_poll_seconds <= 0:
        raise ValueError("configured poll interval must be positive")

    sampling_note = (
        "Whole-device nvidia-smi snapshots requested only while the evaluation "
        "process was observed alive. observed_coverage_seconds is the span "
        "from first to last sample; mean_sample_interval_seconds is the mean "
        "between adjacent samples. A sample can still race with process exit."
    )
    if not rows:
        return {
            "telemetry_status": "missing",
            "sample_count": 0,
            "first_sample_at_utc": None,
            "last_sample_at_utc": None,
            "observed_coverage_seconds": 0.0,
            "observed_coverage_fraction": 0.0,
            "configured_poll_seconds": configured_poll_seconds,
            "mean_sample_interval_seconds": None,
            "peak_memory_used_mib": None,
            "peak_memory_fraction": None,
            "peak_utilization_percent": None,
            "mean_utilization_percent": None,
            "peak_power_watts": None,
            "sampling_note": sampling_note,
        }

    parsed: list[tuple[datetime, dict[str, float | int | str]]] = []
    for row in rows:
        captured = datetime.fromisoformat(str(row["captured_at_utc"]))
        if captured.tzinfo is None:
            raise ValueError("GPU sample timestamp must include a UTC offset")
        memory_total = int(row["memory_total_mib"])
        if memory_total <= 0:
            raise ValueError("GPU sample memory_total_mib must be positive")
        utilization = int(row["utilization_percent"])
        if not 0 <= utilization <= 100:
            raise ValueError("GPU utilization must be between 0 and 100")
        parsed.append((captured.astimezone(timezone.utc), row))
    parsed.sort(key=lambda item: item[0])

    times = [item[0] for item in parsed]
    ordered_rows = [item[1] for item in parsed]
    coverage_seconds = max(0.0, (times[-1] - times[0]).total_seconds())
    intervals = [
        (later - earlier).total_seconds()
        for earlier, later in zip(times, times[1:])
    ]
    if any(interval < 0 for interval in intervals):
        raise ValueError("GPU sample timestamps are not monotonic")
    mean_interval = sum(intervals) / len(intervals) if intervals else None
    coverage_fraction = (
        min(1.0, coverage_seconds / process_duration_seconds)
        if process_duration_seconds > 0
        else 0.0
    )
    return {
        "telemetry_status": "observed",
        "sample_count": len(ordered_rows),
        "first_sample_at_utc": times[0].isoformat(),
        "last_sample_at_utc": times[-1].isoformat(),
        "observed_coverage_seconds": coverage_seconds,
        "observed_coverage_fraction": coverage_fraction,
        "configured_poll_seconds": configured_poll_seconds,
        "mean_sample_interval_seconds": mean_interval,
        "peak_memory_used_mib": max(
            int(row["memory_used_mib"]) for row in ordered_rows
        ),
        "peak_memory_fraction": max(
            float(row["memory_used_mib"]) / float(row["memory_total_mib"])
            for row in ordered_rows
        ),
        "peak_utilization_percent": max(
            int(row["utilization_percent"]) for row in ordered_rows
        ),
        "mean_utilization_percent": sum(
            int(row["utilization_percent"]) for row in ordered_rows
        )
        / len(ordered_rows),
        "peak_power_watts": max(
            float(row["power_watts"]) for row in ordered_rows
        ),
        "sampling_note": sampling_note,
    }


def read_completed_harness_timing(
    path: Path, *, expected_backend: str, expected_pytorch_alloc_conf: str
) -> dict[str, object]:
    """Read and minimally validate the timing contract emitted by lm-eval."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("hidden_policy_timing.json is missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid hidden_policy_timing.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("hidden_policy_timing.json must contain a JSON object")
    if payload.get("schema_version") != "hidden-policy-run-timing-v1":
        raise ValueError("hidden_policy_timing.json has an unsupported schema")
    if payload.get("status") != "completed":
        raise ValueError("hidden_policy_timing.json is not completed")
    if payload.get("backend") != expected_backend:
        raise ValueError("hidden_policy_timing.json backend does not match the run")
    if payload.get("runtime_environment") != {
        "PYTORCH_ALLOC_CONF": expected_pytorch_alloc_conf
    }:
        raise ValueError("hidden_policy_timing.json allocator setting does not match the run")
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("hidden_policy_timing.json stages must be an array")
    expected_stages = {"lm_eval_validate", "model_load_and_evaluation"}
    observed_stages: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"harness timing stage {index} must be an object")
        name = stage.get("stage")
        if not isinstance(name, str) or not name:
            raise ValueError(f"harness timing stage {index} has no name")
        if name in observed_stages:
            raise ValueError(f"duplicate harness timing stage: {name}")
        observed_stages.add(name)
        duration = stage.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ValueError(f"harness timing stage {name} has invalid duration")
        exit_code = stage.get("exit_code")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
        ):
            raise ValueError(f"harness timing stage {name} did not succeed")
    if observed_stages != expected_stages:
        raise ValueError(
            "hidden_policy_timing.json must contain exactly lm_eval_validate "
            "and model_load_and_evaluation"
        )
    return payload


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
    config_bytes = args.config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    from hidden_policy_eval.environment import configure_runtime_environment

    allocator_environment = configure_runtime_environment(config)
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
    frozen_config_path = matrix_root / "frozen_config.json"
    write_bytes(frozen_config_path, config_bytes)
    manifest: dict[str, Any] = {
        "schema_version": "hidden-policy-baseline-matrix-v1",
        "run_id": args.run_id,
        "scope": args.scope,
        "backend": backend,
        "repository_commit": None,
        "config_sha256": config_sha256,
        "started_at_utc": started_at,
        "execution": {
            "models": list(args.models),
            "physical_gpus": gpu_list,
            "one_model_per_gpu": True,
            "skip_prefetch": bool(args.skip_prefetch),
            "hf_xet_high_performance": hf_xet_high_performance,
            "pytorch_alloc_conf": allocator_environment["PYTORCH_ALLOC_CONF"],
            "legacy_pytorch_cuda_alloc_conf_removed": allocator_environment[
                "legacy_pytorch_cuda_alloc_conf_removed"
            ],
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
        "status": "running",
    }
    write_json(matrix_root / "matrix_manifest.json", manifest)

    repository_started = utc_now()
    repository_clock = time.perf_counter()
    repository_error: Exception | None = None
    repository_status = ""
    try:
        repository_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository_status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest["repository_commit"] = repository_commit
    except Exception as exc:
        repository_error = exc
    repository_clean = repository_error is None and not repository_status
    repository_stage: dict[str, object] = {
        "stage": "repository_clean_check",
        "started_at_utc": repository_started,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - repository_clock,
        "exit_code": 0 if repository_clean else 1,
    }
    if repository_error is not None:
        repository_stage["error_type"] = type(repository_error).__name__
    elif repository_status:
        repository_stage["error_type"] = "DirtyRepository"
    manifest["common_stages"].append(repository_stage)
    if not repository_clean:
        manifest["status"] = "failed"
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return 1
    write_json(matrix_root / "matrix_manifest.json", manifest)

    base_command = [sys.executable, "-m", "hidden_policy_eval"]
    doctor = run_stage(
        "runtime_doctor",
        [
            *base_command,
            "doctor",
            "--backend",
            backend,
            "--config",
            str(frozen_config_path),
        ],
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
    gpu_check_error: Exception | None = None
    try:
        initial_gpu_state = gpu_snapshot(set(gpu_list))
    except Exception as exc:
        initial_gpu_state = []
        gpu_check_error = exc
    gpu_check_ok = gpu_check_error is None and (
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
    if gpu_check_error is not None:
        gpu_check.update(
            {
                "error_type": type(gpu_check_error).__name__,
                "error": str(gpu_check_error),
            }
        )
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
            str(frozen_config_path),
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

    try:
        preload_prompt_audit_dependencies()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_stage"] = "prompt_audit_dependency_load"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        write_json(matrix_root / "matrix_manifest.json", manifest)
        return 1

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
    owner_tokens: dict[str, str] = {}
    owner_seed = secrets.token_hex(16)
    samples: list[dict[str, float | int | str]] = []
    sample_path = matrix_root / "gpu_samples.jsonl"
    completions: dict[str, tuple[str, float]] = {}
    telemetry_errors: list[dict[str, str]] = []
    caught_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers: dict[signal.Signals, Any] = {}
    pending_signal: int | None = None

    def interrupt_handler(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        if pending_signal is None:
            pending_signal = signum

    def raise_if_interrupted() -> None:
        if pending_signal is not None:
            raise MatrixInterrupted(pending_signal)

    def restore_signal_handlers() -> None:
        for caught_signal, previous_handler in previous_handlers.items():
            signal.signal(caught_signal, previous_handler)

    def finish_interruption(signum: int) -> int:
        cleanup: dict[str, object] = {}
        for role, (process, console, *_rest) in processes.items():
            try:
                cleanup[role] = terminate_owned_processes(
                    process, owner_tokens[role]
                )
            except Exception as cleanup_exc:
                cleanup[role] = {
                    "stage": "owned_process_cleanup",
                    "status": "failed",
                    "error_type": type(cleanup_exc).__name__,
                }
            finally:
                console.close()
        manifest["status"] = "interrupted"
        manifest["error_type"] = "MatrixInterrupted"
        manifest["interrupt_signal"] = signum
        manifest["interruption_cleanup"] = cleanup
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        manifest["telemetry_errors"] = telemetry_errors
        write_json(matrix_root / "matrix_manifest.json", manifest)
        restore_signal_handlers()
        return 128 + signum

    for caught_signal in caught_signals:
        previous_handlers[caught_signal] = signal.signal(
            caught_signal, interrupt_handler
        )
    try:
        for role, gpu in zip(args.models, gpu_list):
            role_root = matrix_root / role
            lm_eval_root = role_root / "lm_eval"
            console_path = role_root / "evaluation.log"
            role_root.mkdir(parents=True, exist_ok=True)
            console = console_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            owner_token = f"{owner_seed}:{role}"
            environment[OWNER_ENV] = owner_token
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
                str(frozen_config_path),
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
            owner_tokens[role] = owner_token
            raise_if_interrupted()
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
                raise_if_interrupted()
                # Poll before sampling so a faster model is not repeatedly
                # represented by idle-GPU rows while another model finishes.
                for role, (process, *_rest) in processes.items():
                    if role not in completions and process.poll() is not None:
                        completions[role] = (utc_now(), time.perf_counter())
                active_roles = [
                    role for role in processes if role not in completions
                ]
                if not active_roles:
                    break
                active_gpus = {
                    str(manifest["models"][role]["gpu"]) for role in active_roles
                }
                try:
                    current_samples = gpu_snapshot(active_gpus)
                except Exception as exc:
                    telemetry_errors.append(
                        {
                            "captured_at_utc": utc_now(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    current_samples = []
                raise_if_interrupted()
                for sample in current_samples:
                    samples.append(sample)
                    sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                sample_file.flush()
                # Poll again after nvidia-smi so completion detection remains
                # prompt even if the snapshot command itself is slow.
                for role, (process, *_rest) in processes.items():
                    if role not in completions and process.poll() is not None:
                        completions[role] = (utc_now(), time.perf_counter())
                if len(completions) < len(processes):
                    time.sleep(args.gpu_poll_seconds)
                    raise_if_interrupted()
    except BaseException as exc:
        if isinstance(exc, MatrixInterrupted):
            return finish_interruption(exc.signum)
        if isinstance(exc, KeyboardInterrupt):
            return finish_interruption(int(signal.SIGINT))
        cleanup: dict[str, object] = {}
        for role, (process, console, *_rest) in processes.items():
            try:
                cleanup[role] = terminate_owned_processes(
                    process, owner_tokens[role]
                )
            except Exception as cleanup_exc:
                cleanup[role] = {
                    "stage": "owned_process_cleanup",
                    "status": "failed",
                    "error_type": type(cleanup_exc).__name__,
                }
            finally:
                console.close()
        was_signal = isinstance(exc, (KeyboardInterrupt, MatrixInterrupted))
        manifest["status"] = "interrupted" if was_signal else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["interruption_cleanup"] = cleanup
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - total_clock
        manifest["telemetry_errors"] = telemetry_errors
        write_json(matrix_root / "matrix_manifest.json", manifest)
        restore_signal_handlers()
        raise

    if pending_signal is not None:
        return finish_interruption(pending_signal)

    failed = False
    successful_roles: list[str] = []
    for role, (process, console, console_path, lm_eval_root, role_started, clock) in (
        processes.items()
    ):
        if pending_signal is not None:
            return finish_interruption(pending_signal)
        exit_code = process.wait()
        console.close()
        cleanup = terminate_owned_processes(
            process,
            owner_tokens[role],
            passive_grace_seconds=2.0,
            term_grace_seconds=5.0,
            kill_grace_seconds=2.0,
        )
        manifest["models"][role]["process_cleanup"] = cleanup
        if pending_signal is not None:
            return finish_interruption(pending_signal)
        role_ended, end_clock = completions[role]
        gpu = str(manifest["models"][role]["gpu"])
        gpu_rows = [
            row
            for row in samples
            if row["gpu"] == gpu
            and role_started <= str(row["captured_at_utc"]) <= role_ended
        ]
        telemetry = aggregate_gpu_samples(
            gpu_rows,
            process_duration_seconds=end_clock - clock,
            configured_poll_seconds=args.gpu_poll_seconds,
        )
        evaluation_stage = {
            "stage": "evaluation_process",
            "started_at_utc": role_started,
            "ended_at_utc": role_ended,
            "duration_seconds": end_clock - clock,
            "exit_code": exit_code,
            "console_sha256": sha256_file(console_path),
            **telemetry,
        }
        timing_error: Exception | None = None
        timing_path = lm_eval_root / "hidden_policy_timing.json"
        try:
            evaluation_stage["harness_timing"] = read_completed_harness_timing(
                timing_path,
                expected_backend=backend,
                expected_pytorch_alloc_conf=str(
                    config["evaluation"]["pytorch_alloc_conf"]
                ),
            )
            evaluation_stage["harness_timing_validation"] = "valid"
        except Exception as exc:
            timing_error = exc
            evaluation_stage["harness_timing_validation"] = "invalid"
            evaluation_stage["harness_timing_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        manifest["models"][role]["evaluation"] = evaluation_stage
        if cleanup["status"] not in {"already_clean", "best_effort"}:
            failed = True
            telemetry_errors.append(
                {
                    "captured_at_utc": utc_now(),
                    "error_type": "ProcessCleanupFailure",
                    "error": f"owned process cleanup for {role} was not clean",
                }
            )
        if telemetry["sample_count"] == 0:
            failed = True
            telemetry_errors.append(
                {
                    "captured_at_utc": utc_now(),
                    "error_type": "MissingGpuTelemetry",
                    "error": (
                        f"no valid GPU samples captured for {role} on physical GPU {gpu}"
                    ),
                }
            )
        if exit_code != 0 or timing_error is not None:
            failed = True
            write_json(
                matrix_root / role / "run_manifest.json", manifest["models"][role]
            )
            continue
        successful_roles.append(role)

    try:
        for role in successful_roles:
            raise_if_interrupted()
            gpu = str(manifest["models"][role]["gpu"])
            lm_eval_root = processes[role][3]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment[OWNER_ENV] = owner_tokens[role]
            postprocess = run_interruptible_stage(
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
                    str(frozen_config_path),
                    "--log-dir",
                    str(lm_eval_root),
                    "--output-dir",
                    str(matrix_root / role / "normalized"),
                ],
                log_path=matrix_root / role / "postprocess.log",
                environment=environment,
                owner_token=owner_tokens[role],
                check_interrupted=raise_if_interrupted,
            )
            manifest["models"][role]["postprocess"] = postprocess
            raise_if_interrupted()
            if postprocess["exit_code"] != 0:
                failed = True
            write_json(
                matrix_root / role / "run_manifest.json", manifest["models"][role]
            )
    except MatrixInterrupted as exc:
        return finish_interruption(exc.signum)
    except KeyboardInterrupt:
        return finish_interruption(int(signal.SIGINT))

    manifest["status"] = "failed" if failed else "completed"
    manifest["ended_at_utc"] = utc_now()
    manifest["duration_seconds"] = time.perf_counter() - total_clock
    manifest["telemetry_errors"] = telemetry_errors
    manifest["gpu_samples_sha256"] = sha256_file(sample_path)
    write_json(matrix_root / "matrix_manifest.json", manifest)
    if pending_signal is not None:
        return finish_interruption(pending_signal)
    restore_signal_handlers()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
