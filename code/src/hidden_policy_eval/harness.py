"""Thin, auditable command wrapper around lm-evaluation-harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Mapping

from .io import write_json
from .vendor import prepend_pythonpath


TASKS = (
    "plan4_wmdp_ll",
    "plan4_mmlu_ll",
    "plan4_wmdp_strict",
    "plan4_mmlu_strict",
)


@dataclass(frozen=True)
class HarnessRun:
    command: tuple[str, ...]
    environment: dict[str, str]
    output_dir: Path
    backend: str
    model: str
    revision: str
    prompt_protocol: str
    seed: int

    def preflight_command(self) -> tuple[str, ...]:
        tasks = self.command[self.command.index("--tasks") + 1]
        include_path = self.command[self.command.index("--include_path") + 1]
        return (
            self.command[0],
            "-m",
            "lm_eval",
            "validate",
            "--tasks",
            tasks,
            "--include_path",
            include_path,
        )

    def shell_preview(self) -> str:
        env = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in self.environment.items()
        )
        command = " ".join(shlex.quote(part) for part in self.command)
        return f"{env} {command}"


def build_harness_run(
    *,
    model: str,
    revision: str,
    data_dir: str | Path,
    output_dir: str | Path,
    tasks_dir: str | Path,
    harness_root: str | Path,
    backend: str,
    prompt_protocol: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    batch_size: str = "auto",
    pytorch_alloc_conf: str = "expandable_segments:True",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.87,
    max_num_seqs: int = 512,
    max_num_batched_tokens: int = 16384,
    enable_prefix_caching: bool = True,
    language_model_only: bool = True,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    seed: int = 1234,
    trust_remote_code: bool = False,
) -> HarnessRun:
    """Build the pinned command.  No subprocess is started here."""

    if prompt_protocol not in {"chat", "completion"}:
        raise ValueError("prompt protocol must be 'chat' or 'completion'")
    if backend not in {"hf", "vllm"}:
        raise ValueError("backend must be 'hf' or 'vllm'")
    model_arg_parts = [
        f"pretrained={model}",
        f"revision={revision}",
        f"dtype={dtype}",
        "enable_thinking=false",
        f"trust_remote_code={str(trust_remote_code).lower()}",
    ]
    if backend == "vllm":
        model_arg_parts.extend(
            (
                f"tokenizer={model}",
                f"tokenizer_revision={revision}",
                f"max_model_len={max_model_len}",
                f"gpu_memory_utilization={gpu_memory_utilization}",
                f"max_num_seqs={max_num_seqs}",
                f"max_num_batched_tokens={max_num_batched_tokens}",
                f"enable_prefix_caching={str(enable_prefix_caching).lower()}",
                f"language_model_only={str(language_model_only).lower()}",
                f"tensor_parallel_size={tensor_parallel_size}",
                f"data_parallel_size={data_parallel_size}",
                f"seed={seed}",
            )
        )
    else:
        model_arg_parts.append(f"max_length={max_model_len}")
    model_args = ",".join(model_arg_parts)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(TASKS),
        "--include_path",
        str(Path(tasks_dir).resolve()),
        "--batch_size",
        batch_size,
        "--log_samples",
        "--output_path",
        str(Path(output_dir).resolve()),
        "--seed",
        str(seed),
        "--confirm_run_unsafe_code",
    ]
    if backend == "hf":
        command.extend(("--device", device))
    if prompt_protocol == "chat":
        command.append("--apply_chat_template")
    return HarnessRun(
        command=tuple(command),
        environment={
            "HP_EVAL_DATA_DIR": str(Path(data_dir).resolve()),
            "PYTORCH_ALLOC_CONF": pytorch_alloc_conf,
            "PYTHONPATH": prepend_pythonpath(harness_root),
        },
        output_dir=Path(output_dir).resolve(),
        backend=backend,
        model=model,
        revision=revision,
        prompt_protocol=prompt_protocol,
        seed=seed,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_harness(run: HarnessRun) -> int:
    """Run preflight and evaluation while atomically recording stage timings."""

    if run.output_dir.exists() and any(run.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix runs in non-empty output directory: {run.output_dir}"
        )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run.output_dir / "hidden_policy_invocation.json",
        {
            "schema_version": "hidden-policy-invocation-v1",
            "model": run.model,
            "model_revision": run.revision,
            "tokenizer": run.model,
            "tokenizer_revision": run.revision,
            "backend": run.backend,
            "prompt_protocol": run.prompt_protocol,
            "enable_thinking": False,
            "seed": run.seed,
            "runtime_environment": {
                "PYTORCH_ALLOC_CONF": run.environment["PYTORCH_ALLOC_CONF"]
            },
            "command": list(run.command),
        },
    )
    timing_path = run.output_dir / "hidden_policy_timing.json"
    timing: dict[str, object] = {
        "schema_version": "hidden-policy-run-timing-v1",
        "backend": run.backend,
        "status": "running",
        "started_at_utc": _timestamp(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_environment": {
            "PYTORCH_ALLOC_CONF": run.environment["PYTORCH_ALLOC_CONF"]
        },
        "command": list(run.command),
        "stages": [],
    }
    write_json(timing_path, timing)
    environment = os.environ.copy()
    environment.update(run.environment)
    stage_started = _timestamp()
    stage_clock = time.perf_counter()
    preflight = subprocess.run(
        run.preflight_command(), env=environment, check=False
    )
    timing["stages"].append(
        {
            "stage": "lm_eval_validate",
            "started_at_utc": stage_started,
            "ended_at_utc": _timestamp(),
            "duration_seconds": time.perf_counter() - stage_clock,
            "exit_code": preflight.returncode,
        }
    )
    if preflight.returncode != 0:
        timing["status"] = "failed"
        timing["ended_at_utc"] = _timestamp()
        write_json(timing_path, timing)
        return preflight.returncode
    write_json(timing_path, timing)
    stage_started = _timestamp()
    stage_clock = time.perf_counter()
    completed = subprocess.run(run.command, env=environment, check=False)
    timing["stages"].append(
        {
            "stage": "model_load_and_evaluation",
            "started_at_utc": stage_started,
            "ended_at_utc": _timestamp(),
            "duration_seconds": time.perf_counter() - stage_clock,
            "exit_code": completed.returncode,
        }
    )
    timing["status"] = "completed" if completed.returncode == 0 else "failed"
    timing["ended_at_utc"] = _timestamp()
    write_json(timing_path, timing)
    return completed.returncode


def model_from_config(config: Mapping[str, object], role: str) -> tuple[str, str]:
    models = config.get("models")
    if not isinstance(models, dict) or role not in models:
        raise KeyError(f"unknown model role: {role}")
    model = models[role]
    if not isinstance(model, dict):
        raise TypeError(f"model config for {role} must be an object")
    return str(model["repository"]), str(model["revision"])
