#!/usr/bin/env python3
"""Validate baseline matrix artifacts and publish a content-free HTML report.

The publisher deliberately reads only normalized score files and provenance
records.  Its output schema is an allowlist: prompts, choices, labels, raw model
responses, command lines, and local filesystem paths are never copied into the
publication artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "experiment0.json"
DEFAULT_SPLIT_METADATA = CODE_ROOT / "manifests" / "experiment0" / "metadata.json"
DEFAULT_OUTPUT_JSON = CODE_ROOT / "reports" / "baseline-results.json"
DEFAULT_OUTPUT_HTML = CODE_ROOT / "reports" / "baseline-results.html"

MODEL_ROLES = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")
DATASETS = ("wmdp", "mmlu")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
FORBIDDEN_NORMALIZED_KEYS = {
    "answer",
    "arguments",
    "choices",
    "continuation",
    "doc",
    "prompt",
    "question",
    "raw_response",
    "response",
    "resps",
    "target",
}


class PublicationError(ValueError):
    """Raised when an input cannot support a trustworthy publication."""


@dataclass(frozen=True)
class ModelArtifacts:
    role: str
    manifest: Mapping[str, object]
    summary: Mapping[str, object]
    option_rows: tuple[Mapping[str, object], ...]
    strict_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class MatrixArtifacts:
    root: Path
    manifest: Mapping[str, object]
    models: Mapping[str, ModelArtifacts]


def _fail(message: str) -> None:
    raise PublicationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(f"{label} must be finite and >= {minimum}")
    return result


def _rate(value: object, label: str) -> float:
    result = _number(value, label)
    if result > 1.0:
        _fail(f"{label} must be between 0 and 1")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _hash(value: object, label: str) -> str:
    result = _text(value, label)
    if HASH_PATTERN.fullmatch(result) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return result


def _git_oid(value: object, label: str) -> str:
    result = _text(value, label)
    if GIT_OID_PATTERN.fullmatch(result) is None:
        _fail(f"{label} must be a lowercase 40- or 64-character Git object ID")
    return result


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"missing {label}: {path.name}")
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {label}: {exc}")
    return _object(value, label)


def _read_jsonl(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        _fail(f"missing {label}: {path.name}")
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"invalid JSON in {label} line {line_number}: {exc}")
            row_object = _object(row, f"{label} line {line_number}")
            forbidden = FORBIDDEN_NORMALIZED_KEYS.intersection(row_object)
            if forbidden:
                _fail(
                    f"{label} line {line_number} contains unpublished content key(s): "
                    + ", ".join(sorted(forbidden))
                )
            rows.append(row_object)
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def _stage(
    value: object,
    label: str,
    *,
    expected_name: str | None = None,
) -> Mapping[str, object]:
    stage = _object(value, label)
    name = _text(stage.get("stage"), f"{label}.stage")
    if expected_name is not None and name != expected_name:
        _fail(f"{label}.stage must be {expected_name}, got {name}")
    _number(stage.get("duration_seconds"), f"{label}.duration_seconds")
    if stage.get("exit_code") != 0:
        _fail(f"{label} did not complete successfully")
    return stage


def _flag(command: list[object], name: str, label: str) -> str:
    try:
        index = command.index(name)
        value = command[index + 1]
    except (ValueError, IndexError):
        _fail(f"{label} is missing {name}")
    return _text(value, f"{label} {name}")


def _metric_close(observed: object, expected: float, label: str) -> None:
    actual = _rate(observed, label)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{label} is {actual}, recomputed value is {expected}")


def _id_set_hash(item_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(item_ids)).encode("utf-8")).hexdigest()


def _computed_rates(rows: list[Mapping[str, object]], label: str) -> dict[str, object]:
    by_item: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        item_id = _text(row.get("stable_id"), f"{label}.stable_id")
        permutation = _integer(
            row.get("permutation_id"), f"{label}.permutation_id"
        )
        if permutation not in {0, 1, 2}:
            _fail(f"{label} has permutation outside 0, 1, 2")
        prediction = _integer(
            row.get("predicted_semantic_index"),
            f"{label}.predicted_semantic_index",
        )
        if prediction > 3:
            _fail(f"{label} has semantic prediction outside 0..3")
        if not isinstance(row.get("correct"), bool):
            _fail(f"{label}.correct must be boolean")
        by_item.setdefault(item_id, []).append(row)
    for item_id, views in by_item.items():
        permutations = [int(row["permutation_id"]) for row in views]
        if len(views) != 3 or set(permutations) != {0, 1, 2}:
            _fail(f"{label} item {item_id} does not have exactly three unique views")
    canonical = [row for row in rows if row["permutation_id"] == 0]
    return {
        "items": len(by_item),
        "views": len(rows),
        "complete_three_view_items": len(by_item),
        "item_set_sha256": _id_set_hash(by_item),
        "canonical_accuracy": (
            sum(bool(row["correct"]) for row in canonical) / len(canonical)
            if canonical
            else 0.0
        ),
        "all_view_accuracy": (
            sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "semantic_permutation_consistency": (
            sum(
                len({int(view["predicted_semantic_index"]) for view in views}) == 1
                for views in by_item.values()
            )
            / len(by_item)
            if by_item
            else 0.0
        ),
    }


def _computed_strict_rates(
    rows: list[Mapping[str, object]], label: str
) -> dict[str, object]:
    seen: set[str] = set()
    for row in rows:
        item_id = _text(row.get("stable_id"), f"{label}.stable_id")
        if item_id in seen:
            _fail(f"{label} has duplicate item {item_id}")
        seen.add(item_id)
        status = _text(row.get("status"), f"{label}.status")
        if status not in {"valid", "invalid", "refusal"}:
            _fail(f"{label} has unsupported strict status {status}")
        if not isinstance(row.get("correct"), bool):
            _fail(f"{label}.correct must be boolean")
    count = len(rows)
    return {
        "items": count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count if count else 0.0,
        "invalid_rate": sum(row["status"] == "invalid" for row in rows) / count
        if count
        else 0.0,
        "refusal_rate": sum(row["status"] == "refusal" for row in rows) / count
        if count
        else 0.0,
        "invalid_or_refusal_rate": sum(row["status"] != "valid" for row in rows)
        / count
        if count
        else 0.0,
    }


def _validate_rate_block(
    observed: object, expected: Mapping[str, object], label: str
) -> None:
    block = _object(observed, label)
    for key in ("items", "views", "complete_three_view_items"):
        if block.get(key) != expected[key]:
            _fail(f"{label}.{key} does not match normalized score rows")
    if _hash(block.get("item_set_sha256"), f"{label}.item_set_sha256") != expected[
        "item_set_sha256"
    ]:
        _fail(f"{label}.item_set_sha256 does not match normalized score rows")
    for key in (
        "canonical_accuracy",
        "all_view_accuracy",
        "semantic_permutation_consistency",
    ):
        _metric_close(block.get(key), float(expected[key]), f"{label}.{key}")


def _validate_strict_block(
    observed: object, expected: Mapping[str, object], label: str
) -> None:
    block = _object(observed, label)
    if block.get("items") != expected["items"]:
        _fail(f"{label}.items does not match normalized score rows")
    for key in (
        "accuracy",
        "invalid_rate",
        "refusal_rate",
        "invalid_or_refusal_rate",
    ):
        _metric_close(block.get(key), float(expected[key]), f"{label}.{key}")


def _validate_summary_and_rows(
    summary: Mapping[str, object],
    option_rows: tuple[Mapping[str, object], ...],
    strict_rows: tuple[Mapping[str, object], ...],
    *,
    expected_counts: Mapping[str, int],
    expected_dataset_revisions: Mapping[str, str],
    label: str,
) -> None:
    if summary.get("schema_version") != "hidden-policy-experiment0-summary-v1":
        _fail(f"{label} has unsupported summary schema")
    datasets = _object(summary.get("datasets"), f"{label}.datasets")
    if set(datasets) != set(DATASETS):
        _fail(f"{label}.datasets must contain exactly WMDP and MMLU")
    for dataset in DATASETS:
        dataset_summary = _object(datasets[dataset], f"{label}.{dataset}")
        if dataset_summary.get("dataset_revision") != expected_dataset_revisions[dataset]:
            _fail(f"{label}.{dataset} uses an unexpected dataset revision")
        dataset_options = [row for row in option_rows if row.get("dataset") == dataset]
        dataset_strict = [row for row in strict_rows if row.get("dataset") == dataset]
        unknown_options = [row for row in option_rows if row.get("dataset") not in DATASETS]
        unknown_strict = [row for row in strict_rows if row.get("dataset") not in DATASETS]
        if unknown_options or unknown_strict:
            _fail(f"{label} normalized rows contain an unknown dataset")
        for row in [*dataset_options, *dataset_strict]:
            if row.get("dataset_revision") != expected_dataset_revisions[dataset]:
                _fail(f"{label}.{dataset} row uses an unexpected revision")
            _text(row.get("subject"), f"{label}.{dataset}.subject")
        computed_options = _computed_rates(
            dataset_options, f"{label}.{dataset}.option_scores"
        )
        computed_strict = _computed_strict_rates(
            dataset_strict, f"{label}.{dataset}.strict_scores"
        )
        expected_count = expected_counts[dataset]
        if computed_options["items"] != expected_count:
            _fail(
                f"{label}.{dataset} has {computed_options['items']} items; "
                f"expected {expected_count}"
            )
        if computed_strict["items"] != expected_count:
            _fail(f"{label}.{dataset} strict item count is not {expected_count}")
        option_ids = {str(row["stable_id"]) for row in dataset_options}
        strict_ids = {str(row["stable_id"]) for row in dataset_strict}
        if option_ids != strict_ids:
            _fail(f"{label}.{dataset} likelihood and strict item sets differ")
        _validate_rate_block(
            dataset_summary.get("option_likelihood"),
            computed_options,
            f"{label}.{dataset}.option_likelihood",
        )
        _validate_strict_block(
            dataset_summary.get("strict_generation"),
            computed_strict,
            f"{label}.{dataset}.strict_generation",
        )
        subjects = _object(dataset_summary.get("subjects"), f"{label}.{dataset}.subjects")
        observed_subjects = {
            str(row["subject"]) for row in [*dataset_options, *dataset_strict]
        }
        if set(subjects) != observed_subjects or not observed_subjects:
            _fail(f"{label}.{dataset} subject set does not match score rows")
        for subject in sorted(observed_subjects):
            subject_summary = _object(
                subjects[subject], f"{label}.{dataset}.subjects.{subject}"
            )
            subject_options = [
                row for row in dataset_options if str(row["subject"]) == subject
            ]
            subject_strict = [
                row for row in dataset_strict if str(row["subject"]) == subject
            ]
            _validate_rate_block(
                subject_summary.get("option_likelihood"),
                _computed_rates(
                    subject_options, f"{label}.{dataset}.{subject}.option_scores"
                ),
                f"{label}.{dataset}.subjects.{subject}.option_likelihood",
            )
            _validate_strict_block(
                subject_summary.get("strict_generation"),
                _computed_strict_rates(
                    subject_strict, f"{label}.{dataset}.{subject}.strict_scores"
                ),
                f"{label}.{dataset}.subjects.{subject}.strict_generation",
            )


def _validate_provenance(
    summary: Mapping[str, object],
    *,
    config: Mapping[str, object],
    config_sha256: str,
    manifest_checksums: Mapping[str, object],
    role: str,
    backend: str,
    scope: str,
    label: str,
) -> None:
    model_config = _object(
        _object(config.get("models"), "config.models").get(role),
        f"config.models.{role}",
    )
    evaluation = _object(config.get("evaluation"), "config.evaluation")
    provenance = _object(summary.get("provenance"), f"{label}.provenance")
    expected_scalar = {
        "model": model_config.get("repository"),
        "model_revision": model_config.get("revision"),
        "tokenizer": model_config.get("repository"),
        "tokenizer_revision": model_config.get("revision"),
        "backend": backend,
        "prompt_protocol": evaluation.get("prompt_protocol"),
        "enable_thinking": evaluation.get("enable_thinking"),
        "primary_score": evaluation.get("normalization"),
        "invocation_seed": evaluation.get("seed"),
    }
    for key, expected in expected_scalar.items():
        if provenance.get(key) != expected:
            _fail(f"{label}.provenance.{key} does not match the frozen config")
    _hash(provenance.get("runtime_fingerprint"), f"{label}.runtime_fingerprint")
    runtime = _object(
        provenance.get("runtime_provenance"), f"{label}.runtime_provenance"
    )
    if runtime.get("scope") != scope:
        _fail(f"{label} runtime scope must be {scope}")
    if runtime.get("config_sha256") != config_sha256:
        _fail(f"{label} runtime config hash does not match the frozen config")
    if runtime.get("manifest_checksums") != manifest_checksums:
        _fail(f"{label} runtime manifest checksums do not match tracked checksums")
    _hash(runtime.get("implementation_sha256"), f"{label}.implementation_sha256")
    _hash(runtime.get("task_bundle_sha256"), f"{label}.task_bundle_sha256")
    harness = _object(runtime.get("harness"), f"{label}.runtime.harness")
    expected_harness = {
        "repository": evaluation.get("harness_repository"),
        "version": evaluation.get("harness_version"),
        "commit": evaluation.get("harness_commit"),
        "tree": evaluation.get("harness_tree"),
    }
    if harness != expected_harness or provenance.get("harness") != expected_harness:
        _fail(f"{label} harness provenance does not match the frozen checkout")
    software = _object(
        provenance.get("software_environment"), f"{label}.software_environment"
    )
    expected_versions = {
        "datasets": evaluation.get("datasets_version"),
        "lm_eval": evaluation.get("harness_version"),
        "transformers": evaluation.get("transformers_version"),
        "torch": evaluation.get("torch_version"),
    }
    for package, expected in expected_versions.items():
        observed = _text(software.get(package), f"{label}.software.{package}")
        if observed.split("+", 1)[0] != str(expected):
            _fail(f"{label} has unexpected {package} version {observed}")
    torch_version = str(software["torch"])
    if f"+{evaluation.get('cuda_wheel')}" not in torch_version:
        _fail(f"{label} did not use the frozen CUDA PyTorch wheel")
    if backend == "vllm":
        observed_vllm = _text(software.get("vllm"), f"{label}.software.vllm")
        if observed_vllm.split("+", 1)[0] != str(evaluation.get("vllm_version")):
            _fail(f"{label} has unexpected vLLM version {observed_vllm}")
    if software.get("cuda_available") is not True:
        _fail(f"{label} was not produced with CUDA available")
    _integer(software.get("cuda_device_count"), f"{label}.cuda_device_count", minimum=1)


def _validate_model_artifacts(
    matrix_root: Path,
    role: str,
    model_manifest: Mapping[str, object],
    *,
    matrix_backend: str,
    matrix_scope: str,
    config: Mapping[str, object],
    config_sha256: str,
    manifest_checksums: Mapping[str, object],
    expected_counts: Mapping[str, int],
    expected_dataset_revisions: Mapping[str, str],
) -> ModelArtifacts:
    label = f"{matrix_scope}/{matrix_backend}/{role}"
    model_config = _object(
        _object(config.get("models"), "config.models").get(role),
        f"config.models.{role}",
    )
    if model_manifest.get("repository") != model_config.get("repository"):
        _fail(f"{label} repository does not match config")
    if model_manifest.get("revision") != model_config.get("revision"):
        _fail(f"{label} revision does not match config")
    command = _list(model_manifest.get("evaluation_command"), f"{label}.command")
    if _flag(command, "--model-role", f"{label}.command") != role:
        _fail(f"{label} command used another model role")
    if _flag(command, "--scope", f"{label}.command") != matrix_scope:
        _fail(f"{label} command used another scope")
    if _flag(command, "--backend", f"{label}.command") != matrix_backend:
        _fail(f"{label} command used another backend")
    if "--skip-prepare" not in command:
        _fail(f"{label} command did not reuse the audited shared runtime")

    prompt_audit = _stage(
        model_manifest.get("prompt_length_audit"),
        f"{label}.prompt_length_audit",
        expected_name="prompt_length_audit",
    )
    observed_max = _integer(
        prompt_audit.get("observed_max_request_tokens"),
        f"{label}.observed_max_request_tokens",
    )
    configured_max = _integer(
        prompt_audit.get("configured_max_model_len"),
        f"{label}.configured_max_model_len",
        minimum=1,
    )
    if configured_max != config["evaluation"]["max_model_len"] or observed_max > configured_max:
        _fail(f"{label} prompt-length audit is inconsistent")
    if "prefetch" in model_manifest:
        prefetch = _stage(
            model_manifest["prefetch"],
            f"{label}.prefetch",
            expected_name="model_prefetch",
        )
        if prefetch.get("repository") != model_config.get("repository") or prefetch.get(
            "revision"
        ) != model_config.get("revision"):
            _fail(f"{label} prefetch resolved another model")
        if prefetch.get("snapshot_revision") != model_config.get("revision"):
            _fail(f"{label} prefetch did not resolve the pinned revision")

    evaluation = _stage(
        model_manifest.get("evaluation"),
        f"{label}.evaluation",
        expected_name="evaluation_process",
    )
    for key in (
        "peak_memory_used_mib",
        "peak_memory_fraction",
        "peak_utilization_percent",
        "mean_utilization_percent",
        "peak_power_watts",
    ):
        _number(evaluation.get(key), f"{label}.evaluation.{key}")
    if float(evaluation["peak_utilization_percent"]) > 100:
        _fail(f"{label} peak GPU utilization exceeds 100%")
    if float(evaluation["mean_utilization_percent"]) > 100:
        _fail(f"{label} mean GPU utilization exceeds 100%")
    if float(evaluation["peak_memory_fraction"]) > 1:
        _fail(f"{label} peak GPU memory fraction exceeds 1")
    harness_timing = _object(
        evaluation.get("harness_timing"), f"{label}.harness_timing"
    )
    if harness_timing.get("schema_version") != "hidden-policy-run-timing-v1":
        _fail(f"{label} has unsupported harness timing schema")
    if harness_timing.get("status") != "completed" or harness_timing.get(
        "backend"
    ) != matrix_backend:
        _fail(f"{label} harness timing did not complete on {matrix_backend}")
    if str(harness_timing.get("cuda_visible_devices")) != str(model_manifest.get("gpu")):
        _fail(f"{label} GPU assignment differs from CUDA_VISIBLE_DEVICES")
    harness_stages = _list(harness_timing.get("stages"), f"{label}.harness_stages")
    by_stage: dict[str, Mapping[str, object]] = {}
    for index, raw_stage in enumerate(harness_stages):
        stage = _stage(raw_stage, f"{label}.harness_stages[{index}]")
        stage_name = str(stage["stage"])
        if stage_name in by_stage:
            _fail(f"{label} has duplicate harness stage {stage_name}")
        by_stage[stage_name] = stage
    if set(by_stage) != {"lm_eval_validate", "model_load_and_evaluation"}:
        _fail(f"{label} harness timing has unexpected stages")
    _stage(
        model_manifest.get("postprocess"),
        f"{label}.postprocess",
        expected_name="postprocess",
    )

    role_root = matrix_root / role
    stored_model_manifest = _read_json(
        role_root / "run_manifest.json", f"{label} run manifest"
    )
    if stored_model_manifest != model_manifest:
        _fail(f"{label} run manifest differs from matrix manifest")
    normalized = role_root / "normalized"
    summary = _read_json(normalized / "summary.json", f"{label} summary")
    option_rows = _read_jsonl(
        normalized / "option_scores.jsonl", f"{label} option scores"
    )
    strict_rows = _read_jsonl(
        normalized / "strict_scores.jsonl", f"{label} strict scores"
    )
    _validate_provenance(
        summary,
        config=config,
        config_sha256=config_sha256,
        manifest_checksums=manifest_checksums,
        role=role,
        backend=matrix_backend,
        scope=matrix_scope,
        label=label,
    )
    _validate_summary_and_rows(
        summary,
        option_rows,
        strict_rows,
        expected_counts=expected_counts,
        expected_dataset_revisions=expected_dataset_revisions,
        label=label,
    )
    return ModelArtifacts(role, model_manifest, summary, option_rows, strict_rows)


def load_matrix(
    root: Path,
    *,
    expected_backend: str,
    expected_scope: str,
    expected_roles: tuple[str, ...],
    config: Mapping[str, object],
    config_sha256: str,
    manifest_checksums: Mapping[str, object],
    expected_counts: Mapping[str, int],
    expected_dataset_revisions: Mapping[str, str],
) -> MatrixArtifacts:
    manifest = _read_json(root / "matrix_manifest.json", f"{expected_scope} matrix")
    if manifest.get("schema_version") != "hidden-policy-baseline-matrix-v1":
        _fail(f"{expected_scope} matrix has unsupported schema")
    if manifest.get("status") != "completed":
        _fail(f"{expected_scope} matrix is not completed")
    if manifest.get("scope") != expected_scope:
        _fail(f"matrix scope must be {expected_scope}")
    if manifest.get("backend") != expected_backend:
        _fail(f"{expected_scope} matrix backend must be {expected_backend}")
    if manifest.get("config_sha256") != config_sha256:
        _fail(f"{expected_scope} matrix config hash does not match current config")
    _text(manifest.get("run_id"), f"{expected_scope} matrix run_id")
    _git_oid(manifest.get("repository_commit"), f"{expected_scope} repository_commit")
    _number(manifest.get("duration_seconds"), f"{expected_scope} matrix duration")

    common = _list(manifest.get("common_stages"), f"{expected_scope}.common_stages")
    common_by_name: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(common):
        stage = _stage(value, f"{expected_scope}.common_stages[{index}]")
        name = str(stage["stage"])
        if name in common_by_name:
            _fail(f"{expected_scope} matrix has duplicate common stage {name}")
        common_by_name[name] = stage
    if set(common_by_name) != {
        "runtime_doctor",
        "gpu_availability",
        "prepare_runtime",
    }:
        _fail(f"{expected_scope} matrix has unexpected common stages")

    raw_models = _object(manifest.get("models"), f"{expected_scope}.models")
    if set(raw_models) != set(expected_roles):
        _fail(
            f"{expected_scope}/{expected_backend} matrix must contain exactly: "
            + ", ".join(expected_roles)
        )
    models: dict[str, ModelArtifacts] = {}
    for role in expected_roles:
        model_manifest = _object(raw_models[role], f"{expected_scope}.models.{role}")
        models[role] = _validate_model_artifacts(
            root,
            role,
            model_manifest,
            matrix_backend=expected_backend,
            matrix_scope=expected_scope,
            config=config,
            config_sha256=config_sha256,
            manifest_checksums=manifest_checksums,
            expected_counts=expected_counts,
            expected_dataset_revisions=expected_dataset_revisions,
        )

    reference = models[expected_roles[0]].summary
    reference_provenance = _object(reference["provenance"], "reference provenance")
    for role in expected_roles[1:]:
        provenance = _object(models[role].summary["provenance"], f"{role} provenance")
        for key in (
            "backend",
            "prompt_protocol",
            "enable_thinking",
            "primary_score",
            "runtime_fingerprint",
            "runtime_provenance",
            "harness",
        ):
            if provenance.get(key) != reference_provenance.get(key):
                _fail(f"{expected_scope} model summaries disagree in {key}")
        for dataset in DATASETS:
            left = reference["datasets"][dataset]
            right = models[role].summary["datasets"][dataset]
            if left["option_likelihood"]["item_set_sha256"] != right[
                "option_likelihood"
            ]["item_set_sha256"]:
                _fail(f"{expected_scope} models use different {dataset} item sets")
            if set(left["subjects"]) != set(right["subjects"]):
                _fail(f"{expected_scope} models use different {dataset} subjects")
    return MatrixArtifacts(root, manifest, models)


def _scientific_provenance(summary: Mapping[str, object]) -> Mapping[str, object]:
    provenance = _object(summary["provenance"], "summary provenance")
    runtime = _object(provenance["runtime_provenance"], "runtime provenance")
    return {
        "prompt_protocol": provenance["prompt_protocol"],
        "enable_thinking": provenance["enable_thinking"],
        "primary_score": provenance["primary_score"],
        "config_sha256": runtime["config_sha256"],
        "manifest_checksums": runtime["manifest_checksums"],
        "implementation_sha256": runtime["implementation_sha256"],
        "task_bundle_sha256": runtime["task_bundle_sha256"],
        "harness": runtime["harness"],
    }


def _validate_cross_matrix(
    pilot: MatrixArtifacts, full: MatrixArtifacts
) -> None:
    if pilot.manifest["repository_commit"] != full.manifest["repository_commit"]:
        _fail("pilot and full matrices were produced from different repository commits")
    for role in MODEL_ROLES:
        pilot_model = pilot.models[role]
        full_model = full.models[role]
        if _scientific_provenance(pilot_model.summary) != _scientific_provenance(
            full_model.summary
        ):
            _fail(f"pilot and full provenance differs for {role}")
        for dataset in DATASETS:
            pilot_ids = {
                str(row["stable_id"])
                for row in pilot_model.option_rows
                if row["dataset"] == dataset
            }
            full_ids = {
                str(row["stable_id"])
                for row in full_model.option_rows
                if row["dataset"] == dataset
            }
            if not pilot_ids.issubset(full_ids):
                _fail(f"{role} pilot {dataset} items are not a subset of full CAL")


def _agreement(
    vllm_model: ModelArtifacts, hf_model: ModelArtifacts
) -> dict[str, object]:
    def option_map(model: ModelArtifacts) -> dict[tuple[str, str, int], int]:
        return {
            (str(row["dataset"]), str(row["stable_id"]), int(row["permutation_id"])): int(
                row["predicted_semantic_index"]
            )
            for row in model.option_rows
        }

    def strict_map(model: ModelArtifacts) -> dict[tuple[str, str], tuple[object, object]]:
        return {
            (str(row["dataset"]), str(row["stable_id"])): (
                row["status"],
                row.get("predicted_display_index"),
            )
            for row in model.strict_rows
        }

    vllm_options, hf_options = option_map(vllm_model), option_map(hf_model)
    vllm_strict, hf_strict = strict_map(vllm_model), strict_map(hf_model)
    if set(vllm_options) != set(hf_options):
        _fail("HF reference and vLLM pilot option-score item/view sets differ")
    if set(vllm_strict) != set(hf_strict):
        _fail("HF reference and vLLM pilot strict item sets differ")

    def block(keys: list[tuple[str, str, int]]) -> dict[str, object]:
        matches = sum(vllm_options[key] == hf_options[key] for key in keys)
        return {
            "views": len(keys),
            "matching_predictions": matches,
            "prediction_agreement": matches / len(keys) if keys else 0.0,
        }

    by_dataset = {
        dataset: block([key for key in sorted(vllm_options) if key[0] == dataset])
        for dataset in DATASETS
    }
    all_keys = sorted(vllm_options)
    canonical_keys = [key for key in all_keys if key[2] == 0]
    strict_matches = sum(vllm_strict[key] == hf_strict[key] for key in vllm_strict)
    return {
        "available": True,
        "model_role": vllm_model.role,
        "all_views": block(all_keys),
        "canonical_views": block(canonical_keys),
        "by_dataset": by_dataset,
        "strict_generation": {
            "items": len(vllm_strict),
            "matching_parsed_predictions": strict_matches,
            "prediction_agreement": strict_matches / len(vllm_strict)
            if vllm_strict
            else 0.0,
        },
        "interpretation": (
            "agreement compares semantic option argmax for likelihood views; "
            "strict agreement compares parsed status and displayed option"
        ),
    }


def _public_metric_block(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "items": value["items"],
        "canonical_accuracy": value["canonical_accuracy"],
        "all_view_accuracy": value["all_view_accuracy"],
        "semantic_permutation_consistency": value[
            "semantic_permutation_consistency"
        ],
    }


def _public_strict_block(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "items": value["items"],
        "accuracy": value["accuracy"],
        "invalid_rate": value["invalid_rate"],
        "refusal_rate": value["refusal_rate"],
        "invalid_or_refusal_rate": value["invalid_or_refusal_rate"],
    }


def _public_datasets(summary: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for dataset in DATASETS:
        source = summary["datasets"][dataset]
        result[dataset] = {
            "dataset_revision": source["dataset_revision"],
            "option_likelihood": _public_metric_block(source["option_likelihood"]),
            "strict_generation": _public_strict_block(source["strict_generation"]),
            "subjects": {
                subject: {
                    "option_likelihood": _public_metric_block(
                        metrics["option_likelihood"]
                    ),
                    "strict_generation": _public_strict_block(
                        metrics["strict_generation"]
                    ),
                }
                for subject, metrics in sorted(source["subjects"].items())
            },
        }
    return result


def _public_timing(model: ModelArtifacts) -> dict[str, object]:
    source = model.manifest
    evaluation = source["evaluation"]
    harness = evaluation["harness_timing"]
    harness_stages = {
        stage["stage"]: stage["duration_seconds"] for stage in harness["stages"]
    }
    return {
        "prefetch_seconds": source.get("prefetch", {}).get("duration_seconds"),
        "prompt_length_audit_seconds": source["prompt_length_audit"][
            "duration_seconds"
        ],
        "evaluation_process_seconds": evaluation["duration_seconds"],
        "lm_eval_validate_seconds": harness_stages["lm_eval_validate"],
        "model_load_and_evaluation_seconds": harness_stages[
            "model_load_and_evaluation"
        ],
        "postprocess_seconds": source["postprocess"]["duration_seconds"],
    }


def _public_gpu(model: ModelArtifacts) -> dict[str, object]:
    evaluation = model.manifest["evaluation"]
    software = model.summary["provenance"]["software_environment"]
    devices = software.get("cuda_devices", [])
    return {
        "assigned_gpu_slot": str(model.manifest["gpu"]),
        "device_name": devices[0] if isinstance(devices, list) and devices else None,
        "peak_memory_used_mib": evaluation["peak_memory_used_mib"],
        "peak_memory_fraction": evaluation["peak_memory_fraction"],
        "peak_utilization_percent": evaluation["peak_utilization_percent"],
        "mean_utilization_percent": evaluation["mean_utilization_percent"],
        "peak_power_watts": evaluation["peak_power_watts"],
        "note": "polled whole-device peak, not process-isolated peak",
    }


def _public_matrix(matrix: MatrixArtifacts, config: Mapping[str, object]) -> dict[str, object]:
    common = {
        stage["stage"]: stage["duration_seconds"]
        for stage in matrix.manifest["common_stages"]
    }
    return {
        "run_id": matrix.manifest["run_id"],
        "scope": matrix.manifest["scope"],
        "backend": matrix.manifest["backend"],
        "status": "validated",
        "matrix_duration_seconds": matrix.manifest["duration_seconds"],
        "common_stage_seconds": common,
        "models": {
            role: {
                "display_name": config["models"][role]["display_name"],
                "repository": config["models"][role]["repository"],
                "revision": config["models"][role]["revision"],
                "parameters_billions": config["models"][role][
                    "parameters_billions"
                ],
                "timing": _public_timing(model),
                "gpu": _public_gpu(model),
                "datasets": _public_datasets(model.summary),
            }
            for role, model in matrix.models.items()
        },
    }


def _software_allowlist(summary: Mapping[str, object]) -> dict[str, object]:
    software = summary["provenance"]["software_environment"]
    keys = (
        "python",
        "datasets",
        "lm_eval",
        "transformers",
        "torch",
        "torch_cuda",
        "vllm",
        "cuda_device_count",
        "cuda_devices",
    )
    return {key: software.get(key) for key in keys}


def build_publication(
    pilot: MatrixArtifacts,
    full: MatrixArtifacts,
    *,
    config: Mapping[str, object],
    config_sha256: str,
    manifest_checksums: Mapping[str, object],
    hf_reference: MatrixArtifacts | None,
) -> dict[str, object]:
    _validate_cross_matrix(pilot, full)
    reference_model = pilot.models[MODEL_ROLES[0]]
    hf_agreement: dict[str, object]
    if hf_reference is None:
        hf_agreement = {
            "available": False,
            "reason": "未提供 Qwen3.5-2B Hugging Face pilot reference matrix。",
        }
    else:
        if hf_reference.manifest["repository_commit"] != pilot.manifest[
            "repository_commit"
        ]:
            _fail("HF reference and vLLM matrices use different repository commits")
        hf_model = hf_reference.models[MODEL_ROLES[0]]
        if _scientific_provenance(hf_model.summary) != _scientific_provenance(
            reference_model.summary
        ):
            _fail("HF reference and vLLM pilot scientific provenance differs")
        for dataset in DATASETS:
            vllm_dataset = reference_model.summary["datasets"][dataset]
            hf_dataset = hf_model.summary["datasets"][dataset]
            if vllm_dataset["option_likelihood"]["item_set_sha256"] != hf_dataset[
                "option_likelihood"
            ]["item_set_sha256"]:
                _fail(f"HF and vLLM references use different {dataset} items")
        hf_agreement = _agreement(reference_model, hf_model)

    provenance = reference_model.summary["provenance"]
    runtime = provenance["runtime_provenance"]
    evaluation = config["evaluation"]
    limitations = [
        "本报告只覆盖 CAL；TEST-Q3 与 TEST-Q4 保持 sealed，不能据此回答最终 Q3/Q4。",
        "三者均为原始 post-trained checkpoint；这是描述性 capability baseline，不是 sandbagging 训练结果。",
        "本矩阵按当前任务只比较 2B/4B/9B，未运行 0.8B weak reference；因此不执行 Plan 4 的 weak-headroom PASS/STOP gate。",
        "primary accuracy 来自完整选项文本的 continuation-token normalized likelihood；strict generation 只用于格式与拒答诊断。",
        "BF16 与不同推理 kernel 可能产生微小数值差异；HF reference 仅检查 2B pilot，不能证明所有尺寸逐 token 等价。",
        "subject 样本量不同，尤其 pilot 很小；subject 数值应视为诊断而非独立确认性结论。",
        "GPU 峰值来自定时轮询的整张物理卡，可能漏掉采样间峰值，也不等同于进程独占显存。",
        "没有置信区间、训练 seed 或因果干预，因此不能从模型尺寸差异推出 scaling law 或机制结论。",
    ]
    if hf_reference is None:
        limitations.append(
            "未提供 HF reference matrix；当前报告没有 HF-vLLM backend prediction agreement 证据。"
        )
    return {
        "schema_version": "hidden-policy-baseline-publication-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_language": "zh-CN",
        "validation_status": "PASS",
        "pilot": _public_matrix(pilot, config),
        "full_cal": _public_matrix(full, config),
        "hf_reference": (
            _public_matrix(hf_reference, config) if hf_reference is not None else None
        ),
        "hf_vllm_pilot_agreement": hf_agreement,
        "provenance": {
            "evaluated_repository_commit": pilot.manifest["repository_commit"],
            "config_sha256": config_sha256,
            "manifest_checksums": dict(manifest_checksums),
            "dataset_revisions": {
                dataset: config["datasets"][dataset]["revision"]
                for dataset in DATASETS
            },
            "harness": dict(runtime["harness"]),
            "implementation_sha256": runtime["implementation_sha256"],
            "task_bundle_sha256": runtime["task_bundle_sha256"],
            "runtime_fingerprints": {
                "pilot_vllm": provenance["runtime_fingerprint"],
                "full_vllm": full.models[MODEL_ROLES[0]].summary["provenance"][
                    "runtime_fingerprint"
                ],
                "pilot_hf_reference": (
                    hf_reference.models[MODEL_ROLES[0]].summary["provenance"][
                        "runtime_fingerprint"
                    ]
                    if hf_reference is not None
                    else None
                ),
            },
            "evaluation_protocol": {
                key: evaluation[key]
                for key in (
                    "backend",
                    "vllm_version",
                    "prompt_protocol",
                    "enable_thinking",
                    "candidate",
                    "normalization",
                    "permutation_count",
                    "dtype",
                    "batch_size",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "enable_prefix_caching",
                    "language_model_only",
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "seed",
                    "trust_remote_code",
                )
            },
            "software_environment": _software_allowlist(reference_model.summary),
        },
        "limitations": limitations,
        "privacy": {
            "content_free": True,
            "excluded": [
                "questions",
                "answer labels",
                "option texts",
                "raw responses",
                "command lines",
                "absolute local paths",
            ],
        },
    }


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def _seconds(value: object) -> str:
    if value is None:
        return "—"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


def _cell(value: object) -> str:
    return escape(str(value), quote=True)


def _aggregate_rows(matrix: Mapping[str, object]) -> str:
    rows: list[str] = []
    for role in MODEL_ROLES:
        model = matrix["models"][role]
        for dataset in DATASETS:
            metrics = model["datasets"][dataset]
            ll = metrics["option_likelihood"]
            strict = metrics["strict_generation"]
            rows.append(
                "<tr>"
                f"<td>{_cell(model['display_name'])}</td>"
                f"<td>{dataset.upper()}</td>"
                f"<td>{ll['items']}</td>"
                f"<td>{_pct(ll['canonical_accuracy'])}</td>"
                f"<td>{_pct(ll['all_view_accuracy'])}</td>"
                f"<td>{_pct(ll['semantic_permutation_consistency'])}</td>"
                f"<td>{_pct(strict['accuracy'])}</td>"
                f"<td>{_pct(strict['invalid_rate'])}</td>"
                f"<td>{_pct(strict['refusal_rate'])}</td>"
                "</tr>"
            )
    return "".join(rows)


def _subject_table(matrix: Mapping[str, object], dataset: str) -> str:
    models = matrix["models"]
    subjects = sorted(
        {
            subject
            for role in MODEL_ROLES
            for subject in models[role]["datasets"][dataset]["subjects"]
        }
    )
    headers = "".join(
        f"<th colspan='3'>{_cell(models[role]['display_name'])}</th>"
        for role in MODEL_ROLES
    )
    subheaders = "".join("<th>N</th><th>Acc</th><th>Perm.</th>" for _ in MODEL_ROLES)
    rows: list[str] = []
    for subject in subjects:
        cells = [f"<td>{_cell(subject)}</td>"]
        for role in MODEL_ROLES:
            metric = models[role]["datasets"][dataset]["subjects"].get(subject)
            if metric is None:
                cells.extend(("<td>—</td>", "<td>—</td>", "<td>—</td>"))
            else:
                ll = metric["option_likelihood"]
                cells.extend(
                    (
                        f"<td>{ll['items']}</td>",
                        f"<td>{_pct(ll['canonical_accuracy'])}</td>",
                        f"<td>{_pct(ll['semantic_permutation_consistency'])}</td>",
                    )
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr><th rowspan='2'>Subject</th>"
        + headers
        + "</tr><tr>"
        + subheaders
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _timing_rows(matrix: Mapping[str, object]) -> str:
    rows: list[str] = []
    for role in MODEL_ROLES:
        model = matrix["models"][role]
        timing, gpu = model["timing"], model["gpu"]
        rows.append(
            "<tr>"
            f"<td>{_cell(model['display_name'])}</td>"
            f"<td>{_seconds(timing['prefetch_seconds'])}</td>"
            f"<td>{_seconds(timing['prompt_length_audit_seconds'])}</td>"
            f"<td>{_seconds(timing['lm_eval_validate_seconds'])}</td>"
            f"<td>{_seconds(timing['model_load_and_evaluation_seconds'])}</td>"
            f"<td>{_seconds(timing['postprocess_seconds'])}</td>"
            f"<td>{float(gpu['peak_memory_used_mib']) / 1024:.2f} GiB</td>"
            f"<td>{float(gpu['peak_memory_fraction']) * 100:.1f}%</td>"
            f"<td>{float(gpu['mean_utilization_percent']):.1f}%</td>"
            f"<td>{float(gpu['peak_utilization_percent']):.0f}%</td>"
            f"<td>{float(gpu['peak_power_watts']):.1f} W</td>"
            "</tr>"
        )
    return "".join(rows)


def render_html(report: Mapping[str, object]) -> str:
    full = report["full_cal"]
    pilot = report["pilot"]
    agreement = report["hf_vllm_pilot_agreement"]
    if agreement["available"]:
        agreement_html = (
            "<div class='agreement-grid'>"
            f"<div><strong>{_pct(agreement['all_views']['prediction_agreement'])}</strong>"
            f"<span>全部 likelihood views ({agreement['all_views']['matching_predictions']}/"
            f"{agreement['all_views']['views']})</span></div>"
            f"<div><strong>{_pct(agreement['canonical_views']['prediction_agreement'])}</strong>"
            f"<span>canonical views</span></div>"
            f"<div><strong>{_pct(agreement['strict_generation']['prediction_agreement'])}</strong>"
            f"<span>strict parsed predictions</span></div>"
            "</div>"
        )
    else:
        agreement_html = f"<p class='notice'>{_cell(agreement['reason'])}</p>"

    provenance = report["provenance"]
    protocol = provenance["evaluation_protocol"]
    software = provenance["software_environment"]
    limitation_items = "".join(
        f"<li>{_cell(item)}</li>" for item in report["limitations"]
    )
    model_cards = "".join(
        (
            "<div class='model-card'>"
            f"<span>{_cell(full['models'][role]['display_name'])}</span>"
            f"<strong>WMDP {_pct(full['models'][role]['datasets']['wmdp']['option_likelihood']['canonical_accuracy'])}</strong>"
            f"<small>MMLU {_pct(full['models'][role]['datasets']['mmlu']['option_likelihood']['canonical_accuracy'])}</small>"
            "</div>"
        )
        for role in MODEL_ROLES
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hidden Policy · Qwen3.5 基础能力测试</title>
<style>
:root{{--ink:#17212b;--muted:#607080;--line:#dce3e8;--paper:#fff;--wash:#f4f7f9;--accent:#176b87;--accent2:#dd6b3d;--good:#217a52}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:36px 24px 64px}} header{{background:linear-gradient(130deg,#12384a,#176b87);color:#fff;border-radius:18px;padding:34px;box-shadow:0 12px 35px #163a4a22}}
h1{{font-size:30px;margin:0 0 8px}} header p{{margin:0;opacity:.86}} section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin-top:22px;padding:24px}}
h2{{font-size:21px;margin:0 0 14px}} h3{{font-size:17px;margin:22px 0 10px}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}}
.model-card{{background:#fff;color:var(--ink);border-radius:12px;padding:17px;display:flex;flex-direction:column}} .model-card span{{color:var(--muted)}} .model-card strong{{font-size:22px;color:var(--accent);margin-top:6px}} .model-card small{{font-size:14px}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;background:#daf2e6;color:var(--good);font-weight:700}} .meta{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .meta div{{background:var(--wash);padding:12px;border-radius:9px}} .meta span{{display:block;color:var(--muted);font-size:12px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th{{background:#edf3f6;color:#334b59}} th:first-child,td:first-child{{text-align:left}} tbody tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f8fafb}}
.agreement-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .agreement-grid div{{background:#edf7fa;border-left:4px solid var(--accent);padding:16px;border-radius:8px}} .agreement-grid strong{{display:block;color:var(--accent);font-size:24px}} .agreement-grid span{{color:var(--muted)}}
.notice{{background:#fff3dd;border-left:4px solid var(--accent2);padding:13px}} code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}} ul{{padding-left:22px}} footer{{color:var(--muted);text-align:center;margin-top:24px;font-size:12px}}
@media(max-width:760px){{.cards,.meta,.agreement-grid{{grid-template-columns:1fr}} main{{padding:18px 10px 40px}} section,header{{padding:18px}}}}
</style>
</head>
<body><main>
<header><span class="badge">输入验证 PASS</span><h1>Qwen3.5 基础能力测试</h1><p>WMDP / MMLU · non-thinking · full-option likelihood · CAL only</p><div class="cards">{model_cards}</div></header>
<section><h2>读者摘要</h2><p>本页比较 Qwen3.5-2B、4B、9B 原始 post-trained checkpoint。主结果是 full CAL 的完整选项文本 token-normalized likelihood；strict generation 与三种选项排列用于诊断格式失败和位置敏感性。此处没有训练 sandbagger，也没有解封 Q3/Q4 test。</p><div class="meta"><div><span>主后端</span>{_cell(full['backend'])}</div><div><span>Full CAL 总耗时</span>{_seconds(full['matrix_duration_seconds'])}</div><div><span>评测代码 commit</span><code>{_cell(provenance['evaluated_repository_commit'])}</code></div></div></section>
<section><h2>Full CAL 汇总</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>数据集</th><th>N</th><th>Canonical Acc</th><th>All-view Acc</th><th>Permutation consistency</th><th>Strict Acc</th><th>Invalid</th><th>Refusal</th></tr></thead><tbody>{_aggregate_rows(full)}</tbody></table></div></section>
<section><h2>Full CAL subject 诊断</h2><h3>WMDP</h3>{_subject_table(full,'wmdp')}<h3>MMLU</h3>{_subject_table(full,'mmlu')}<p class="notice">Subject 表中的 Acc 为 canonical-order likelihood accuracy；Perm. 为三种排列映射回 semantic option 后的一致率。</p></section>
<section><h2>32-item pilot 汇总</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>数据集</th><th>N</th><th>Canonical Acc</th><th>All-view Acc</th><th>Permutation consistency</th><th>Strict Acc</th><th>Invalid</th><th>Refusal</th></tr></thead><tbody>{_aggregate_rows(pilot)}</tbody></table></div></section>
<section><h2>HF ↔ vLLM pilot 一致性</h2>{agreement_html}</section>
<section><h2>Full CAL 耗时与 GPU 峰值</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>Prefetch</th><th>Prompt audit</th><th>lm-eval validate</th><th>Load + eval</th><th>Postprocess</th><th>峰值显存</th><th>显存占比</th><th>平均利用率</th><th>峰值利用率</th><th>峰值功耗</th></tr></thead><tbody>{_timing_rows(full)}</tbody></table></div><p class="notice">三模型并行执行，各自绑定一张物理 GPU；耗时会受共享 CPU、磁盘和 PCIe 影响。GPU 数值来自固定间隔的整卡 nvidia-smi 轮询，因此可能漏掉瞬时峰值，也不是进程级精确 profile。</p></section>
<section><h2>可复现性</h2><div class="meta"><div><span>lm-evaluation-harness</span><code>{_cell(provenance['harness']['version'])} · {_cell(provenance['harness']['commit'])}</code></div><div><span>vLLM / Transformers</span>{_cell(protocol['vllm_version'])} / {_cell(software['transformers'])}</div><div><span>PyTorch / CUDA</span>{_cell(software['torch'])} / {_cell(software['torch_cuda'])}</div><div><span>Prompt</span>{_cell(protocol['prompt_protocol'])}; thinking={_cell(protocol['enable_thinking'])}</div><div><span>Normalization</span>{_cell(protocol['normalization'])}</div><div><span>Runtime fingerprint (full)</span><code>{_cell(provenance['runtime_fingerprints']['full_vllm'])}</code></div></div></section>
<section><h2>限制与解释边界</h2><ul>{limitation_items}</ul></section>
<footer>本报告为自包含静态文件；不含题目、选项、答案标签、raw response、命令行或本机绝对路径。</footer>
</main></body></html>"""
    return html


def _assert_publication_safe(value: object, label: str = "report") -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_NORMALIZED_KEYS.intersection(value)
        if forbidden:
            _fail(f"{label} contains forbidden publication key(s): {sorted(forbidden)}")
        for key, child in value.items():
            _assert_publication_safe(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_publication_safe(child, f"{label}[{index}]")
    elif isinstance(value, str):
        if re.search(r"(?:^|[\s\"'])/(?:home|Users)/", value):
            _fail(f"{label} contains an absolute home path")
        if value.startswith("file://"):
            _fail(f"{label} contains a local file URI")


def generate_report(
    *,
    pilot_matrix: Path,
    full_matrix: Path,
    config_path: Path = DEFAULT_CONFIG,
    split_metadata_path: Path = DEFAULT_SPLIT_METADATA,
    hf_reference_matrix: Path | None = None,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_html: Path = DEFAULT_OUTPUT_HTML,
) -> dict[str, object]:
    config = _read_json(config_path, "experiment config")
    config_sha256 = _sha256_file(config_path)
    split_metadata = _read_json(split_metadata_path, "split metadata")
    if split_metadata.get("schema_version") != "hidden-policy-split-build-v1":
        _fail("split metadata has unsupported schema")
    manifest_checksums = _read_json(
        split_metadata_path.parent / "checksums.json", "manifest checksums"
    )
    dataset_config = _object(config.get("datasets"), "config.datasets")
    expected_revisions = {
        dataset: _text(
            _object(dataset_config.get(dataset), f"config.datasets.{dataset}").get(
                "revision"
            ),
            f"config.datasets.{dataset}.revision",
        )
        for dataset in DATASETS
    }
    pilot_config = _object(config.get("pilot"), "config.pilot")
    pilot_per_dataset = _object(
        pilot_config.get("per_dataset"), "config.pilot.per_dataset"
    )
    pilot_counts = {
        dataset: _integer(
            pilot_per_dataset.get(dataset), f"config.pilot.{dataset}", minimum=1
        )
        for dataset in DATASETS
    }
    if sum(pilot_counts.values()) != pilot_config.get("total_items"):
        _fail("pilot per-dataset counts do not sum to total_items")
    split_datasets = _object(split_metadata.get("datasets"), "split metadata.datasets")
    full_counts = {
        dataset: _integer(
            _object(split_datasets.get(dataset), f"split metadata.{dataset}").get(
                "cal_rows"
            ),
            f"split metadata.{dataset}.cal_rows",
            minimum=1,
        )
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        dataset_meta = _object(split_datasets[dataset], f"split metadata.{dataset}")
        split_counts = _object(
            dataset_meta.get("split_counts"), f"split metadata.{dataset}.split_counts"
        )
        if split_counts.get("CAL") != full_counts[dataset]:
            _fail(f"split metadata {dataset} CAL counts disagree")
        if dataset_meta.get("revision") != expected_revisions[dataset]:
            _fail(f"split metadata {dataset} revision differs from config")

    pilot = load_matrix(
        pilot_matrix,
        expected_backend="vllm",
        expected_scope="pilot",
        expected_roles=MODEL_ROLES,
        config=config,
        config_sha256=config_sha256,
        manifest_checksums=manifest_checksums,
        expected_counts=pilot_counts,
        expected_dataset_revisions=expected_revisions,
    )
    full = load_matrix(
        full_matrix,
        expected_backend="vllm",
        expected_scope="full",
        expected_roles=MODEL_ROLES,
        config=config,
        config_sha256=config_sha256,
        manifest_checksums=manifest_checksums,
        expected_counts=full_counts,
        expected_dataset_revisions=expected_revisions,
    )
    hf_reference = (
        load_matrix(
            hf_reference_matrix,
            expected_backend="hf",
            expected_scope="pilot",
            expected_roles=(MODEL_ROLES[0],),
            config=config,
            config_sha256=config_sha256,
            manifest_checksums=manifest_checksums,
            expected_counts=pilot_counts,
            expected_dataset_revisions=expected_revisions,
        )
        if hf_reference_matrix is not None
        else None
    )
    report = build_publication(
        pilot,
        full,
        config=config,
        config_sha256=config_sha256,
        manifest_checksums=manifest_checksums,
        hf_reference=hf_reference,
    )
    _assert_publication_safe(report)
    json_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    html = render_html(report)
    if re.search(r"(?:^|[\s\"'])/(?:home|Users)/", html) or "file://" in html:
        _fail("rendered HTML contains a local path")
    _atomic_write(output_json, json_bytes)
    _atomic_write(output_html, html.encode("utf-8"))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pilot/full matrices and publish a content-free report."
    )
    parser.add_argument("--pilot-matrix", type=Path, required=True)
    parser.add_argument("--full-matrix", type=Path, required=True)
    parser.add_argument("--hf-reference-matrix", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split-metadata", type=Path, default=DEFAULT_SPLIT_METADATA
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_report(
        pilot_matrix=args.pilot_matrix,
        full_matrix=args.full_matrix,
        hf_reference_matrix=args.hf_reference_matrix,
        config_path=args.config,
        split_metadata_path=args.split_metadata,
        output_json=args.output_json,
        output_html=args.output_html,
    )
    print(
        json.dumps(
            {
                "validation_status": report["validation_status"],
                "output_json": str(args.output_json),
                "output_html": str(args.output_html),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
