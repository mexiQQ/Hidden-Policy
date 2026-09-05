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
from itertools import permutations
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "experiment0.json"
DEFAULT_SPLIT_METADATA = CODE_ROOT / "manifests" / "experiment0" / "metadata.json"
DEFAULT_OUTPUT_JSON = CODE_ROOT / "reports" / "baseline-results.json"
DEFAULT_OUTPUT_HTML = CODE_ROOT / "reports" / "baseline-results.html"

BASE_MODEL_ROLES = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")
WEAK_MODEL_ROLE = "weak"
MODEL_ROLES = (WEAK_MODEL_ROLE, *BASE_MODEL_ROLES)
HF_REFERENCE_ROLE = "qwen3_5_2b"
DATASETS = ("wmdp", "mmlu")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
STABLE_ID_PATTERN = re.compile(r"^mcq-[0-9a-f]{64}$")
GIT_OID_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
PERMUTATION_SALT = "hidden-policy-plan4-permutations-v1"
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
OPTION_SCORE_KEYS = {
    "schema_version",
    "dataset",
    "dataset_revision",
    "stable_id",
    "content_hash",
    "subject",
    "source_split",
    "split",
    "permutation_id",
    "display_to_semantic",
    "raw_log_likelihood_by_semantic",
    "continuation_tokens_by_semantic",
    "mean_log_likelihood_by_semantic",
    "predicted_semantic_index",
    "gold_semantic_index",
    "correct",
    "prompt_hash",
}
STRICT_SCORE_KEYS = {
    "schema_version",
    "dataset",
    "dataset_revision",
    "stable_id",
    "content_hash",
    "subject",
    "source_split",
    "split",
    "status",
    "predicted_display_index",
    "gold_display_index",
    "correct",
    "response_sha256",
    "prompt_hash",
}
ITEM_IDENTITY_FIELDS = (
    "dataset",
    "dataset_revision",
    "content_hash",
    "subject",
    "source_split",
    "split",
)
OPTION_INPUT_FIELDS = (
    *ITEM_IDENTITY_FIELDS,
    "prompt_hash",
    "gold_semantic_index",
    "display_to_semantic",
    "continuation_tokens_by_semantic",
)
STRICT_INPUT_FIELDS = (
    *ITEM_IDENTITY_FIELDS,
    "prompt_hash",
    "gold_display_index",
)


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


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
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


def _stable_id(value: object, label: str) -> str:
    result = _text(value, label)
    if STABLE_ID_PATTERN.fullmatch(result) is None:
        _fail(f"{label} must be mcq- followed by a lowercase SHA-256 digest")
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


def _exact_keys(row: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = expected.difference(row)
    unexpected = set(row).difference(expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected: " + ", ".join(sorted(unexpected)))
        _fail(f"{label} does not match its normalized schema ({'; '.join(details)})")


def _four_finite_numbers(value: object, label: str) -> list[float]:
    values = _list(value, label)
    if len(values) != 4:
        _fail(f"{label} must contain exactly four values")
    return [
        _finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]


def _four_positive_integers(value: object, label: str) -> list[int]:
    values = _list(value, label)
    if len(values) != 4:
        _fail(f"{label} must contain exactly four values")
    return [
        _integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(values)
    ]


def _expected_display_mappings(stable_id: str) -> tuple[tuple[int, ...], ...]:
    """Independently reproduce the frozen three-view permutation protocol."""

    identity = tuple(range(4))
    candidates = list(permutations(range(4)))
    candidates.remove(identity)
    selected = [identity]
    for view_index in range(1, 3):
        digest = hashlib.sha256(
            f"{PERMUTATION_SALT}\0{stable_id}\0{view_index}".encode("utf-8")
        ).digest()
        candidate_index = int.from_bytes(digest, "big") % len(candidates)
        selected.append(candidates.pop(candidate_index))
    return tuple(selected)


def _validate_common_score_row(row: Mapping[str, object], label: str) -> None:
    dataset = _text(row.get("dataset"), f"{label}.dataset")
    if dataset not in DATASETS:
        _fail(f"{label}.dataset must be WMDP or MMLU")
    _text(row.get("dataset_revision"), f"{label}.dataset_revision")
    _stable_id(row.get("stable_id"), f"{label}.stable_id")
    _hash(row.get("content_hash"), f"{label}.content_hash")
    _text(row.get("subject"), f"{label}.subject")
    _text(row.get("source_split"), f"{label}.source_split")
    if row.get("split") != "CAL":
        _fail(f"{label}.split must be CAL")
    _hash(row.get("prompt_hash"), f"{label}.prompt_hash")


def _validate_option_score_row(row: Mapping[str, object], label: str) -> None:
    _exact_keys(row, OPTION_SCORE_KEYS, label)
    if row.get("schema_version") != "hidden-policy-option-score-v1":
        _fail(f"{label} has unsupported option-score schema")
    _validate_common_score_row(row, label)
    permutation = _integer(row.get("permutation_id"), f"{label}.permutation_id")
    if permutation not in {0, 1, 2}:
        _fail(f"{label} has permutation outside 0, 1, 2")
    mapping = _list(row.get("display_to_semantic"), f"{label}.display_to_semantic")
    if len(mapping) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in mapping
    ) or sorted(mapping) != [0, 1, 2, 3]:
        _fail(f"{label}.display_to_semantic must be a permutation of 0..3")
    raw = _four_finite_numbers(
        row.get("raw_log_likelihood_by_semantic"),
        f"{label}.raw_log_likelihood_by_semantic",
    )
    tokens = _four_positive_integers(
        row.get("continuation_tokens_by_semantic"),
        f"{label}.continuation_tokens_by_semantic",
    )
    normalized = _four_finite_numbers(
        row.get("mean_log_likelihood_by_semantic"),
        f"{label}.mean_log_likelihood_by_semantic",
    )
    for index, (raw_score, token_count, normalized_score) in enumerate(
        zip(raw, tokens, normalized)
    ):
        expected = raw_score / token_count
        if not math.isclose(
            normalized_score, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            _fail(
                f"{label}.mean_log_likelihood_by_semantic[{index}] does not "
                "equal raw log likelihood divided by continuation tokens"
            )
    prediction = _integer(
        row.get("predicted_semantic_index"),
        f"{label}.predicted_semantic_index",
    )
    gold = _integer(row.get("gold_semantic_index"), f"{label}.gold_semantic_index")
    if prediction > 3 or gold > 3:
        _fail(f"{label} semantic prediction and gold must be in 0..3")
    expected_prediction = max(range(4), key=lambda index: normalized[index])
    if prediction != expected_prediction:
        _fail(f"{label}.predicted_semantic_index is not the normalized-score argmax")
    if not isinstance(row.get("correct"), bool):
        _fail(f"{label}.correct must be boolean")
    if row["correct"] != (prediction == gold):
        _fail(f"{label}.correct disagrees with prediction and gold")


def _validate_strict_score_row(row: Mapping[str, object], label: str) -> None:
    _exact_keys(row, STRICT_SCORE_KEYS, label)
    if row.get("schema_version") != "hidden-policy-strict-score-v1":
        _fail(f"{label} has unsupported strict-score schema")
    _validate_common_score_row(row, label)
    _hash(row.get("response_sha256"), f"{label}.response_sha256")
    status = _text(row.get("status"), f"{label}.status")
    if status not in {"valid", "invalid", "refusal"}:
        _fail(f"{label} has unsupported strict status {status}")
    prediction_value = row.get("predicted_display_index")
    if status == "valid":
        prediction = _integer(
            prediction_value, f"{label}.predicted_display_index"
        )
        if prediction > 3:
            _fail(f"{label}.predicted_display_index must be in 0..3")
    else:
        if prediction_value is not None:
            _fail(
                f"{label}.predicted_display_index must be null when status is {status}"
            )
        prediction = None
    gold = _integer(row.get("gold_display_index"), f"{label}.gold_display_index")
    if gold > 3:
        _fail(f"{label}.gold_display_index must be in 0..3")
    if not isinstance(row.get("correct"), bool):
        _fail(f"{label}.correct must be boolean")
    expected_correct = status == "valid" and prediction == gold
    if row["correct"] != expected_correct:
        _fail(f"{label}.correct disagrees with strict status, prediction, and gold")


def _computed_rates(rows: list[Mapping[str, object]], label: str) -> dict[str, object]:
    by_item: dict[str, list[Mapping[str, object]]] = {}
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        _validate_option_score_row(row, row_label)
        item_id = str(row["stable_id"])
        by_item.setdefault(item_id, []).append(row)
    for item_id, views in by_item.items():
        permutations = [int(row["permutation_id"]) for row in views]
        if len(views) != 3 or set(permutations) != {0, 1, 2}:
            _fail(f"{label} item {item_id} does not have exactly three unique views")
        reference = views[0]
        for field in (*ITEM_IDENTITY_FIELDS, "gold_semantic_index"):
            if any(view[field] != reference[field] for view in views[1:]):
                _fail(
                    f"{label} item {item_id} changes {field} across permutations"
                )
        by_permutation = {int(view["permutation_id"]): view for view in views}
        for permutation_id, expected_mapping in enumerate(
            _expected_display_mappings(item_id)
        ):
            observed_mapping = tuple(
                int(value)
                for value in by_permutation[permutation_id]["display_to_semantic"]
            )
            if observed_mapping != expected_mapping:
                _fail(
                    f"{label} item {item_id} permutation {permutation_id} has an "
                    "unexpected display_to_semantic mapping"
                )
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
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        _validate_strict_score_row(row, row_label)
        item_id = str(row["stable_id"])
        if item_id in seen:
            _fail(f"{label} has duplicate item {item_id}")
        seen.add(item_id)
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
        canonical_options = {
            str(row["stable_id"]): row
            for row in dataset_options
            if int(row["permutation_id"]) == 0
        }
        strict_by_item = {str(row["stable_id"]): row for row in dataset_strict}
        for item_id in sorted(option_ids):
            canonical = canonical_options[item_id]
            strict = strict_by_item[item_id]
            for field in ITEM_IDENTITY_FIELDS:
                if strict[field] != canonical[field]:
                    _fail(
                        f"{label}.{dataset} item {item_id} strict row differs "
                        f"from canonical option row in {field}"
                    )
            mapping = [int(value) for value in canonical["display_to_semantic"]]
            strict_gold = int(strict["gold_display_index"])
            semantic_gold = int(canonical["gold_semantic_index"])
            if mapping[strict_gold] != semantic_gold:
                _fail(
                    f"{label}.{dataset} item {item_id} strict gold is inconsistent "
                    "with the canonical display-to-semantic mapping"
                )
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


def _option_input_map(
    model: ModelArtifacts,
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    rows: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in model.option_rows:
        key = (
            str(row["dataset"]),
            str(row["stable_id"]),
            int(row["permutation_id"]),
        )
        if key in rows:
            _fail(f"{model.role} has duplicate option input key {key}")
        rows[key] = row
    return rows


def _strict_input_map(
    model: ModelArtifacts,
) -> dict[tuple[str, str], Mapping[str, object]]:
    rows: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in model.strict_rows:
        key = (str(row["dataset"]), str(row["stable_id"]))
        if key in rows:
            _fail(f"{model.role} has duplicate strict input key {key}")
        rows[key] = row
    return rows


def _validate_matching_model_inputs(
    left: ModelArtifacts,
    right: ModelArtifacts,
    *,
    label: str,
    option_keys: Iterable[tuple[str, str, int]] | None = None,
    strict_keys: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Require score comparisons to refer to identical prompts and labels."""

    left_options = _option_input_map(left)
    right_options = _option_input_map(right)
    if option_keys is None:
        selected_options = set(left_options)
        if selected_options != set(right_options):
            _fail(f"{label} option-score input key sets differ")
    else:
        selected_options = set(option_keys)
        if not selected_options.issubset(left_options) or not selected_options.issubset(
            right_options
        ):
            _fail(f"{label} is missing one or more selected option-score inputs")
    for key in sorted(selected_options):
        for field in OPTION_INPUT_FIELDS:
            if left_options[key][field] != right_options[key][field]:
                _fail(f"{label} option input differs in {field} for {key}")

    left_strict = _strict_input_map(left)
    right_strict = _strict_input_map(right)
    if strict_keys is None:
        selected_strict = set(left_strict)
        if selected_strict != set(right_strict):
            _fail(f"{label} strict input key sets differ")
    else:
        selected_strict = set(strict_keys)
        if not selected_strict.issubset(left_strict) or not selected_strict.issubset(
            right_strict
        ):
            _fail(f"{label} is missing one or more selected strict inputs")
    for key in sorted(selected_strict):
        for field in STRICT_INPUT_FIELDS:
            if left_strict[key][field] != right_strict[key][field]:
                _fail(f"{label} strict input differs in {field} for {key}")


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
    expected_allocator = evaluation.get("pytorch_alloc_conf")
    if software.get("pytorch_alloc_conf") != expected_allocator:
        _fail(f"{label} did not export the frozen PyTorch allocator configuration")
    if software.get("pytorch_cuda_alloc_conf_legacy") is not None:
        _fail(f"{label} retained the legacy PyTorch allocator alias")
    if software.get("pytorch_alloc_conf_at_snapshot") != expected_allocator:
        _fail(f"{label} postprocess snapshot did not parse the frozen allocator setting")
    if software.get("pytorch_allocator_backend") != evaluation.get(
        "pytorch_allocator_backend"
    ):
        _fail(f"{label} used an unexpected PyTorch allocator backend")
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
    _integer(
        evaluation.get("sample_count"),
        f"{label}.evaluation.sample_count",
        minimum=1,
    )
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
    if harness_timing.get("runtime_environment") != {
        "PYTORCH_ALLOC_CONF": config["evaluation"]["pytorch_alloc_conf"]
    }:
        _fail(f"{label} harness timing has an unexpected allocator setting")
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
    cleanup = _object(model_manifest.get("process_cleanup"), f"{label}.process_cleanup")
    if cleanup.get("stage") != "owned_process_cleanup":
        _fail(f"{label} has an unexpected cleanup stage")
    if cleanup.get("status") != "already_clean":
        _fail(f"{label} did not leave a clean owned process tree")
    if _integer(
        cleanup.get("remaining_process_count"),
        f"{label}.process_cleanup.remaining_process_count",
    ) != 0:
        _fail(f"{label} left one or more owned processes running")
    _number(
        cleanup.get("duration_seconds"), f"{label}.process_cleanup.duration_seconds"
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
    frozen_config = root / "frozen_config.json"
    if _sha256_file(frozen_config) != config_sha256:
        _fail(f"{expected_scope} matrix frozen config differs from current config")
    _text(manifest.get("run_id"), f"{expected_scope} matrix run_id")
    _git_oid(manifest.get("repository_commit"), f"{expected_scope} repository_commit")
    _number(manifest.get("duration_seconds"), f"{expected_scope} matrix duration")

    execution = _object(manifest.get("execution"), f"{expected_scope}.execution")
    if execution.get("models") != list(expected_roles):
        _fail(f"{expected_scope} execution model order differs from the selected matrix")
    physical_gpus = _list(
        execution.get("physical_gpus"), f"{expected_scope}.execution.physical_gpus"
    )
    if len(physical_gpus) != len(expected_roles) or len(set(physical_gpus)) != len(
        physical_gpus
    ):
        _fail(f"{expected_scope} execution did not assign one distinct GPU per model")
    if execution.get("one_model_per_gpu") is not True:
        _fail(f"{expected_scope} execution was not one model per GPU")
    if execution.get("hf_xet_high_performance") != config["evaluation"].get(
        "hf_xet_high_performance"
    ):
        _fail(f"{expected_scope} execution changed the frozen Xet setting")
    if execution.get("pytorch_alloc_conf") != config["evaluation"].get(
        "pytorch_alloc_conf"
    ):
        _fail(f"{expected_scope} execution changed the frozen allocator setting")
    if not isinstance(execution.get("legacy_pytorch_cuda_alloc_conf_removed"), bool):
        _fail(f"{expected_scope} execution did not record legacy allocator handling")
    _number(
        execution.get("gpu_poll_seconds"),
        f"{expected_scope}.execution.gpu_poll_seconds",
    )
    memory = _object(
        execution.get("vllm_memory_and_batching"),
        f"{expected_scope}.execution.vllm_memory_and_batching",
    )
    expected_memory = {
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
    }
    if memory != expected_memory:
        _fail(f"{expected_scope} execution changed frozen vLLM memory settings")

    common = _list(manifest.get("common_stages"), f"{expected_scope}.common_stages")
    common_by_name: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(common):
        stage = _stage(value, f"{expected_scope}.common_stages[{index}]")
        name = str(stage["stage"])
        if name in common_by_name:
            _fail(f"{expected_scope} matrix has duplicate common stage {name}")
        common_by_name[name] = stage
    if set(common_by_name) != {
        "repository_clean_check",
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

    reference_model = models[expected_roles[0]]
    reference = reference_model.summary
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
        _validate_matching_model_inputs(
            reference_model,
            models[role],
            label=(
                f"{expected_scope}/{expected_backend} models "
                f"{expected_roles[0]} and {role}"
            ),
        )
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
    if set(pilot.models) != set(full.models):
        _fail("pilot and full matrices contain different model roles")
    for role in pilot.models:
        pilot_model = pilot.models[role]
        full_model = full.models[role]
        if _scientific_provenance(pilot_model.summary) != _scientific_provenance(
            full_model.summary
        ):
            _fail(f"pilot and full provenance differs for {role}")
        pilot_option_keys = set(_option_input_map(pilot_model))
        full_option_keys = set(_option_input_map(full_model))
        pilot_strict_keys = set(_strict_input_map(pilot_model))
        full_strict_keys = set(_strict_input_map(full_model))
        if not pilot_option_keys.issubset(full_option_keys):
            _fail(f"{role} pilot option inputs are not a subset of full CAL")
        if not pilot_strict_keys.issubset(full_strict_keys):
            _fail(f"{role} pilot strict inputs are not a subset of full CAL")
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
        _validate_matching_model_inputs(
            pilot_model,
            full_model,
            label=f"pilot and full {role}",
            option_keys=pilot_option_keys,
            strict_keys=pilot_strict_keys,
        )


def _validate_supplement(
    primary: MatrixArtifacts,
    supplement: MatrixArtifacts,
    *,
    label: str,
) -> None:
    if primary.manifest["repository_commit"] != supplement.manifest[
        "repository_commit"
    ]:
        _fail(f"{label} matrices were produced from different repository commits")
    if set(primary.models).intersection(supplement.models):
        _fail(f"{label} supplement duplicates a primary model role")
    reference = primary.models[HF_REFERENCE_ROLE]
    added = supplement.models[WEAK_MODEL_ROLE]
    if _scientific_provenance(reference.summary) != _scientific_provenance(
        added.summary
    ):
        _fail(f"{label} weak and primary scientific provenance differs")
    _validate_matching_model_inputs(
        reference,
        added,
        label=f"{label} primary and weak model",
    )


def _agreement(
    vllm_model: ModelArtifacts, hf_model: ModelArtifacts
) -> dict[str, object]:
    def strict_map(model: ModelArtifacts) -> dict[tuple[str, str], tuple[object, object]]:
        return {
            (str(row["dataset"]), str(row["stable_id"])): (
                row["status"],
                row.get("predicted_display_index"),
            )
            for row in model.strict_rows
        }

    _validate_matching_model_inputs(
        vllm_model,
        hf_model,
        label="HF reference and vLLM pilot",
    )
    vllm_options = _option_input_map(vllm_model)
    hf_options = _option_input_map(hf_model)
    vllm_strict, hf_strict = strict_map(vllm_model), strict_map(hf_model)
    if set(vllm_options) != set(hf_options):
        _fail("HF reference and vLLM pilot option-score item/view sets differ")
    if set(vllm_strict) != set(hf_strict):
        _fail("HF reference and vLLM pilot strict item sets differ")

    def normalized_scores(row: Mapping[str, object]) -> list[float]:
        # Rows have already passed the closed schema validation above.
        return [float(value) for value in row["mean_log_likelihood_by_semantic"]]

    def centered(scores: list[float]) -> list[float]:
        center = sum(scores) / len(scores)
        return [score - center for score in scores]

    def top_margin(scores: list[float]) -> float:
        ordered = sorted(scores, reverse=True)
        return ordered[0] - ordered[1]

    def block(keys: list[tuple[str, str, int]]) -> dict[str, object]:
        matches = sum(
            vllm_options[key]["predicted_semantic_index"]
            == hf_options[key]["predicted_semantic_index"]
            for key in keys
        )
        centered_differences: list[float] = []
        margin_differences: list[float] = []
        vllm_margins: list[float] = []
        hf_margins: list[float] = []
        for key in keys:
            vllm_scores = normalized_scores(vllm_options[key])
            hf_scores = normalized_scores(hf_options[key])
            centered_differences.extend(
                left - right
                for left, right in zip(
                    centered(vllm_scores), centered(hf_scores)
                )
            )
            vllm_margin = top_margin(vllm_scores)
            hf_margin = top_margin(hf_scores)
            vllm_margins.append(vllm_margin)
            hf_margins.append(hf_margin)
            margin_differences.append(vllm_margin - hf_margin)
        absolute_centered = [abs(value) for value in centered_differences]
        absolute_margins = [abs(value) for value in margin_differences]
        vllm_accuracy = (
            sum(bool(vllm_options[key]["correct"]) for key in keys) / len(keys)
            if keys
            else 0.0
        )
        hf_accuracy = (
            sum(bool(hf_options[key]["correct"]) for key in keys) / len(keys)
            if keys
            else 0.0
        )
        return {
            "views": len(keys),
            "matching_predictions": matches,
            "prediction_agreement": matches / len(keys) if keys else 0.0,
            "accuracy": {
                "vllm": vllm_accuracy,
                "hf": hf_accuracy,
                "delta_vllm_minus_hf": vllm_accuracy - hf_accuracy,
            },
            "centered_per_option_normalized_ll_difference": {
                "values": len(centered_differences),
                "mean_absolute": (
                    sum(absolute_centered) / len(absolute_centered)
                    if absolute_centered
                    else 0.0
                ),
                "root_mean_square": (
                    math.sqrt(
                        sum(value * value for value in centered_differences)
                        / len(centered_differences)
                    )
                    if centered_differences
                    else 0.0
                ),
                "maximum_absolute": max(absolute_centered, default=0.0),
            },
            "top_margin_difference": {
                "vllm_mean": (
                    sum(vllm_margins) / len(vllm_margins)
                    if vllm_margins
                    else 0.0
                ),
                "hf_mean": (
                    sum(hf_margins) / len(hf_margins) if hf_margins else 0.0
                ),
                "mean_signed_vllm_minus_hf": (
                    sum(margin_differences) / len(margin_differences)
                    if margin_differences
                    else 0.0
                ),
                "mean_absolute": (
                    sum(absolute_margins) / len(absolute_margins)
                    if absolute_margins
                    else 0.0
                ),
                "maximum_absolute": max(absolute_margins, default=0.0),
            },
        }

    by_dataset = {
        dataset: block([key for key in sorted(vllm_options) if key[0] == dataset])
        for dataset in DATASETS
    }
    all_keys = sorted(vllm_options)
    canonical_keys = [key for key in all_keys if key[2] == 0]
    strict_matches = sum(vllm_strict[key] == hf_strict[key] for key in vllm_strict)
    return {
        "status": "descriptive",
        "available": True,
        "gate_applied": False,
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
            "Descriptive comparison only; no pass/fail threshold is applied. "
            "Each view's four normalized log likelihoods are mean-centered before "
            "backend subtraction. Top margin is the best score minus the runner-up; "
            "signed deltas are vLLM minus HF. Strict agreement compares parsed status "
            "and displayed option."
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
        "process_cleanup_seconds": source["process_cleanup"]["duration_seconds"],
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
        "sample_count": evaluation["sample_count"],
        "note": "polled whole-device peak, not process-isolated peak",
    }


def _model_metadata(config: Mapping[str, object], role: str) -> dict[str, object]:
    model = config["models"][role]
    repository = str(model["repository"])
    return {
        "display_name": model.get("display_name", repository.rsplit("/", 1)[-1]),
        "repository": repository,
        "revision": model["revision"],
        "parameters_billions": model.get("parameters_billions"),
    }


def _public_matrix(
    matrix: MatrixArtifacts,
    config: Mapping[str, object],
    *,
    public_run_id: str,
) -> dict[str, object]:
    common = {
        stage["stage"]: stage["duration_seconds"]
        for stage in matrix.manifest["common_stages"]
    }
    return {
        # The source run id is an operator-controlled directory name.  Keep it
        # in the validation boundary, but publish only a fixed semantic id so
        # arbitrary local text cannot become an output side channel.
        "run_id": public_run_id,
        "scope": matrix.manifest["scope"],
        "backend": matrix.manifest["backend"],
        "status": "validated",
        "matrix_duration_seconds": matrix.manifest["duration_seconds"],
        "common_stage_seconds": common,
        "models": {
            role: {
                **_model_metadata(config, role),
                "timing": _public_timing(model),
                "gpu": _public_gpu(model),
                "datasets": _public_datasets(model.summary),
            }
            for role, model in matrix.models.items()
        },
    }


def _public_matrix_bundle(
    matrices: tuple[MatrixArtifacts, ...],
    config: Mapping[str, object],
    *,
    public_run_id: str,
) -> dict[str, object]:
    published = [
        _public_matrix(matrix, config, public_run_id=public_run_id)
        for matrix in matrices
    ]
    result = dict(published[0])
    result["matrix_duration_seconds"] = sum(
        float(matrix["matrix_duration_seconds"]) for matrix in published
    )
    result["source_matrix_count"] = len(published)
    common_names = {
        name for matrix in published for name in matrix["common_stage_seconds"]
    }
    result["common_stage_seconds"] = {
        name: sum(
            float(matrix["common_stage_seconds"].get(name, 0.0))
            for matrix in published
        )
        for name in sorted(common_names)
    }
    result["models"] = {
        role: matrix["models"][role]
        for role in MODEL_ROLES
        for matrix in published
        if role in matrix["models"]
    }
    return result


def _software_allowlist(summary: Mapping[str, object]) -> dict[str, object]:
    software = summary["provenance"]["software_environment"]
    keys = (
        "python",
        "datasets",
        "lm_eval",
        "transformers",
        "torch",
        "torch_cuda",
        "pytorch_alloc_conf",
        "pytorch_cuda_alloc_conf_legacy",
        "pytorch_alloc_conf_at_snapshot",
        "pytorch_allocator_backend",
        "vllm",
        "cuda_device_count",
        "cuda_devices",
    )
    return {key: software.get(key) for key in keys}


LEDGER_STAGE_KEYS = (
    ("prefetch", "model_prefetch"),
    ("prompt_length_audit", "prompt_length_audit"),
    ("evaluation", "evaluation_process"),
    ("postprocess", "postprocess"),
    ("process_cleanup", "owned_process_cleanup"),
)


def _optional_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _ledger_timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _ledger_positive_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _ledger_memory_config(memory: Mapping[str, object]) -> dict[str, object]:
    utilization = _optional_duration(memory.get("gpu_memory_utilization"))
    if utilization is not None and not 0 < utilization <= 1:
        utilization = None
    return {
        "gpu_memory_utilization": utilization,
        "max_num_seqs": _ledger_positive_integer(memory.get("max_num_seqs")),
        "max_num_batched_tokens": _ledger_positive_integer(
            memory.get("max_num_batched_tokens")
        ),
        "enable_prefix_caching": memory.get("enable_prefix_caching")
        if isinstance(memory.get("enable_prefix_caching"), bool)
        else None,
        "max_model_len": _ledger_positive_integer(memory.get("max_model_len")),
        "tensor_parallel_size": _ledger_positive_integer(
            memory.get("tensor_parallel_size")
        ),
        "data_parallel_size": _ledger_positive_integer(
            memory.get("data_parallel_size")
        ),
    }


def _ledger_stage(
    value: object,
    *,
    expected_name: str,
    model_role: str | None,
    cleanup_termination_expected: bool = False,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    observed_name = value.get("stage")
    if observed_name is not None and observed_name != expected_name:
        _fail(f"execution ledger stage expected {expected_name}, got {observed_name}")
    exit_code = value.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None
    raw_status = value.get("status")
    if exit_code is not None:
        status = "completed" if exit_code == 0 else "failed"
    elif expected_name == "owned_process_cleanup" and (
        raw_status in {"already_clean", "best_effort"}
        or (
            cleanup_termination_expected
            and raw_status == "terminated_lingering_processes"
        )
    ):
        status = "completed"
    elif raw_status in {
        "failed",
        "remaining_processes",
        "cleanup_with_errors",
        "best_effort_with_errors",
        "terminated_lingering_processes",
    }:
        status = "failed"
    else:
        status = "incomplete"
    stage: dict[str, object] = {
        "stage": expected_name,
        "model_role": model_role,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": _optional_duration(value.get("duration_seconds")),
    }
    if expected_name == "evaluation_process":
        stage["peak_memory_used_mib"] = _optional_duration(
            value.get("peak_memory_used_mib")
        )
        stage["peak_memory_fraction"] = _optional_duration(
            value.get("peak_memory_fraction")
        )
    return stage


def build_execution_ledger(
    ledger_root: Path,
    *,
    selected: Mapping[Path, object],
) -> dict[str, object]:
    """Build a content-free timing ledger from immediate matrix manifests only."""

    root = ledger_root.resolve()
    if not root.is_dir():
        _fail("execution ledger root is not a directory")
    selected_resolved = {path.resolve(): spec for path, spec in selected.items()}
    entries: list[dict[str, object]] = []
    discovered: set[Path] = set()
    seen_run_ids: set[str] = set()
    manifest_paths: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            _fail(f"execution ledger refuses symlink child: {child.name}")
        if not child.is_dir():
            continue
        manifest_path = child / "matrix_manifest.json"
        if not manifest_path.exists():
            continue
        if manifest_path.is_symlink():
            _fail(f"execution ledger refuses symlink manifest: {child.name}")
        resolved_manifest = manifest_path.resolve()
        if not resolved_manifest.is_relative_to(root) or resolved_manifest.parent != child.resolve():
            _fail(f"execution ledger manifest escapes its run directory: {child.name}")
        manifest_paths.append(manifest_path)
    for manifest_path in manifest_paths:
        matrix_root = manifest_path.parent.resolve()
        manifest = _read_json(manifest_path, f"execution ledger {matrix_root.name}")
        if manifest.get("schema_version") != "hidden-policy-baseline-matrix-v1":
            _fail(f"execution ledger {matrix_root.name} has an unsupported schema")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
        ):
            _fail(f"execution ledger {matrix_root.name} has an invalid run id")
        if run_id != matrix_root.name:
            _fail(f"execution ledger run id differs from its directory name: {run_id}")
        if run_id in seen_run_ids:
            _fail(f"execution ledger contains duplicate run id: {run_id}")
        seen_run_ids.add(run_id)
        selected_spec = selected_resolved.get(matrix_root)
        selected_label: str | None = None
        if isinstance(selected_spec, Mapping):
            selected_label_value = selected_spec.get("label")
            selected_label = (
                selected_label_value if isinstance(selected_label_value, str) else None
            )
            if _sha256_file(manifest_path) != selected_spec.get("manifest_sha256"):
                _fail(f"selected execution ledger manifest changed: {run_id}")
            for identity_key in (
                "run_id",
                "scope",
                "backend",
                "repository_commit",
                "config_sha256",
            ):
                if manifest.get(identity_key) != selected_spec.get(identity_key):
                    _fail(f"selected execution ledger identity changed: {run_id}")
        elif selected_spec is not None:
            _fail("selected execution ledger specifications must be bound mappings")
        if selected_label not in {
            None,
            "pilot_vllm",
            "full_vllm",
            "pilot_hf_reference",
            "pilot_vllm_weak",
            "full_vllm_weak",
        }:
            _fail(f"selected execution ledger label is invalid: {selected_label}")
        status_value = manifest.get("status")
        status = (
            "interrupted"
            if isinstance(status_value, str) and status_value.startswith("aborted")
            else status_value
            if status_value in {"completed", "failed", "interrupted"}
            else "incomplete"
        )
        if status == "completed" and selected_label is None:
            status = "completed_unverified"
        scope = manifest.get("scope")
        backend = manifest.get("backend")
        if scope not in {"pilot", "full"} or backend not in {"vllm", "hf"}:
            _fail(f"execution ledger {run_id} has an invalid scope or backend")
        commit = manifest.get("repository_commit")
        if not isinstance(commit, str) or GIT_OID_PATTERN.fullmatch(commit) is None:
            commit = None
        config_sha256 = manifest.get("config_sha256")
        if not isinstance(config_sha256, str) or HASH_PATTERN.fullmatch(config_sha256) is None:
            config_sha256 = None

        execution = manifest.get("execution")
        execution = execution if isinstance(execution, Mapping) else {}
        memory = execution.get("vllm_memory_and_batching")
        memory = memory if isinstance(memory, Mapping) else {}
        public_memory = _ledger_memory_config(memory)
        models = execution.get("models")
        public_models = (
            [role for role in models if role in MODEL_ROLES]
            if isinstance(models, list)
            else []
        )
        stages: list[dict[str, object]] = []
        common = manifest.get("common_stages")
        if isinstance(common, list):
            for value in common:
                if not isinstance(value, Mapping):
                    continue
                name = value.get("stage")
                if name not in {
                    "repository_clean_check",
                    "runtime_doctor",
                    "gpu_availability",
                    "prepare_runtime",
                }:
                    _fail(f"execution ledger {run_id} has an unexpected common stage")
                stage = _ledger_stage(value, expected_name=str(name), model_role=None)
                if stage is not None:
                    stages.append(stage)
        raw_models = manifest.get("models")
        if isinstance(raw_models, Mapping):
            for role in MODEL_ROLES:
                model = raw_models.get(role)
                if not isinstance(model, Mapping):
                    continue
                for key, expected_name in LEDGER_STAGE_KEYS:
                    stage = _ledger_stage(
                        model.get(key), expected_name=expected_name, model_role=role
                    )
                    if stage is not None:
                        stages.append(stage)
                evaluation = model.get("evaluation")
                if isinstance(evaluation, Mapping):
                    harness = evaluation.get("harness_timing")
                    if isinstance(harness, Mapping) and isinstance(
                        harness.get("stages"), list
                    ):
                        for raw_stage in harness["stages"]:
                            if not isinstance(raw_stage, Mapping):
                                continue
                            name = raw_stage.get("stage")
                            if name not in {
                                "lm_eval_validate",
                                "model_load_and_evaluation",
                            }:
                                continue
                            stage = _ledger_stage(
                                raw_stage, expected_name=str(name), model_role=role
                            )
                            if stage is not None:
                                stages.append(stage)
        interruption_cleanup = manifest.get("interruption_cleanup")
        if isinstance(interruption_cleanup, Mapping):
            for role in MODEL_ROLES:
                stage = _ledger_stage(
                    interruption_cleanup.get(role),
                    expected_name="owned_process_cleanup",
                    model_role=role,
                    cleanup_termination_expected=True,
                )
                if stage is not None:
                    stages.append(stage)
        if manifest.get("error_stage") == "prompt_audit_dependency_load":
            stages.append(
                {
                    "stage": "prompt_audit_dependency_load",
                    "model_role": None,
                    "status": "failed",
                    "exit_code": None,
                    "duration_seconds": None,
                }
            )

        allocator = execution.get("pytorch_alloc_conf")
        if allocator != "expandable_segments:True":
            allocator = None

        entries.append(
            {
                # Replaced with a deterministic public attempt id after
                # chronological sorting below.  Never publish the raw,
                # operator-controlled run/directory name.
                "run_id": None,
                "selected_as": selected_label,
                "scope": scope,
                "backend": backend,
                "status": status,
                "repository_commit": commit,
                "config": {
                    "sha256": config_sha256,
                    "models": public_models,
                    "skip_prefetch": execution.get("skip_prefetch")
                    if isinstance(execution.get("skip_prefetch"), bool)
                    else None,
                    "hf_xet_high_performance": execution.get(
                        "hf_xet_high_performance"
                    )
                    if isinstance(execution.get("hf_xet_high_performance"), bool)
                    else None,
                    "pytorch_alloc_conf": allocator,
                    "gpu_poll_seconds": _optional_duration(
                        execution.get("gpu_poll_seconds")
                    ),
                    "vllm": public_memory,
                },
                "started_at_utc": _ledger_timestamp(manifest.get("started_at_utc")),
                "ended_at_utc": _ledger_timestamp(manifest.get("ended_at_utc")),
                "total_duration_seconds": _optional_duration(
                    manifest.get("duration_seconds")
                ),
                "stages": stages,
            }
        )
        discovered.add(matrix_root)
    missing_selected = set(selected_resolved).difference(discovered)
    if missing_selected:
        _fail("execution ledger root does not contain every selected matrix")
    # ``manifest_paths`` is already sorted by its private run-directory name,
    # which provides a deterministic tie breaker without exposing that name.
    entries.sort(key=lambda entry: str(entry["started_at_utc"] or ""))
    for index, entry in enumerate(entries, start=1):
        entry["run_id"] = f"attempt_{index:03d}"
    return {
        "schema_version": "hidden-policy-execution-ledger-v1",
        "entries": entries,
        "timing_note": (
            "Total duration is wall-clock time. Model stages can run concurrently, "
            "so stage durations must not be summed. A completed ledger status is a "
            "runner outcome; only selected matrices receive full artifact validation."
        ),
    }


def build_publication(
    pilot: MatrixArtifacts,
    full: MatrixArtifacts,
    *,
    config: Mapping[str, object],
    config_sha256: str,
    manifest_checksums: Mapping[str, object],
    hf_reference: MatrixArtifacts | None,
    weak_pilot: MatrixArtifacts | None = None,
    weak_full: MatrixArtifacts | None = None,
) -> dict[str, object]:
    _validate_cross_matrix(pilot, full)
    if (weak_pilot is None) != (weak_full is None):
        _fail("weak pilot and full matrices must be provided together")
    if weak_pilot is not None and weak_full is not None:
        _validate_cross_matrix(weak_pilot, weak_full)
        _validate_supplement(pilot, weak_pilot, label="pilot")
        _validate_supplement(full, weak_full, label="full")
    pilot_matrices = (pilot,) if weak_pilot is None else (weak_pilot, pilot)
    full_matrices = (full,) if weak_full is None else (weak_full, full)
    reference_model = pilot.models[HF_REFERENCE_ROLE]
    hf_agreement: dict[str, object]
    if hf_reference is None:
        hf_agreement = {
            "status": "not_run",
            "available": False,
            "gate_applied": False,
            "reason": "未提供 Qwen3.5-2B Hugging Face pilot reference matrix。",
        }
    else:
        if hf_reference.manifest["repository_commit"] != pilot.manifest[
            "repository_commit"
        ]:
            _fail("HF reference and vLLM matrices use different repository commits")
        hf_model = hf_reference.models[HF_REFERENCE_ROLE]
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
    has_weak = weak_pilot is not None
    limitations = [
        "本报告只覆盖 CAL；TEST-Q3 与 TEST-Q4 保持 sealed，不能据此回答最终 Q3/Q4。",
        (
            "四者均为原始 post-trained checkpoint；这是描述性 capability baseline，不是 sandbagging 训练结果。"
            if has_weak
            else "三者均为原始 post-trained checkpoint；这是描述性 capability baseline，不是 sandbagging 训练结果。"
        ),
        "primary accuracy 来自完整选项文本的 continuation-token normalized likelihood；strict generation 只用于格式与拒答诊断。",
        "BF16 与不同推理 kernel 可能产生数值差异；HF reference 仅对 2B pilot 做描述性预测、centered score、top-margin 与 accuracy-delta 比较，没有预设 pass/fail gate，也不能代表所有尺寸。",
        "subject 样本量不同，尤其 pilot 很小；subject 数值应视为诊断而非独立确认性结论。",
        "GPU 峰值来自定时轮询的整张物理卡，可能漏掉采样间峰值，也不等同于进程独占显存。",
        "allocator 字段记录冻结请求、评测前 runtime verification 与 postprocess 启动时快照；PyTorch 不提供完整内部 flag getter，因此它不是 evaluation 全程 allocator 状态的直接 trace。",
        "没有置信区间、训练 seed 或因果干预，因此不能从模型尺寸差异推出 scaling law 或机制结论。",
    ]
    if hf_reference is None:
        limitations.append(
            "未提供 HF reference matrix；当前报告没有 HF-vLLM backend prediction agreement 证据。"
        )
    return {
        "schema_version": "hidden-policy-baseline-publication-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_language": "zh-CN",
        "artifact_validation_status": "PASS",
        "hf_comparison_status": hf_agreement["status"],
        "pilot": _public_matrix_bundle(
            pilot_matrices, config, public_run_id="pilot_vllm"
        ),
        "full_cal": _public_matrix_bundle(
            full_matrices, config, public_run_id="full_vllm"
        ),
        "hf_reference": (
            _public_matrix(
                hf_reference,
                config,
                public_run_id="pilot_hf_reference",
            )
            if hf_reference is not None
            else None
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
                "full_vllm": full.models[HF_REFERENCE_ROLE].summary["provenance"][
                    "runtime_fingerprint"
                ],
                "pilot_hf_reference": (
                    hf_reference.models[HF_REFERENCE_ROLE].summary["provenance"][
                        "runtime_fingerprint"
                    ]
                    if hf_reference is not None
                    else None
                ),
                "pilot_vllm_weak": (
                    weak_pilot.models[WEAK_MODEL_ROLE].summary["provenance"][
                        "runtime_fingerprint"
                    ]
                    if weak_pilot is not None
                    else None
                ),
                "full_vllm_weak": (
                    weak_full.models[WEAK_MODEL_ROLE].summary["provenance"][
                        "runtime_fingerprint"
                    ]
                    if weak_full is not None
                    else None
                ),
            },
            "evaluation_protocol": {
                key: evaluation[key]
                for key in (
                    "backend",
                    "vllm_version",
                    "hf_xet_high_performance",
                    "pytorch_alloc_conf",
                    "pytorch_allocator_backend",
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


def _decimal(value: object) -> str:
    return f"{float(value):.6g}"


def _signed_percentage_points(value: object) -> str:
    return f"{100.0 * float(value):+.2f} pp"


def _cell(value: object) -> str:
    return escape(str(value), quote=True)


def _present_roles(matrix: Mapping[str, object]) -> tuple[str, ...]:
    models = matrix["models"]
    return tuple(role for role in MODEL_ROLES if role in models)


def _aggregate_rows(matrix: Mapping[str, object]) -> str:
    rows: list[str] = []
    for role in _present_roles(matrix):
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
    roles = _present_roles(matrix)
    subjects = sorted(
        {
            subject
            for role in roles
            for subject in models[role]["datasets"][dataset]["subjects"]
        }
    )
    headers = "".join(
        f"<th colspan='3'>{_cell(models[role]['display_name'])}</th>"
        for role in roles
    )
    subheaders = "".join("<th>N</th><th>Acc</th><th>Perm.</th>" for _ in roles)
    rows: list[str] = []
    for subject in subjects:
        cells = [f"<td>{_cell(subject)}</td>"]
        for role in roles:
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
    for role in _present_roles(matrix):
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
            f"<td>{_seconds(timing['process_cleanup_seconds'])}</td>"
            f"<td>{float(gpu['peak_memory_used_mib']) / 1024:.2f} GiB</td>"
            f"<td>{float(gpu['peak_memory_fraction']) * 100:.1f}%</td>"
            f"<td>{float(gpu['mean_utilization_percent']):.1f}%</td>"
            f"<td>{float(gpu['peak_utilization_percent']):.0f}%</td>"
            f"<td>{float(gpu['peak_power_watts']):.1f} W</td>"
            f"<td>{int(gpu['sample_count'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _execution_ledger_html(ledger: Mapping[str, object]) -> str:
    blocks: list[str] = []
    for entry in ledger["entries"]:
        config = entry["config"]
        vllm = config["vllm"]
        config_summary = (
            f"util={vllm.get('gpu_memory_utilization')} · "
            f"seq={vllm.get('max_num_seqs')} · "
            f"tokens={vllm.get('max_num_batched_tokens')} · "
            f"alloc={config.get('pytorch_alloc_conf') or 'not recorded'}"
        )
        stage_rows = "".join(
            "<tr>"
            f"<td>{_cell(stage['model_role'] or 'shared')}</td>"
            f"<td>{_cell(stage['stage'])}</td>"
            f"<td>{_cell(stage['status'])}</td>"
            f"<td>{_cell(stage['exit_code'] if stage['exit_code'] is not None else '—')}</td>"
            f"<td>{_seconds(stage['duration_seconds'])}</td>"
            f"<td>{_cell(stage.get('peak_memory_used_mib') or '—')}</td>"
            "</tr>"
            for stage in entry["stages"]
        )
        commit = entry["repository_commit"]
        config_hash = config["sha256"]
        blocks.append(
            "<details class='ledger-run'>"
            "<summary>"
            f"<span class='run-status {_cell(entry['status'])}'>{_cell(entry['status'])}</span> "
            f"<strong>{_cell(entry['run_id'])}</strong> · "
            f"{_cell(entry['scope'])}/{_cell(entry['backend'])} · "
            f"{_seconds(entry['total_duration_seconds'])}"
            "</summary>"
            "<div class='ledger-meta'>"
            f"Selected: {_cell(entry['selected_as'] or 'no')} · "
            f"Commit: <code>{_cell(commit[:12] if commit else 'not recorded')}</code> · "
            f"Config: <code>{_cell(config_hash[:12] if config_hash else 'not recorded')}</code> · "
            f"Start: {_cell(entry['started_at_utc'] or 'not recorded')}<br>"
            f"{_cell(config_summary)}"
            "</div>"
            "<div class='table-wrap'><table><thead><tr><th>Model</th><th>Stage</th>"
            "<th>Status</th><th>Exit</th><th>Duration</th><th>Peak MiB</th>"
            f"</tr></thead><tbody>{stage_rows}</tbody></table></div></details>"
        )
    return "".join(blocks)


def _metric_guide_html() -> str:
    """Explain every reader-facing metric without exposing benchmark content."""

    return """
<section class="metric-guide">
  <h2>先读这里：指标含义与计算方式</h2>
  <p>同一道选择题会走两条互补的评测路线。<strong>Likelihood</strong> 比较模型给四个完整选项文本的概率分数；<strong>Strict generation</strong> 则让模型真的生成一个字母。两者不一致并不矛盾：前者更接近受约束的知识判断，后者还会受到指令遵循与输出格式影响。</p>
  <div class="metric-mode-grid">
    <article class="metric-mode">
      <span class="metric-kicker">主结果 · 3 views / 题</span>
      <h3>Likelihood（完整选项似然）</h3>
      <ol>
        <li>在同一题目提示下，分别把 A–D 的<strong>完整选项文本</strong>当作候选 continuation。</li>
        <li>计算每个候选文本所有 token 的 log likelihood 之和，再除以该候选的 token 数。</li>
        <li>选择平均分最高（通常即数值最不负）的选项；换序 view 的显示位置会先映射回原始 semantic option 再判分。</li>
      </ol>
      <div class="metric-formula"><code>score(i) = Σₜ log p(tokenₜ | prompt, prefix) / Tᵢ</code><br><code>prediction = argmaxᵢ score(i)</code></div>
      <p class="metric-foot">它不要求模型生成字母，因此主要衡量四选一相对偏好。按 token 取平均是为了减轻选项长度差异，但不代表完全消除所有长度或措辞效应。</p>
    </article>
    <article class="metric-mode strict-mode">
      <span class="metric-kicker">格式诊断 · canonical only</span>
      <h3>Strict generation（严格生成）</h3>
      <ol>
        <li>模型看到题目和 A–D 选项，并被要求只生成一个大写字母。</li>
        <li>解析器只接受可带前后空白的单个 <code>A</code>、<code>B</code>、<code>C</code> 或 <code>D</code>。</li>
        <li>只有“解析有效且字母对应正确显示位置”才记为答对；例如 <code>Answer: C</code> 会记为 Invalid。</li>
      </ol>
      <div class="metric-formula"><code>correct = valid ∧ (predicted display index = gold display index)</code></div>
      <p class="metric-foot">它同时衡量知识判断、指令遵循和格式控制。当前只在原始选项顺序（permutation 0）上每题生成一次。</p>
    </article>
  </div>

  <h3>汇总表中的指标</h3>
  <p class="denominator-note"><strong>先看分母：</strong><code>N</code> 是不重复题目数。Likelihood 每题有 3 个选项顺序，因此共有 <code>3N</code> 个 views；Strict generation 每题只有 1 个 canonical view，因此共有 <code>N</code> 次生成。</p>
  <div class="metric-grid">
    <article class="metric-card"><h4>Canonical Acc ↑</h4><code>permutation 0 答对数 / N</code><p>Likelihood 在原始选项顺序上的准确率，也是本报告的主能力指标。</p></article>
    <article class="metric-card"><h4>All-view Acc ↑</h4><code>3 种排列全部答对数 / 3N</code><p>Likelihood 汇总三个选项顺序后的准确率；预测会先映射回 semantic option。</p></article>
    <article class="metric-card"><h4>Permutation consistency ↑</h4><code>三次 semantic prediction 完全相同的题数 / N</code><p>衡量换序稳定性。<strong>Consistency 不等于正确率</strong>：稳定地选错也会得到一致。</p></article>
    <article class="metric-card"><h4>Strict Acc ↑</h4><code>严格生成且答对的题数 / N</code><p>只要输出格式无效、拒答或选错，都会进入分母并记为不正确。</p></article>
    <article class="metric-card"><h4>Invalid ↓</h4><code>非拒答的无效格式数 / N</code><p>回复不是单个 A–D，且未命中拒答模式。例如解释文字或 <code>Answer: C</code>。</p></article>
    <article class="metric-card"><h4>Refusal ↓</h4><code>命中拒答模式的回复数 / N</code><p>拒答与 Invalid 互斥；两者相加就是所有未被解析为有效字母的比例。</p></article>
  </div>

  <details class="metric-details">
    <summary>HF ↔ vLLM 对照指标怎么读？</summary>
    <ul>
      <li><strong>Prediction agreement：</strong>两个 backend 选出同一 semantic option 的 views 比例；strict agreement 还要求解析状态与字母位置都相同。越接近 100% 说明决策越一致，但不保证决策正确。</li>
      <li><strong>Centered per-option normalized LL mean |Δ|：</strong>先在每个 backend 内把同一 view 的四个 normalized likelihood 减去其均值，再逐选项计算 HF 与 vLLM 的绝对差并取平均。越接近 0，表示相对分数形状越相似。</li>
      <li><strong>Top-margin mean Δ (vLLM−HF)：</strong>每个 view 的最高分减次高分得到 top margin，再计算 vLLM margin − HF margin 的平均值。正值表示 vLLM 平均更有“领先幅度”，不表示更准确。</li>
      <li><strong>All-view accuracy Δ (vLLM−HF)：</strong>两个 backend 的 All-view Acc 之差，以 percentage points（pp）表示；正值表示该 pilot 上 vLLM 更高。</li>
    </ul>
    <p class="metric-foot">这些只用于检查 backend 差异，目前没有 pass/fail 阈值。</p>
  </details>
  <details class="metric-details">
    <summary>耗时与 GPU 指标怎么读？</summary>
    <p>各阶段时间是 wall-clock duration。峰值显存、显存占比、平均/峰值 GPU 利用率与峰值功耗来自固定间隔的整卡 <code>nvidia-smi</code> 采样；<code>Samples</code> 是采样次数。它们描述吞吐和资源使用，不是模型质量指标，也可能漏掉采样间隔内的瞬时峰值。</p>
  </details>
</section>"""


def render_html(report: Mapping[str, object]) -> str:
    full = report["full_cal"]
    pilot = report["pilot"]
    agreement = report["hf_vllm_pilot_agreement"]
    if agreement["available"]:
        all_views = agreement["all_views"]
        centered = all_views["centered_per_option_normalized_ll_difference"]
        margin = all_views["top_margin_difference"]
        accuracy = all_views["accuracy"]
        agreement_html = (
            "<p class='notice'><strong>HF comparison: DESCRIPTIVE.</strong> "
            "未设置 pass/fail 阈值；Artifact PASS 不代表两个 backend 等价。</p>"
            "<div class='agreement-grid'>"
            f"<div><strong>{_pct(all_views['prediction_agreement'])}</strong>"
            f"<span>全部 likelihood views ({all_views['matching_predictions']}/"
            f"{all_views['views']})</span></div>"
            f"<div><strong>{_pct(agreement['canonical_views']['prediction_agreement'])}</strong>"
            f"<span>canonical views</span></div>"
            f"<div><strong>{_pct(agreement['strict_generation']['prediction_agreement'])}</strong>"
            f"<span>strict parsed predictions</span></div>"
            f"<div><strong>{_decimal(centered['mean_absolute'])}</strong>"
            "<span>centered per-option normalized LL mean |Δ|</span></div>"
            f"<div><strong>{_decimal(margin['mean_signed_vllm_minus_hf'])}</strong>"
            "<span>top-margin mean Δ (vLLM−HF)</span></div>"
            f"<div><strong>{_signed_percentage_points(accuracy['delta_vllm_minus_hf'])}</strong>"
            "<span>all-view accuracy Δ (vLLM−HF)</span></div>"
            "</div>"
        )
    else:
        agreement_html = (
            "<p class='notice'><strong>HF comparison: NOT RUN.</strong> "
            f"{_cell(agreement['reason'])}</p>"
        )

    provenance = report["provenance"]
    protocol = provenance["evaluation_protocol"]
    software = provenance["software_environment"]
    limitation_items = "".join(
        f"<li>{_cell(item)}</li>" for item in report["limitations"]
    )
    full_roles = _present_roles(full)
    model_size_text = (
        "Qwen3.5-0.8B、2B、4B、9B"
        if WEAK_MODEL_ROLE in full_roles
        else "Qwen3.5-2B、4B、9B"
    )
    model_cards = "".join(
        (
            "<div class='model-card'>"
            f"<span>{_cell(full['models'][role]['display_name'])}</span>"
            f"<strong>WMDP {_pct(full['models'][role]['datasets']['wmdp']['option_likelihood']['canonical_accuracy'])}</strong>"
            f"<small>MMLU {_pct(full['models'][role]['datasets']['mmlu']['option_likelihood']['canonical_accuracy'])}</small>"
            "</div>"
        )
        for role in full_roles
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
h2{{font-size:21px;margin:0 0 14px}} h3{{font-size:17px;margin:22px 0 10px}} .cards{{display:grid;grid-template-columns:repeat({len(full_roles)},1fr);gap:14px;margin-top:20px}}
.model-card{{background:#fff;color:var(--ink);border-radius:12px;padding:17px;display:flex;flex-direction:column}} .model-card span{{color:var(--muted)}} .model-card strong{{font-size:22px;color:var(--accent);margin-top:6px}} .model-card small{{font-size:14px}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;background:#daf2e6;color:var(--good);font-weight:700;margin-right:7px}} .badge.secondary{{background:#e7edf1;color:#425868}} .meta{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .meta div{{background:var(--wash);padding:12px;border-radius:9px}} .meta span{{display:block;color:var(--muted);font-size:12px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th{{background:#edf3f6;color:#334b59}} th:first-child,td:first-child{{text-align:left}} tbody tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f8fafb}}
.agreement-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .agreement-grid div{{background:#edf7fa;border-left:4px solid var(--accent);padding:16px;border-radius:8px}} .agreement-grid strong{{display:block;color:var(--accent);font-size:24px}} .agreement-grid span{{color:var(--muted)}}
.metric-guide>p:first-of-type{{font-size:16px;max-width:1000px}} .metric-mode-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .metric-mode{{background:#eef7fa;border:1px solid #cfe4eb;border-radius:12px;padding:18px}} .metric-mode.strict-mode{{background:#fff7eb;border-color:#f0dbc0}} .metric-mode h3{{margin:4px 0 10px}} .metric-mode ol{{margin:0;padding-left:21px}} .metric-mode li+li{{margin-top:7px}} .metric-kicker{{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:.03em;text-transform:uppercase}} .metric-formula{{background:#fff;border:1px solid var(--line);border-radius:8px;margin:14px 0 10px;padding:11px;overflow:auto}} .metric-formula code{{white-space:nowrap}} .metric-foot{{color:var(--muted);font-size:13px;margin-bottom:0}} .denominator-note{{background:#edf3f6;border-radius:9px;padding:12px}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}} .metric-card{{border:1px solid var(--line);border-radius:10px;padding:14px}} .metric-card h4{{font-size:15px;margin:0 0 7px;color:var(--accent)}} .metric-card code{{display:block;background:var(--wash);border-radius:6px;padding:7px;margin-bottom:8px}} .metric-card p{{margin:0;color:var(--muted);font-size:13px}} .metric-details{{border-top:1px solid var(--line);margin-top:16px;padding-top:13px}} .metric-details summary{{cursor:pointer;font-weight:700;color:#334b59}} .metric-details ul{{margin-bottom:5px}}
.notice{{background:#fff3dd;border-left:4px solid var(--accent2);padding:13px}} code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}} ul{{padding-left:22px}} footer{{color:var(--muted);text-align:center;margin-top:24px;font-size:12px}}
.ledger-run{{border:1px solid var(--line);border-radius:9px;margin:10px 0;padding:12px}} .ledger-run summary{{cursor:pointer}} .ledger-meta{{color:var(--muted);font-size:13px;margin:10px 0}} .run-status{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700;background:#e7edf1}} .run-status.completed{{background:#daf2e6;color:#217a52}} .run-status.failed{{background:#fde3e0;color:#9c2f27}} .run-status.interrupted,.run-status.incomplete{{background:#fff0d2;color:#8a5a08}}
@media(max-width:900px){{.metric-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}} @media(max-width:760px){{.cards,.meta,.agreement-grid,.metric-mode-grid,.metric-grid{{grid-template-columns:1fr}} main{{padding:18px 10px 40px}} section,header{{padding:18px}}}}
</style>
</head>
<body><main>
<header><span class="badge">Selected artifacts 验证 {_cell(report['artifact_validation_status'])}</span><span class="badge secondary">HF 对照 {_cell(str(report['hf_comparison_status']).replace('_', ' ').upper())}</span><h1>Qwen3.5 基础能力测试</h1><p>WMDP / MMLU · non-thinking · full-option likelihood · CAL only</p><div class="cards">{model_cards}</div></header>
<section><h2>读者摘要</h2><p>本页比较 {model_size_text} 原始 post-trained checkpoint。主结果是 full CAL 的完整选项文本 token-normalized likelihood；三种选项排列用于诊断位置敏感性，canonical strict generation 用于诊断生成正确率、格式失败和拒答。此处没有训练 sandbagger，也没有解封 Q3/Q4 test。</p><div class="meta"><div><span>主后端</span>{_cell(full['backend'])}</div><div><span>Full CAL 总耗时</span>{_seconds(full['matrix_duration_seconds'])}</div><div><span>评测代码 commit</span><code>{_cell(provenance['evaluated_repository_commit'])}</code></div></div></section>
{_metric_guide_html()}
<section><h2>Full CAL 汇总</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>数据集</th><th>N</th><th>Canonical Acc</th><th>All-view Acc</th><th>Permutation consistency</th><th>Strict Acc</th><th>Invalid</th><th>Refusal</th></tr></thead><tbody>{_aggregate_rows(full)}</tbody></table></div></section>
<section><h2>Full CAL subject 诊断</h2><h3>WMDP</h3>{_subject_table(full,'wmdp')}<h3>MMLU</h3>{_subject_table(full,'mmlu')}<p class="notice">Subject 表中的 Acc 为 canonical-order likelihood accuracy；Perm. 为三种排列映射回 semantic option 后的一致率。</p></section>
<section><h2>32-item pilot 汇总</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>数据集</th><th>N</th><th>Canonical Acc</th><th>All-view Acc</th><th>Permutation consistency</th><th>Strict Acc</th><th>Invalid</th><th>Refusal</th></tr></thead><tbody>{_aggregate_rows(pilot)}</tbody></table></div></section>
<section><h2>HF ↔ vLLM pilot 描述性比较</h2>{agreement_html}</section>
<section><h2>Full CAL 耗时与 GPU 峰值</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>Prefetch</th><th>Prompt audit</th><th>lm-eval validate</th><th>Load + eval</th><th>Postprocess</th><th>Cleanup</th><th>峰值显存</th><th>显存占比</th><th>平均利用率</th><th>峰值利用率</th><th>峰值功耗</th><th>Samples</th></tr></thead><tbody>{_timing_rows(full)}</tbody></table></div><p class="notice">三模型并行执行，各自绑定一张物理 GPU；耗时会受共享 CPU、磁盘和 PCIe 影响。GPU 数值来自固定间隔的整卡 nvidia-smi 轮询，因此可能漏掉瞬时峰值，也不是进程级精确 profile。</p></section>
<section><h2>执行与调参记录</h2><p class="notice">总耗时是 wall-clock；模型并行执行时，各阶段耗时不可直接相加。ledger 中 completed 只表示 runner 正常结束；只有标记为 selected 的正式矩阵通过了本报告的完整 artifact 校验。</p>{_execution_ledger_html(report['execution_ledger'])}</section>
<section><h2>可复现性</h2><div class="meta"><div><span>lm-evaluation-harness</span><code>{_cell(provenance['harness']['version'])} · {_cell(provenance['harness']['commit'])}</code></div><div><span>vLLM / Transformers</span>{_cell(protocol['vllm_version'])} / {_cell(software['transformers'])}</div><div><span>PyTorch / CUDA</span>{_cell(software['torch'])} / {_cell(software['torch_cuda'])}</div><div><span>CUDA allocator（postprocess snapshot）</span><code>{_cell(software['pytorch_allocator_backend'])} · {_cell(software['pytorch_alloc_conf_at_snapshot'])}</code></div><div><span>Prompt</span>{_cell(protocol['prompt_protocol'])}; thinking={_cell(protocol['enable_thinking'])}</div><div><span>Normalization</span>{_cell(protocol['normalization'])}</div><div><span>Runtime fingerprint (full)</span><code>{_cell(provenance['runtime_fingerprints']['full_vllm'])}</code></div></div></section>
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
    weak_pilot_matrix: Path | None = None,
    weak_full_matrix: Path | None = None,
    execution_ledger_root: Path | None = None,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_html: Path = DEFAULT_OUTPUT_HTML,
) -> dict[str, object]:
    if (weak_pilot_matrix is None) != (weak_full_matrix is None):
        _fail("weak pilot and full matrices must be provided together")
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

    selected_manifest_hashes = {
        pilot_matrix.resolve(): _sha256_file(pilot_matrix / "matrix_manifest.json"),
        full_matrix.resolve(): _sha256_file(full_matrix / "matrix_manifest.json"),
    }
    if hf_reference_matrix is not None:
        selected_manifest_hashes[hf_reference_matrix.resolve()] = _sha256_file(
            hf_reference_matrix / "matrix_manifest.json"
        )
    for weak_matrix in (weak_pilot_matrix, weak_full_matrix):
        if weak_matrix is not None:
            selected_manifest_hashes[weak_matrix.resolve()] = _sha256_file(
                weak_matrix / "matrix_manifest.json"
            )

    pilot = load_matrix(
        pilot_matrix,
        expected_backend="vllm",
        expected_scope="pilot",
        expected_roles=BASE_MODEL_ROLES,
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
        expected_roles=BASE_MODEL_ROLES,
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
            expected_roles=(HF_REFERENCE_ROLE,),
            config=config,
            config_sha256=config_sha256,
            manifest_checksums=manifest_checksums,
            expected_counts=pilot_counts,
            expected_dataset_revisions=expected_revisions,
        )
        if hf_reference_matrix is not None
        else None
    )
    weak_pilot = (
        load_matrix(
            weak_pilot_matrix,
            expected_backend="vllm",
            expected_scope="pilot",
            expected_roles=(WEAK_MODEL_ROLE,),
            config=config,
            config_sha256=config_sha256,
            manifest_checksums=manifest_checksums,
            expected_counts=pilot_counts,
            expected_dataset_revisions=expected_revisions,
        )
        if weak_pilot_matrix is not None
        else None
    )
    weak_full = (
        load_matrix(
            weak_full_matrix,
            expected_backend="vllm",
            expected_scope="full",
            expected_roles=(WEAK_MODEL_ROLE,),
            config=config,
            config_sha256=config_sha256,
            manifest_checksums=manifest_checksums,
            expected_counts=full_counts,
            expected_dataset_revisions=expected_revisions,
        )
        if weak_full_matrix is not None
        else None
    )
    for selected_path, expected_hash in selected_manifest_hashes.items():
        if _sha256_file(selected_path / "matrix_manifest.json") != expected_hash:
            _fail("a selected matrix manifest changed while it was being validated")
    report = build_publication(
        pilot,
        full,
        config=config,
        config_sha256=config_sha256,
        manifest_checksums=manifest_checksums,
        hf_reference=hf_reference,
        weak_pilot=weak_pilot,
        weak_full=weak_full,
    )
    def selected_spec(
        matrix_path: Path, matrix: MatrixArtifacts, label: str
    ) -> dict[str, object]:
        return {
            "label": label,
            "manifest_sha256": selected_manifest_hashes[matrix_path.resolve()],
            **{
                key: matrix.manifest.get(key)
                for key in (
                    "run_id",
                    "scope",
                    "backend",
                    "repository_commit",
                    "config_sha256",
                )
            },
        }

    selected_matrices: dict[Path, object] = {
        pilot_matrix: selected_spec(pilot_matrix, pilot, "pilot_vllm"),
        full_matrix: selected_spec(full_matrix, full, "full_vllm"),
    }
    if hf_reference_matrix is not None:
        assert hf_reference is not None
        selected_matrices[hf_reference_matrix] = selected_spec(
            hf_reference_matrix, hf_reference, "pilot_hf_reference"
        )
    if weak_pilot_matrix is not None and weak_full_matrix is not None:
        assert weak_pilot is not None and weak_full is not None
        selected_matrices[weak_pilot_matrix] = selected_spec(
            weak_pilot_matrix, weak_pilot, "pilot_vllm_weak"
        )
        selected_matrices[weak_full_matrix] = selected_spec(
            weak_full_matrix, weak_full, "full_vllm_weak"
        )
    report["execution_ledger"] = build_execution_ledger(
        execution_ledger_root or pilot_matrix.parent,
        selected=selected_matrices,
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
    parser.add_argument("--weak-pilot-matrix", type=Path)
    parser.add_argument("--weak-full-matrix", type=Path)
    parser.add_argument(
        "--execution-ledger-root",
        type=Path,
        help=(
            "directory whose immediate child matrix manifests form the "
            "content-free execution/tuning ledger"
        ),
    )
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
        weak_pilot_matrix=args.weak_pilot_matrix,
        weak_full_matrix=args.weak_full_matrix,
        execution_ledger_root=args.execution_ledger_root,
        config_path=args.config,
        split_metadata_path=args.split_metadata,
        output_json=args.output_json,
        output_html=args.output_html,
    )
    print(
        json.dumps(
            {
                "artifact_validation_status": report["artifact_validation_status"],
                "hf_comparison_status": report["hf_comparison_status"],
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
