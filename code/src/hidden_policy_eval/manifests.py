"""Deterministic, content-addressed manifests for Plan 4 Experiment 0.

This module deliberately knows nothing about Hugging Face datasets.  Callers pass
plain mappings so the split protocol can be tested without downloading data or
importing an evaluation framework.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "hidden-policy-eval-manifest-v3"
CANONICALIZATION_VERSION = "canonical-mcq-row-v3"
DEFAULT_SPLIT_SALT = "hidden-policy-plan4-v1"

_SEALED_ENTRY_FIELDS = frozenset(
    {
        "dataset",
        "dataset_revision",
        "stable_id",
        "content_hash",
        "subject",
        "source_split",
        "split",
    }
)
_CONTENT_FIELDS = frozenset({"question", "choices", "answer"})


def _normalize_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    # Make hashes independent of platform line endings and inconsequential
    # whitespace at the boundary.  Internal whitespace remains meaningful.
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_answer(value: object, *, number_of_choices: int) -> int:
    if isinstance(value, bool):
        raise TypeError("answer must be an integer index or an option label")
    if isinstance(value, int):
        answer = value
    elif isinstance(value, str):
        label = value.strip().upper()
        if len(label) != 1 or not ("A" <= label <= "Z"):
            raise ValueError("string answer must be a single option label")
        answer = ord(label) - ord("A")
    else:
        raise TypeError("answer must be an integer index or an option label")

    if not 0 <= answer < number_of_choices:
        raise ValueError("answer is outside the available choices")
    return answer


def canonical_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical, content-bearing representation of one MCQ row.

    Only fields that define the semantic item are included.  In particular, an
    upstream row number or split name cannot change an item's content identity.
    """

    try:
        raw_choices = row["choices"]
    except KeyError as exc:
        raise KeyError("row is missing required field 'choices'") from exc
    if isinstance(raw_choices, (str, bytes)) or not isinstance(raw_choices, Iterable):
        raise TypeError("choices must be an iterable of strings")
    choices = tuple(
        _normalize_text(choice, field=f"choices[{index}]")
        for index, choice in enumerate(raw_choices)
    )
    if len(choices) < 2:
        raise ValueError("an MCQ row must contain at least two choices")

    return {
        "subject": _normalize_text(row.get("subject"), field="subject"),
        "question": _normalize_text(row.get("question"), field="question"),
        "choices": list(choices),
        "answer": _normalize_answer(row.get("answer"), number_of_choices=len(choices)),
    }


def canonical_row_bytes(row: Mapping[str, object]) -> bytes:
    """Serialize a canonical row identically across runs and platforms."""

    return json.dumps(
        canonical_row(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonicalize_row(row: Mapping[str, object]) -> str:
    """Return the canonical JSON string used to derive IDs and content hashes."""

    return canonical_row_bytes(row).decode("utf-8")


def content_hash(row: Mapping[str, object]) -> str:
    """Hash public prompt content and subject, deliberately excluding the label."""

    public_content = canonical_row(row)
    public_content.pop("answer")
    payload = json.dumps(
        public_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_item_id(row: Mapping[str, object]) -> str:
    """Identify the rendered MCQ prompt, excluding subject and answer label."""

    identity = canonical_row(row)
    identity.pop("subject")
    identity.pop("answer")
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"mcq-{hashlib.sha256(payload).hexdigest()}"


def make_source_record(
    *,
    subject: str,
    source_split: str,
    question: str,
    choices: Iterable[str],
    answer: int | str,
) -> dict[str, object]:
    """Create the framework-neutral row shape consumed by the split builder."""

    record: dict[str, object] = {
        "subject": subject,
        "source_split": source_split,
        "question": question,
        "choices": list(choices),
        "answer": answer,
    }
    # Validate eagerly while preserving the source split for the manifest.
    canonical_row(record)
    _source_split(record)
    return record


def _source_split(row: Mapping[str, object]) -> str:
    value = row.get("source_split", row.get("split"))
    return _normalize_text(value, field="source_split").lower()


def _assignment_hash(
    *, dataset_revision: str, subject: str, item_id: str, split_salt: str
) -> str:
    payload = json.dumps(
        [split_salt, dataset_revision, subject, item_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _largest_remainder_counts(size: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    """Allocate exactly ``size`` slots according to integer ratio ``weights``."""

    total = sum(weights)
    base = [(size * weight) // total for weight in weights]
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: (-(size * weights[index] % total), index),
    )
    for index in remainder_order[: size - sum(base)]:
        base[index] += 1
    return tuple(base)


def _ranked_assignments(
    rows: list[dict[str, str]],
    *,
    dataset_revision: str,
    split_salt: str,
    names: tuple[str, ...],
    weights: tuple[int, ...],
) -> dict[str, str]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _assignment_hash(
                dataset_revision=dataset_revision,
                subject=row["subject"],
                item_id=row["stable_id"],
                split_salt=split_salt,
            ),
            row["stable_id"],
        ),
    )
    counts = _largest_remainder_counts(len(ranked), weights)
    assignments: dict[str, str] = {}
    cursor = 0
    if len(names) != len(counts):
        raise ValueError("split names and allocation weights must have equal length")
    for name, count in zip(names, counts):
        for row in ranked[cursor : cursor + count]:
            assignments[row["stable_id"]] = name
        cursor += count
    return assignments


def build_sealed_manifest(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    dataset_revision: str,
    split_salt: str = DEFAULT_SPLIT_SALT,
) -> dict[str, object]:
    """Build a deterministic ID-only manifest for WMDP or MMLU.

    WMDP is split within each subject as CAL/TEST-Q3/TEST-Q4 with a 20/40/40
    largest-remainder allocation.  MMLU dev and validation remain CAL; MMLU
    test is split within each subject 50/50 between TEST-Q3 and TEST-Q4.
    """

    normalized_dataset = dataset.strip().lower()
    if normalized_dataset not in {"wmdp", "mmlu"}:
        raise ValueError("dataset must be 'wmdp' or 'mmlu'")
    revision = _normalize_text(dataset_revision, field="dataset_revision")
    salt = _normalize_text(split_salt, field="split_salt")

    prepared: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        canonical = canonical_row(row)
        digest = content_hash(canonical)
        item_id = stable_item_id(canonical)
        if item_id in seen_ids:
            raise ValueError(f"duplicate canonical item: {item_id}")
        seen_ids.add(item_id)
        prepared.append(
            {
                "dataset": normalized_dataset,
                "dataset_revision": revision,
                "stable_id": item_id,
                "content_hash": digest,
                "subject": str(canonical["subject"]),
                "source_split": _source_split(row),
            }
        )

    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prepared:
        by_subject[row["subject"]].append(row)

    assignments: dict[str, str] = {}
    if normalized_dataset == "wmdp":
        for subject_rows in by_subject.values():
            assignments.update(
                _ranked_assignments(
                    subject_rows,
                    dataset_revision=revision,
                    split_salt=salt,
                    names=("CAL", "TEST-Q3", "TEST-Q4"),
                    weights=(20, 40, 40),
                )
            )
    else:
        allowed = {"dev", "validation", "test"}
        unsupported = {row["source_split"] for row in prepared} - allowed
        if unsupported:
            raise ValueError(f"unsupported MMLU source split(s): {sorted(unsupported)}")
        for row in prepared:
            if row["source_split"] in {"dev", "validation"}:
                assignments[row["stable_id"]] = "CAL"
        test_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in prepared:
            if row["source_split"] == "test":
                test_by_subject[row["subject"]].append(row)
        for subject_rows in test_by_subject.values():
            assignments.update(
                _ranked_assignments(
                    subject_rows,
                    dataset_revision=revision,
                    split_salt=salt,
                    names=("TEST-Q3", "TEST-Q4"),
                    weights=(50, 50),
                )
            )

    entries = [dict(row, split=assignments[row["stable_id"]]) for row in prepared]
    entries.sort(key=lambda row: (row["subject"], row["stable_id"]))
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "dataset": normalized_dataset,
        "dataset_revision": revision,
        "split_salt": salt,
        "entries": entries,
    }
    validate_sealed_manifest(manifest)
    return manifest


def build_plan4_split(
    records: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    dataset_revision: str,
    split_salt: str = DEFAULT_SPLIT_SALT,
) -> dict[str, object]:
    """Named Plan 4 entry point; equivalent to :func:`build_sealed_manifest`."""

    return build_sealed_manifest(
        records,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split_salt=split_salt,
    )


def validate_sealed_manifest(manifest: Mapping[str, object]) -> None:
    """Raise if a manifest leaks MCQ content or has malformed entries."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if manifest.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ValueError("unsupported manifest canonicalization_version")
    if manifest.get("dataset") not in {"wmdp", "mmlu"}:
        raise ValueError("unsupported manifest dataset")
    for field in ("dataset_revision", "split_salt"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"manifest {field} must be a non-empty string")
    leaked_top_level = frozenset(manifest) & _CONTENT_FIELDS
    if leaked_top_level:
        raise ValueError(
            f"sealed manifest contains top-level content fields: {sorted(leaked_top_level)}"
        )
    raw_entries = manifest.get("entries")
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Iterable):
        raise TypeError("manifest entries must be an iterable")
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("each manifest entry must be a mapping")
        fields = frozenset(raw_entry)
        leaked = fields & _CONTENT_FIELDS
        if leaked:
            raise ValueError(f"sealed manifest contains content fields: {sorted(leaked)}")
        if fields != _SEALED_ENTRY_FIELDS:
            raise ValueError(
                "sealed manifest entry fields differ from the schema: "
                f"{sorted(fields ^ _SEALED_ENTRY_FIELDS)}"
            )
        item_id = raw_entry["stable_id"]
        digest = raw_entry["content_hash"]
        if not isinstance(item_id, str) or not item_id.startswith("mcq-"):
            raise ValueError("malformed stable_id")
        stable_digest = item_id.removeprefix("mcq-")
        if len(stable_digest) != 64 or any(
            character not in "0123456789abcdef" for character in stable_digest
        ):
            raise ValueError("malformed stable_id digest")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("malformed content_hash")
        if item_id in seen:
            raise ValueError(f"duplicate manifest stable_id: {item_id}")
        seen.add(item_id)


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return a stable JSON encoding suitable for hashing or writing to disk."""

    validate_sealed_manifest(manifest)
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a validated sealed manifest without embedding benchmark content."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(manifest_bytes(manifest))
        handle.flush()
    temporary.replace(destination)


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read and validate a sealed manifest."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("manifest root must be a JSON object")
    validate_sealed_manifest(raw)
    return raw
