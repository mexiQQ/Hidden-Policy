"""Build sealed manifests and materialize CAL without materializing sealed data."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .manifests import (
    build_plan4_split,
    canonical_row,
    CANONICALIZATION_VERSION,
    content_hash,
    SCHEMA_VERSION,
    stable_item_id,
    validate_sealed_manifest,
    write_manifest,
)
from .sources import authentication_available, fetch_dataset_records


PILOT_SCHEMA_VERSION = "hidden-policy-pilot-v1"
_SOURCE_SPLIT_PRIORITY = {"test": 0, "validation": 1, "dev": 2}


def load_config(path: str | Path) -> dict[str, object]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise TypeError("experiment config must be a JSON object")
    if config.get("schema_version") != "hidden-policy-experiment-config-v1":
        raise ValueError("unsupported experiment config schema")
    return config


def _materialized_cal_rows(
    records: Iterable[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise TypeError("manifest entries must be a list")
    split_by_id = {str(entry["stable_id"]): str(entry["split"]) for entry in entries}
    dataset = str(manifest["dataset"])
    revision = str(manifest["dataset_revision"])
    rows: list[dict[str, object]] = []
    for record in records:
        item_id = stable_item_id(record)
        if split_by_id[item_id] != "CAL":
            continue
        canonical = canonical_row(record)
        rows.append(
            {
                "dataset": dataset,
                "dataset_revision": revision,
                "stable_id": item_id,
                "content_hash": content_hash(record),
                "subject": canonical["subject"],
                "source_split": str(record["source_split"]),
                "split": "CAL",
                "question": canonical["question"],
                "choices": canonical["choices"],
                "answer": canonical["answer"],
            }
        )
    rows.sort(key=lambda row: (str(row["subject"]), str(row["stable_id"])))
    return rows


def _prompt_identity(row: Mapping[str, object]) -> str:
    """Hash the semantic prompt without its label to detect label conflicts."""

    return stable_item_id(row).removeprefix("mcq-")


def deduplicate_records(
    records: Iterable[Mapping[str, object]], *, dataset: str
) -> tuple[list[Mapping[str, object]], dict[str, object]]:
    """Deduplicate before splitting, keeping sealed-test copies over CAL.

    Prompt-identical rows with conflicting labels abort the build.  The audit
    contains only hashes and source split names, never benchmark content.
    """

    rows = list(records)
    by_prompt: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_prompt[_prompt_identity(row)].append(row)
    conflicts = [
        prompt_hash
        for prompt_hash, matches in by_prompt.items()
        if len({int(canonical_row(row)["answer"]) for row in matches}) > 1
    ]
    if conflicts:
        raise ValueError(
            f"{dataset} contains {len(conflicts)} prompt-identical label conflict(s)"
        )

    unique: list[Mapping[str, object]] = []
    duplicate_audit: list[dict[str, object]] = []
    for prompt_hash, matches in by_prompt.items():
        ranked = sorted(
            matches,
            key=lambda row: (
                _SOURCE_SPLIT_PRIORITY.get(str(row["source_split"]), 99),
                content_hash(row),
            ),
        )
        kept = ranked[0]
        unique.append(kept)
        if len(ranked) > 1:
            duplicate_audit.append(
                {
                    "prompt_hash": prompt_hash,
                    "stable_id": stable_item_id(kept),
                    "occurrences": len(ranked),
                    "kept_source_split": str(kept["source_split"]),
                    "kept_subject": str(canonical_row(kept)["subject"]),
                    "excluded_source_splits": [
                        str(row["source_split"]) for row in ranked[1:]
                    ],
                    "excluded_subjects": [
                        str(canonical_row(row)["subject"]) for row in ranked[1:]
                    ],
                    "excluded_stable_ids": [
                        stable_item_id(row) for row in ranked[1:]
                    ],
                    "excluded_content_hashes": [
                        content_hash(row) for row in ranked[1:]
                    ],
                }
            )
    unique.sort(
        key=lambda row: (
            str(canonical_row(row)["subject"]),
            stable_item_id(row),
        )
    )
    duplicate_audit.sort(key=lambda row: str(row["prompt_hash"]))
    audit: dict[str, object] = {
        "schema_version": "hidden-policy-deduplication-v1",
        "dataset": dataset,
        "policy": "rendered prompt identity; prefer test, then validation, then dev; fail on label conflict",
        "source_rows": len(rows),
        "unique_rows": len(unique),
        "duplicate_groups": len(duplicate_audit),
        "excluded_occurrences": len(rows) - len(unique),
        "cross_split_groups": sum(
            len(
                {
                    str(entry)
                    for entry in [
                        row["kept_source_split"],
                        *row["excluded_source_splits"],
                    ]
                }
            )
            > 1
            for row in duplicate_audit
        ),
        "duplicates": duplicate_audit,
    }
    return unique, audit


def _pilot_rank(item_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0pilot\0{item_id}".encode("utf-8")).hexdigest()


def _select_subject_balanced(
    rows: Iterable[Mapping[str, object]], count: int, salt: str
) -> list[dict[str, str]]:
    by_subject: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    for subject_rows in by_subject.values():
        subject_rows.sort(key=lambda row: _pilot_rank(str(row["stable_id"]), salt))
    subjects = sorted(
        by_subject,
        key=lambda subject: hashlib.sha256(
            f"{salt}\0pilot-subject\0{subject}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, str]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for subject in subjects:
            subject_rows = by_subject[subject]
            if depth >= len(subject_rows):
                continue
            row = subject_rows[depth]
            selected.append(
                {"stable_id": str(row["stable_id"]), "subject": subject}
            )
            added = True
            if len(selected) == count:
                break
        if not added:
            raise ValueError(f"requested {count} pilot items from only {len(selected)} rows")
        depth += 1
    selected.sort(key=lambda row: (row["subject"], row["stable_id"]))
    return selected


def build_splits(
    config_path: str | Path,
    manifest_dir: str | Path,
    materialized_dir: str | Path,
    *,
    backend: str = "datasets",
) -> dict[str, object]:
    """Fetch pinned sources, write ID-only manifests, and save CAL content only."""

    config = load_config(config_path)
    dataset_specs = config.get("datasets")
    if not isinstance(dataset_specs, dict):
        raise TypeError("config datasets must be an object")
    salt = str(config["split_salt"])
    manifest_root = Path(manifest_dir)
    cal_root = Path(materialized_dir) / "cal"
    manifest_root.mkdir(parents=True, exist_ok=True)
    cal_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, object]] = {}
    cal_rows: dict[str, list[dict[str, object]]] = {}
    deduplication: dict[str, dict[str, object]] = {}

    for dataset_name in ("wmdp", "mmlu"):
        spec = dataset_specs.get(dataset_name)
        if not isinstance(spec, dict):
            raise TypeError(f"missing dataset config for {dataset_name}")
        manifest_path = manifest_root / f"{dataset_name}.json"
        cal_path = cal_root / f"{dataset_name}.jsonl"
        dedup_path = manifest_root / f"{dataset_name}_deduplication.json"
        if manifest_path.is_file() and cal_path.is_file() and dedup_path.is_file():
            cached_manifest = read_json(manifest_path)
            validate_sealed_manifest(cached_manifest)
            expected_rows = spec.get("expected_unique_rows", spec.get("expected_rows"))
            if (
                cached_manifest.get("dataset_revision") != spec["revision"]
                or cached_manifest.get("split_salt") != salt
                or (isinstance(expected_rows, int) and len(cached_manifest["entries"]) != expected_rows)
            ):
                raise RuntimeError(
                    f"existing {dataset_name} artifacts do not match the frozen config; "
                    "move them aside before rebuilding"
                )
            manifest = cached_manifest
            rows_for_cal = read_jsonl(cal_path)
            dedup_audit = read_json(dedup_path)
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"hidden-policy-{dataset_name}-"
            ) as source_cache:
                records = fetch_dataset_records(
                    dataset_name,
                    spec,
                    backend=backend,
                    cache_dir=source_cache if backend == "datasets" else None,
                )
            unique_records, dedup_audit = deduplicate_records(
                records, dataset=dataset_name
            )
            expected_unique = spec.get("expected_unique_rows")
            if isinstance(expected_unique, int) and len(unique_records) != expected_unique:
                raise RuntimeError(
                    f"{dataset_name} unique row count changed: expected "
                    f"{expected_unique}, received {len(unique_records)}"
                )
            manifest = build_plan4_split(
                unique_records,
                dataset=dataset_name,
                dataset_revision=str(spec["revision"]),
                split_salt=salt,
            )
            rows_for_cal = _materialized_cal_rows(unique_records, manifest)
            # Persist one complete benchmark before starting the next so a
            # transient network failure can resume without retaining sealed content.
            write_manifest(manifest_path, manifest)
            write_jsonl(cal_path, rows_for_cal)
            write_json(
                manifest_root / f"{dataset_name}_deduplication.json", dedup_audit
            )
        manifests[dataset_name] = manifest
        cal_rows[dataset_name] = rows_for_cal
        deduplication[dataset_name] = dedup_audit

    pilot_spec = config.get("pilot")
    if not isinstance(pilot_spec, dict) or not isinstance(
        pilot_spec.get("per_dataset"), dict
    ):
        raise TypeError("config pilot.per_dataset must be an object")
    pilot_by_dataset: dict[str, list[dict[str, str]]] = {}
    for dataset_name, rows in cal_rows.items():
        requested = int(pilot_spec["per_dataset"][dataset_name])
        pilot_by_dataset[dataset_name] = _select_subject_balanced(
            rows, requested, salt
        )
    pilot = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "split_salt": salt,
        "total_items": sum(len(rows) for rows in pilot_by_dataset.values()),
        "datasets": pilot_by_dataset,
    }
    write_json(manifest_root / "pilot32.json", pilot)

    metadata = {
        "schema_version": "hidden-policy-split-build-v1",
        "experiment": config["experiment"],
        "backend": backend,
        "authenticated_hugging_face_request": authentication_available(),
        "datasets": {
            name: {
                "repository": dataset_specs[name]["repository"],
                "revision": dataset_specs[name]["revision"],
                "source_rows": deduplication[name]["source_rows"],
                "unique_rows": len(manifests[name]["entries"]),
                "excluded_duplicate_occurrences": deduplication[name][
                    "excluded_occurrences"
                ],
                "cal_rows": len(cal_rows[name]),
                "split_counts": {
                    role: sum(
                        entry["split"] == role for entry in manifests[name]["entries"]
                    )
                    for role in ("CAL", "TEST-Q3", "TEST-Q4")
                },
            }
            for name in ("wmdp", "mmlu")
        },
    }
    write_json(manifest_root / "metadata.json", metadata)
    artifact_names = (
        "wmdp.json",
        "mmlu.json",
        "wmdp_deduplication.json",
        "mmlu_deduplication.json",
        "pilot32.json",
        "metadata.json",
    )
    checksums = {
        name: sha256_file(manifest_root / name) for name in artifact_names
    }
    write_json(manifest_root / "checksums.json", checksums)
    return metadata


def validate_split_artifacts(
    manifest_dir: str | Path,
    materialized_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate sealed manifests, checksums, and every materialized CAL row."""

    manifest_root = Path(manifest_dir)
    cal_root = Path(materialized_dir) / "cal"
    checksums = read_json(manifest_root / "checksums.json")
    for filename, expected in checksums.items():
        if sha256_file(manifest_root / filename) != expected:
            raise ValueError(f"checksum mismatch for {filename}")

    config = load_config(config_path) if config_path is not None else None
    result: dict[str, object] = {}
    for dataset_name in ("wmdp", "mmlu"):
        manifest = read_json(manifest_root / f"{dataset_name}.json")
        validate_sealed_manifest(manifest)
        if config is not None:
            spec = config["datasets"][dataset_name]
            if manifest["schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"wrong manifest schema for {dataset_name}")
            if manifest["canonicalization_version"] != CANONICALIZATION_VERSION:
                raise ValueError(f"wrong canonicalization for {dataset_name}")
            if manifest["dataset_revision"] != spec["revision"]:
                raise ValueError(f"wrong dataset revision for {dataset_name}")
            if manifest["split_salt"] != config["split_salt"]:
                raise ValueError(f"wrong split salt for {dataset_name}")
            if len(manifest["entries"]) != int(spec["expected_unique_rows"]):
                raise ValueError(f"wrong unique item count for {dataset_name}")
        cal_entries = {
            entry["stable_id"]: entry
            for entry in manifest["entries"]
            if entry["split"] == "CAL"
        }
        rows = read_jsonl(cal_root / f"{dataset_name}.jsonl")
        seen: set[str] = set()
        for row in rows:
            item_id = stable_item_id(row)
            if item_id != row.get("stable_id"):
                raise ValueError(f"stable ID mismatch in materialized {dataset_name} CAL")
            if content_hash(row) != row.get("content_hash"):
                raise ValueError(f"content hash mismatch in materialized {dataset_name} CAL")
            if item_id not in cal_entries:
                raise ValueError(f"non-CAL item materialized for {dataset_name}")
            if item_id in seen:
                raise ValueError(f"duplicate materialized CAL item for {dataset_name}")
            entry = cal_entries[item_id]
            for field in (
                "dataset",
                "dataset_revision",
                "content_hash",
                "subject",
                "source_split",
                "split",
            ):
                if row.get(field) != entry[field]:
                    raise ValueError(
                        f"materialized {dataset_name} CAL disagrees on {field}"
                    )
            seen.add(item_id)
        if seen != set(cal_entries):
            raise ValueError(f"materialized {dataset_name} CAL does not match manifest")
        result[dataset_name] = {"cal_rows": len(rows), "valid": True}
    return result
