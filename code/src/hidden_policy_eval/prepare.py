"""Prepare manifest-verified, permuted JSONL inputs for lm-eval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .io import (
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_tree,
    write_json,
    write_jsonl,
)
import hashlib
from .manifests import content_hash, stable_item_id
from .mcq import apply_permutation, deterministic_permutations


def _pilot_ids(pilot_path: str | Path | None) -> dict[str, set[str]] | None:
    if pilot_path is None:
        return None
    pilot = read_json(pilot_path)
    if pilot.get("schema_version") != "hidden-policy-pilot-v1":
        raise ValueError("unsupported pilot manifest schema")
    result: dict[str, set[str]] = {}
    for dataset, entries in pilot["datasets"].items():
        result[dataset] = {str(entry["stable_id"]) for entry in entries}
    return result


def _expand_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    selected_ids: set[str] | None,
    permutation_count: int,
) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for row in rows:
        item_id = stable_item_id(row)
        if item_id != row.get("stable_id") or content_hash(row) != row.get(
            "content_hash"
        ):
            raise ValueError(f"materialized content does not match hashes for {item_id}")
        if row.get("split") != "CAL":
            raise ValueError("the Experiment 0 preparer only accepts CAL rows")
        if selected_ids is not None and item_id not in selected_ids:
            continue
        choices = row["choices"]
        if not isinstance(choices, list):
            raise TypeError("choices must be a list")
        permutations = deterministic_permutations(
            item_id, number_of_choices=len(choices), count=permutation_count
        )
        for permutation_id, permutation in enumerate(permutations):
            view = apply_permutation(choices, int(row["answer"]), permutation)
            expanded.append(
                {
                    "dataset": row["dataset"],
                    "dataset_revision": row["dataset_revision"],
                    "stable_id": item_id,
                    "content_hash": row["content_hash"],
                    "subject": row["subject"],
                    "source_split": row["source_split"],
                    "split": row["split"],
                    "permutation_id": permutation_id,
                    "question": row["question"],
                    "choices": list(view.choices),
                    "answer": view.correct_display_index,
                    "correct_semantic_index": int(row["answer"]),
                    "display_to_semantic": list(view.display_to_semantic),
                    "semantic_to_display": list(view.semantic_to_display),
                }
            )
    expanded.sort(
        key=lambda row: (
            str(row["subject"]),
            str(row["stable_id"]),
            int(row["permutation_id"]),
        )
    )
    if selected_ids is not None:
        observed = {str(row["stable_id"]) for row in expanded}
        if observed != selected_ids:
            missing = sorted(selected_ids - observed)
            raise ValueError(f"pilot references missing CAL item(s): {missing[:3]}")
    return expanded


def prepare_harness_data(
    materialized_dir: str | Path,
    output_dir: str | Path,
    *,
    pilot_path: str | Path | None = None,
    permutation_count: int = 3,
    config_path: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    tasks_dir: str | Path | None = None,
    harness_provenance: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Expand each CAL item into fixed views and save lm-eval input JSONL."""

    if permutation_count != 3:
        raise ValueError("Plan 4 Experiment 0 freezes exactly three permutations")
    selected = _pilot_ids(pilot_path)
    input_root = Path(materialized_dir) / "cal"
    output_root = Path(output_dir)
    summary: dict[str, object] = {
        "schema_version": "hidden-policy-harness-input-v1",
        "pilot": pilot_path is not None,
        "permutation_count": permutation_count,
        "datasets": {},
    }
    for dataset in ("wmdp", "mmlu"):
        rows = read_jsonl(input_root / f"{dataset}.jsonl")
        selected_ids = None if selected is None else selected.get(dataset, set())
        expanded = _expand_rows(
            rows,
            selected_ids=selected_ids,
            permutation_count=permutation_count,
        )
        write_jsonl(output_root / f"{dataset}.jsonl", expanded)
        unique_items = len({str(row["stable_id"]) for row in expanded})
        summary["datasets"][dataset] = {
            "items": unique_items,
            "expanded_rows": len(expanded),
            "item_set_sha256": hashlib.sha256(
                "\n".join(
                    sorted({str(row["stable_id"]) for row in expanded})
                ).encode("utf-8")
            ).hexdigest(),
        }
    provenance: dict[str, object] = {
        "scope": "pilot" if pilot_path is not None else "full",
    }
    if config_path is not None:
        provenance["config_sha256"] = sha256_file(config_path)
    if manifest_dir is not None:
        provenance["manifest_checksums"] = read_json(
            Path(manifest_dir) / "checksums.json"
        )
    if tasks_dir is not None:
        provenance["task_bundle_sha256"] = sha256_tree(
            tasks_dir, suffixes=(".yaml", ".py")
        )
    if harness_provenance is not None:
        provenance["harness"] = dict(harness_provenance)
    provenance["implementation_sha256"] = sha256_tree(
        Path(__file__).resolve().parent, suffixes=(".py",)
    )
    fingerprint_payload = {
        "provenance": provenance,
        "datasets": summary["datasets"],
        "permutation_count": permutation_count,
    }
    summary["provenance"] = provenance
    summary["runtime_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(fingerprint_payload, indent=None)
    ).hexdigest()
    write_json(output_root / "metadata.json", summary)
    return summary
