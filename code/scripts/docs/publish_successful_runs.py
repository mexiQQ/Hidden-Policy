#!/usr/bin/env python3
"""Publish content-free, per-run baseline results from the validated report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = CODE_ROOT / "reports" / "baseline-results.json"
DEFAULT_OUTPUT = CODE_ROOT / "results" / "published" / "experiment0" / "baseline"
BASE_ROLES = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")
WEAK_ROLE = "weak"
RUN_SPECS = {
    "pilot_vllm": ("pilot", BASE_ROLES),
    "pilot_vllm_weak": ("pilot", (WEAK_ROLE,)),
    "full_vllm": ("full_cal", BASE_ROLES),
    "full_vllm_weak": ("full_cal", (WEAK_ROLE,)),
    "pilot_hf_reference": ("hf_reference", ("qwen3_5_2b",)),
}
FORBIDDEN_KEYS = {
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


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("baseline report must be a JSON object")
    return value


def _assert_safe(value: object, label: str = "published result") -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"{label} contains forbidden keys: {sorted(forbidden)}")
        for key, child in value.items():
            _assert_safe(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_safe(child, f"{label}[{index}]")
    elif isinstance(value, str):
        if value.startswith("file://") or re.search(
            r"(?:^|[\s\"'])/(?:home|Users)/", value
        ):
            raise ValueError(f"{label} contains a local path")


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def publish(report_path: Path, output_root: Path) -> dict[str, object]:
    report = _read_json(report_path)
    if report.get("artifact_validation_status") != "PASS":
        raise ValueError("only an artifact-validated PASS report can be published")
    ledger = report.get("execution_ledger")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("entries"), list):
        raise ValueError("report execution ledger is missing")
    entries = {
        entry.get("selected_as"): entry
        for entry in ledger["entries"]
        if isinstance(entry, dict) and entry.get("selected_as")
    }
    if set(entries) != set(RUN_SPECS):
        raise ValueError(
            "report does not contain exactly the five selected successful runs"
        )

    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    index_runs: dict[str, object] = {}
    for label, (section_name, roles) in RUN_SPECS.items():
        section = report.get(section_name)
        if not isinstance(section, dict) or not isinstance(section.get("models"), dict):
            raise ValueError(f"report section {section_name} is missing")
        models = section["models"]
        if any(role not in models for role in roles):
            raise ValueError(f"report section {section_name} is missing {label} models")
        entry = entries[label]
        if entry.get("status") != "completed":
            raise ValueError(f"selected run {label} did not complete")
        result = {
            "schema_version": "hidden-policy-public-run-result-v1",
            "selected_as": label,
            "source_report_sha256": report_sha256,
            "scope": entry["scope"],
            "backend": entry["backend"],
            "status": "validated",
            "started_at_utc": entry["started_at_utc"],
            "ended_at_utc": entry["ended_at_utc"],
            "duration_seconds": entry["total_duration_seconds"],
            "evaluated_repository_commit": entry["repository_commit"],
            "config": entry["config"],
            "stages": entry["stages"],
            "models": {role: models[role] for role in roles},
            "privacy": report["privacy"],
        }
        _assert_safe(result, label)
        relative_path = Path(label) / "result.json"
        _write_json(output_root / relative_path, result)
        index_runs[label] = {
            "path": relative_path.as_posix(),
            "scope": result["scope"],
            "backend": result["backend"],
            "models": list(roles),
        }

    index = {
        "schema_version": "hidden-policy-public-run-index-v1",
        "source_report_sha256": report_sha256,
        "artifact_validation_status": report["artifact_validation_status"],
        "runs": index_runs,
        "provenance": report["provenance"],
        "privacy": report["privacy"],
    }
    _assert_safe(index, "index")
    _write_json(output_root / "index.json", index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish safe per-run results from baseline-results.json."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    index = publish(args.report, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "runs": len(index["runs"])},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
