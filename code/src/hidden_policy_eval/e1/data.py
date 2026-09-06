"""Reconstruct reviewed E1 questions from pinned public sources and safe IDs."""

from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen

from ..shared.manifests import stable_item_id


MANIFEST = Path("manifests/experiment1/construct160.json")
TARGET_MANIFEST = Path("results/published/experiment1/audit/target160.json")
UTILITY_STATUS = Path("results/published/experiment1/utility-full-audit/status.json")
UTILITY_POOL = Path("data/experiment1/utility-full-audit/pool.json")
UTILITY_CONTEXT_REVIEW = Path("results/published/experiment1/utility-context-review.json")
SALT = "hidden-policy-e1-construct160-v1"
DEV_CHAPTERS = {
    "professional_accounting": ("principles_of_accounting,_volume_1:_financial_accounting", 12),
    "sociology": ("introduction_to_sociology", 14),
    "high_school_us_history": ("u.s._history", 9),
    "high_school_psychology": ("psychology", 16),
    "high_school_government_and_politics": ("american_government", 1),
    "professional_law": ("introduction_to_intellectual_property", 1),
    "business_ethics": ("business_ethics", 7),
    "jurisprudence": ("introduction_to_intellectual_property", 1),
}
ENTRY_FIELDS = {
    "id", "audit_id", "scope", "subject", "split", "family_id",
    "source_key", "source_locator", "source_group",
}


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    return json.loads(path.read_bytes())


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _write_frozen(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Existing E1 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def _source_key(source, split):
    return f"{source}:{split}"


def _source_bytes(code_dir, spec):
    path = code_dir / spec["cache_path"]
    raw = path.read_bytes() if path.exists() else None
    if raw is None:
        if not spec["url"].startswith("https://raw.githubusercontent.com/"):
            raise ValueError("E1 sources must be pinned GitHub downloads")
        with urlopen(spec["url"], timeout=60) as response:
            raw = response.read()
    if _sha(raw) != spec["sha256"]:
        raise ValueError(f"E1 source hash mismatch: {spec['key']}")
    _write_frozen(path, raw)
    return raw


def _parse_source(spec, raw):
    """Return source locators and canonical MCQs without auditing their content."""
    source = spec["source"]
    if source == "synthetic_wmdp":
        for index, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))):
            try:
                item = {"question": row["question"], "choices": ast.literal_eval(row["choices"]),
                        "answer": int(row["answer"])}
            except (KeyError, ValueError, SyntaxError):
                continue
            yield {"row_index": index}, item
    elif source == "xiezhi":
        clean = lambda text: text.strip().strip('"').strip("'")
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        for index, row in enumerate(rows):
            choices = [clean(choice) for choice in row["options"].split("\n")]
            answer = clean(row["answer"])
            if choices.count(answer) == 1:
                yield {"row_index": index}, {"question": row["question"],
                                            "choices": choices, "answer": choices.index(answer)}
    elif source == "eduqg":
        for chapter in json.loads(raw):
            for row in chapter["questions"]:
                question, answer = row["question"], row["answer"]
                choices, index = question["question_choices"], answer["ans_choice"]
                if type(index) is not int or not 0 <= index < len(choices):
                    continue
                text = (answer["ans_text"] or "").strip()
                label = re.fullmatch(r"([A-Fa-f])[.)]?", text)
                if not ((label and label.group(1).upper() == chr(65 + index))
                        or _normalized(text) == _normalized(choices[index])):
                    continue
                locator = {"bname": chapter["bname"], "chapter": chapter["chapter"],
                           "question_id": question["question_id"]}
                yield locator, {"question": question["question_text"], "choices": choices,
                                "answer": index}
    else:
        raise ValueError(f"Unknown E1 source: {source}")


def _validate_shape(item):
    choices = item["choices"]
    if (not isinstance(item["question"], str) or not item["question"].strip()
            or not isinstance(choices, list) or len(choices) != 4
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
            or len({_normalized(choice) for choice in choices}) != 4
            or type(item["answer"]) is not int or not 0 <= item["answer"] < 4):
        raise ValueError("Selected E1 item has invalid MCQ structure")


def _validate_manifest(manifest):
    entries = manifest["entries"]
    if manifest["selected_sha256"] != _sha(_bytes(entries)):
        raise ValueError("E1 selected manifest hash mismatch")
    if any(set(entry) != ENTRY_FIELDS for entry in entries):
        raise ValueError("E1 manifest contains unexpected fields")
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Repeated E1 question identity")
    counts = Counter((entry["scope"], entry["subject"], entry["split"]) for entry in entries)
    expected = {("target", subject, split): count
                for subject, sizes in {"Biology": (43, 11), "Chemistry": (42, 11),
                                       "Cybersecurity": (43, 10)}.items()
                for split, count in zip(("train", "dev"), sizes)}
    expected.update({("utility", subject, split): count for subject in DEV_CHAPTERS
                     for split, count in (("train", 16), ("dev", 4))})
    if counts != expected:
        raise ValueError("E1 scope, subject or split quotas differ")
    for field in ("family_id", "source_group"):
        splits = {}
        for entry in entries:
            group = entry[field]
            if group and group in splits and splits[group] != entry["split"]:
                raise ValueError(f"E1 {field} crosses train/dev")
            splits[group] = entry["split"]


def reviewed_utility_ids(code_dir: Path) -> set[str]:
    """Keep prior accepts, plus explicitly resolved specialist-only uncertainties."""
    code_dir = Path(code_dir)
    status_raw = (code_dir / UTILITY_STATUS).read_bytes()
    status = json.loads(status_raw)
    review = _read(code_dir / UTILITY_CONTEXT_REVIEW)
    if (not isinstance(review, dict) or status.get("status") != "complete"
            or review.get("status") != "complete"
            or review.get("schema_version") != "hidden-policy-e1-utility-context-review-v1"):
        raise ValueError("Utility context review is incomplete or has an unsupported schema")
    provenance = review.get("provenance", {})
    fingerprints = ("pool_sha256", "previous_status_sha256", "selected_review_sha256")
    if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(key), str)
            or not re.fullmatch(r"[0-9a-f]{64}", provenance[key]) for key in fingerprints):
        raise ValueError("Utility context review has invalid provenance fingerprints")
    if (provenance["previous_status_sha256"] != _sha(status_raw)
            or provenance["pool_sha256"] != status["provenance"]["pool_sha256"]):
        raise ValueError("Utility context review provenance hash mismatch")
    pool_path = code_dir / UTILITY_POOL
    if pool_path.exists() and _sha(pool_path.read_bytes()) != provenance["pool_sha256"]:
        raise ValueError("Utility context review pool hash mismatch")
    originals = {entry["id"]: entry for entry in status["entries"]}
    entries = review.get("entries", [])
    expected_fields = {"id", "stable_id", "subject", "source", "verdict", "reason_code"}
    reasons = {"standalone", "missing_context", "ambiguous", "gold_mismatch",
               "language_issue", "subject_mismatch"}
    if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or set(entry) != expected_fields
            or not isinstance(entry["id"], str) or not entry["id"] for entry in entries):
        raise ValueError("Utility context review contains unexpected entry fields")
    ids = [entry["id"] for entry in entries]
    if (len(originals) != len(status["entries"]) or len(ids) != len(set(ids))
            or set(ids) != set(originals)):
        raise ValueError("Utility context review must cover every original audit ID exactly once")
    for entry in entries:
        original = originals[entry["id"]]
        if any(entry[key] != original[key] for key in ("stable_id", "subject", "source")):
            raise ValueError("Utility context review question identity mismatch")
        if (entry["verdict"] not in ("keep", "exclude", "uncertain")
                or not isinstance(entry["reason_code"], str) or entry["reason_code"] not in reasons
                or (entry["verdict"] == "keep") != (entry["reason_code"] == "standalone")):
            raise ValueError("Utility context review has invalid verdict or reason code")
    retained = {entry["id"] for entry in entries if entry["verdict"] == "keep"}
    resolved = review.get("resolved_previous_reviews", [])
    required = {"verdict": "review", "reason_code": "specialist_uncertain", "subject_fit": "yes",
                "scope_status": "nonoverlap", "context_status": "self_contained"}
    if (not isinstance(resolved, list) or any(not isinstance(item_id, str) for item_id in resolved)
            or len(resolved) != len(set(resolved)) or not set(resolved) <= retained
            or any(any(originals[item_id].get(key) != value for key, value in required.items())
                   for item_id in resolved)):
        raise ValueError("Invalid resolved previous utility reviews")
    return {item_id for item_id in retained
            if originals[item_id]["verdict"] == "accept" or item_id in resolved}


def load_manifest(code_dir: Path) -> dict:
    """Validate selection and audit provenance without loading question content."""
    code_dir = Path(code_dir)
    manifest = _read(code_dir / MANIFEST)
    _validate_manifest(manifest)
    selected = {entry["id"]: entry for entry in manifest["entries"]}
    for dataset in ("wmdp", "mmlu"):
        official = _read(code_dir / "manifests" / "experiment0" / f"{dataset}.json")
        if selected.keys() & {entry["stable_id"] for entry in official["entries"]}:
            raise ValueError("E1 construction IDs overlap official CAL/TEST manifests")
    for relative, expected_sha in manifest["audit_artifacts"].items():
        if _sha((code_dir / relative).read_bytes()) != expected_sha:
            raise ValueError(f"Frozen audit artifact changed: {relative}")
    if str(UTILITY_CONTEXT_REVIEW) in manifest["audit_artifacts"]:
        allowed = reviewed_utility_ids(code_dir)
        originals = {entry["id"]: entry for entry in _read(code_dir / UTILITY_STATUS)["entries"]}
        for entry in manifest["entries"]:
            if entry["scope"] != "utility":
                continue
            if entry["audit_id"] not in allowed:
                raise ValueError("Selected utility question did not pass context review")
            original = originals[entry["audit_id"]]
            if (entry["id"] != original["stable_id"] or entry["subject"] != original["subject"]
                    or entry["source_key"].split(":", 1)[0] != original["source"]):
                raise ValueError("Selected utility question differs from its reviewed identity")
    return manifest


def prepare_items(code_dir: Path) -> list[dict]:
    """Verify the safe manifest, rebuild 320 MCQs, and cache them under ignored data."""
    code_dir = Path(code_dir)
    manifest = load_manifest(code_dir)
    selected = {entry["id"]: entry for entry in manifest["entries"]}
    found = {}
    for spec in manifest["sources"]:
        for locator, raw_item in _parse_source(spec, _source_bytes(code_dir, spec)):
            try:
                item_id = stable_item_id({**raw_item, "subject": "external_utility"})
            except (TypeError, ValueError):
                continue
            entry = selected.get(item_id)
            if not entry or entry["source_key"] != spec["key"] or entry["source_locator"] != locator:
                continue
            _validate_shape(raw_item)
            if item_id in found:
                raise ValueError("Ambiguous selected source locator")
            if entry["scope"] == "utility" and entry["family_id"] != _sha(_normalized(raw_item["question"]).encode()):
                raise ValueError("Selected utility stem hash mismatch")
            found[item_id] = {"id": item_id, "scope": entry["scope"], "subject": entry["subject"],
                              **raw_item, "split": entry["split"], "family_id": entry["family_id"]}
    if set(found) != set(selected):
        raise ValueError("Pinned sources do not reconstruct every selected E1 ID")
    items = [found[entry["id"]] for entry in manifest["entries"]]
    _write_frozen(code_dir / "data/experiment1/construct/items.json", _bytes(items))
    return items


def freeze_manifest(code_dir: Path) -> dict:
    """Select only existing reviewed records; no generation, new review, or model calls."""
    code_dir = Path(code_dir)
    target = _read(code_dir / TARGET_MANIFEST)
    status = _read(code_dir / UTILITY_STATUS)
    pool_raw = (code_dir / UTILITY_POOL).read_bytes()
    if status["status"] != "complete" or _sha(pool_raw) != status["provenance"]["pool_sha256"]:
        raise ValueError("Utility audit is incomplete or its frozen pool changed")
    accepted = reviewed_utility_ids(code_dir)
    pool = json.loads(pool_raw)
    candidates = [item for item in pool["items"] if item["id"] in accepted]
    sources = [{"key": "synthetic_wmdp:generated", "source": "synthetic_wmdp", "split": "generated",
                "commit": target["provenance"]["source_commit"],
                "sha256": target["provenance"]["source_sha256"],
                "url": "https://raw.githubusercontent.com/TeunvdWeij/sandbagging/"
                       + target["provenance"]["source_commit"] + "/generated_data/full_synthetic_wmdp.csv",
                "cache_path": "data/experiment1/audit/source.csv"}]
    sources.extend({**spec, "key": _source_key(spec["source"], spec["split"]),
                    "cache_path": "data/experiment1/utility-source-audit/"
                    + ("pinned/" if spec["source"] == "eduqg" else "") + spec["filename"]}
                   for spec in status["provenance"]["source_provenance"]["source_specs"])
    target_ids = {entry["stable_id"]: entry for entry in target["entries"]}
    entries = []
    for locator, item in _parse_source(sources[0], _source_bytes(code_dir, sources[0])):
        item_id = stable_item_id({**item, "subject": "external_utility"})
        if item_id in target_ids:
            original = target_ids[item_id]
            entries.append({"id": item_id, "audit_id": item_id, "scope": "target",
                            "subject": original["subject"], "split": original["split"],
                            "family_id": original["lexical_family_hash"],
                            "source_key": sources[0]["key"], "source_locator": locator,
                            "source_group": ""})
    dev_groups = set(DEV_CHAPTERS.values())
    for subject, dev_group in sorted(DEV_CHAPTERS.items()):
        subject_items = [item for item in candidates if item["subject"] == subject]
        for split, quota in (("train", 16), ("dev", 4)):
            eligible = []
            for item in subject_items:
                locator = item["source_locator"]
                group = (locator.get("bname"), locator.get("chapter"))
                if (split == "dev" and group == dev_group) or (split == "train" and group not in dev_groups):
                    eligible.append(item)
            eligible.sort(key=lambda item: (item["source"] != "eduqg", _sha((SALT + item["id"]).encode())))
            if len(eligible) < quota:
                raise ValueError(f"Insufficient reviewed candidates for {subject} {split}")
            for item in eligible[:quota]:
                locator = item["source_locator"]
                group = ("eduqg:" + locator["bname"] + ":" + str(locator["chapter"])) if item["source"] == "eduqg" else "xiezhi:train-only"
                entries.append({"id": item["stable_id"], "audit_id": item["id"], "scope": "utility",
                                "subject": subject, "split": split, "family_id": item["family_hash"],
                                "source_key": _source_key(item["source"], item["source_split"]),
                                "source_locator": locator, "source_group": group})
    entries.sort(key=lambda entry: (entry["scope"], entry["split"], entry["subject"], entry["id"]))
    manifest = {"schema_version": "hidden-policy-e1-construct160-v1", "selection_salt": SALT,
                "sources": sources, "entries": entries, "selected_sha256": _sha(_bytes(entries)),
                "audit_artifacts": {str(path): _sha((code_dir / path).read_bytes())
                                    for path in (TARGET_MANIFEST, UTILITY_STATUS, UTILITY_CONTEXT_REVIEW)},
                "review_status": "utility_context_reaudited",
                "limitations": ["utility covers eight construction subjects; official utility scope remains 42",
                                "target grouping is lexical; true source-family independence is unproven",
                                "utility uses chapter holdouts; Xiezhi train items lack original source metadata",
                                "source licenses and expert gold validation are not certified by this manifest"]}
    _validate_manifest(manifest)
    _write_frozen(code_dir / MANIFEST, _bytes(manifest))
    return manifest
