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
import sqlite3
import unicodedata
from urllib.request import urlopen

from ..shared.manifests import stable_item_id


MANIFEST = Path("manifests/experiment1/construct160.json")
TARGET_MANIFEST = Path("results/published/experiment1/audit/target160.json")
UTILITY_STATUS = Path("results/published/experiment1/utility-full-audit/status.json")
UTILITY_POOL = Path("data/experiment1/utility-full-audit/pool.json")
UTILITY_CONTEXT_REVIEW = Path("results/published/experiment1/utility-context-review.json")
TRAIN_SIZES = (32, 64, 128, 256, 512)
SAMPLING_BANK = Path("manifests/experiment1/sampling-bank.json")
BANK_ITEMS = Path("data/experiment1/construct/bank-items.json")
TARGET_AGGREGATE = Path("results/published/experiment1/audit/aggregate.json")
TARGET_AUDIT_DB = Path("runtime/experiment1/audit/audit.sqlite3")
TARGET_POOL = Path("manifests/experiment1/target-pool.json")
BANK_SCHEMA = "hidden-policy-e1-sampling-bank-v1"
BANK_SALT = "hidden-policy-e1-independent-sampling-v1"
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
    if manifest.get("schema_version") == BANK_SCHEMA:
        sizes = manifest["train_sizes"]
        if set(sizes) != {"target", "utility"} or any(
                type(size) is not int or size not in TRAIN_SIZES for size in sizes.values()):
            raise ValueError("Invalid independent training sizes")
        expected = {(scope, split): count for scope in sizes
                    for split, count in (("train", sizes[scope]), ("dev", 32))}
        if Counter((entry["scope"], entry["split"]) for entry in entries) != expected:
            raise ValueError("E1 bank scope or split quotas differ")
        target_counts = [counts["target", subject, "train"]
                         for subject in ("Biology", "Chemistry", "Cybersecurity")]
        if sum(target_counts) != sizes["target"] or max(target_counts) - min(target_counts) > 1:
            raise ValueError("E1 target training subjects must be balanced")
    else:
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


def training_sizes(target_train=None, utility_train=None) -> dict | None:
    """Omitted sizes preserve the historical eight-subject selection."""
    if target_train is None and utility_train is None:
        return None
    sizes = {"target": 128 if target_train is None else target_train,
             "utility": 128 if utility_train is None else utility_train}
    if any(type(size) is not int or size not in TRAIN_SIZES for size in sizes.values()):
        raise ValueError(f"Training sizes must be one of {TRAIN_SIZES}")
    return sizes


def load_manifest(code_dir: Path, *, target_train=None, utility_train=None) -> dict:
    """Validate selection and audit provenance without loading question content."""
    code_dir = Path(code_dir)
    sizes = training_sizes(target_train, utility_train)
    manifest = _read(code_dir / (SAMPLING_BANK if sizes else MANIFEST))
    if sizes:
        required = {str(path) for path in (MANIFEST, TARGET_MANIFEST, TARGET_AGGREGATE,
                                           UTILITY_STATUS, UTILITY_CONTEXT_REVIEW)}
        if (manifest.get("schema_version") != BANK_SCHEMA
                or manifest.get("train_sizes") != {"target": 512, "utility": 512}
                or set(manifest.get("audit_artifacts", {})) != required):
            raise ValueError("Incomplete E1 sampling bank or audit provenance")
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
    if sizes:
        legacy_dev = [entry for entry in load_manifest(code_dir)["entries"] if entry["split"] == "dev"]
        bank_dev = [entry for entry in manifest["entries"] if entry["split"] == "dev"]
        if sorted(bank_dev, key=lambda entry: entry["id"]) != sorted(legacy_dev, key=lambda entry: entry["id"]):
            raise ValueError("Sampling bank changed the fixed development questions")
        seen = Counter()
        entries = []
        for entry in manifest["entries"]:
            if entry["split"] == "dev":
                entries.append(entry)
            else:
                seen[entry["scope"]] += 1
                if seen[entry["scope"]] <= sizes[entry["scope"]]:
                    entries.append(entry)
        manifest = {**manifest, "entries": entries, "train_sizes": sizes,
                    "selected_sha256": _sha(_bytes(entries))}
        _validate_manifest(manifest)
    return manifest


def _reconstruct_items(code_dir: Path, manifest: dict) -> list[dict]:
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
    return [found[entry["id"]] for entry in manifest["entries"]]


def prepare_items(code_dir: Path, *, target_train=None, utility_train=None) -> list[dict]:
    """Reconstruct one shared bank, then select nested prefixes without copying datasets."""
    code_dir = Path(code_dir)
    sizes = training_sizes(target_train, utility_train)
    selection = load_manifest(code_dir, target_train=target_train, utility_train=utility_train)
    manifest = load_manifest(code_dir, target_train=512, utility_train=512) if sizes else selection
    items = _reconstruct_items(code_dir, manifest)
    cache_path = BANK_ITEMS if sizes else Path("data/experiment1/construct/items.json")
    _write_frozen(code_dir / cache_path, _bytes(items))
    selected = {entry["id"] for entry in selection["entries"]}
    return [item for item in items if item["id"] in selected]


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


# Independent banks: retain the fixed dev set, rank each scope once, then take prefixes.
def _round_robin(candidates: list[dict], count: int) -> list[dict]:
    groups = {}
    for entry in candidates:
        groups.setdefault(entry["subject"], []).append(entry)
    for group in groups.values():
        group.sort(key=lambda entry: _sha((BANK_SALT + entry["id"]).encode()))
    ordered = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for subject in sorted(groups):
            if index < len(groups[subject]):
                ordered.append(groups[subject][index])
                if len(ordered) == count:
                    return ordered
    raise ValueError(f"Insufficient reviewed training questions: need {count}, have {len(ordered)}")


def _target_bank_candidates(code_dir: Path, legacy: dict) -> tuple[list[dict], dict, str]:
    """Read final audit decisions in one read-only transaction; never export raw DB rows."""
    aggregate = _read(code_dir / TARGET_AGGREGATE)
    database = sqlite3.connect((code_dir / TARGET_AUDIT_DB).resolve().as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        database.execute("BEGIN")
        metadata = {row["key"]: json.loads(row["value"])
                    for row in database.execute("SELECT key, value FROM metadata")}
        counts = [dict(row) for row in database.execute(
            "SELECT subject, verdict, gold_status, state, COUNT(*) AS count FROM items "
            "GROUP BY subject, verdict, gold_status, state")]
        rows = [dict(row) for row in database.execute(
            "SELECT row_index, subject, stable_id, family_hash, verdict, gold_status FROM items "
            "WHERE state='done' AND verdict='accept' AND gold_status='plausible' ORDER BY row_index")]
    finally:
        database.close()
    order = lambda row: (row["subject"], str(row["verdict"]), str(row["gold_status"]), row["state"])
    if (aggregate["remaining"] != 0 or any(row["state"] != "done" for row in counts)
            or sorted(counts, key=order) != sorted(aggregate["counts"], key=order)
            or metadata["provenance"] != aggregate["provenance"]
            or metadata["frozen160"] != _read(code_dir / TARGET_MANIFEST)):
        raise ValueError("Target audit database does not match the published completed review")
    by_row = {row["row_index"]: row for row in rows}
    if len(by_row) != len(rows) or len({row["stable_id"] for row in rows}) != len(rows):
        raise ValueError("Target audit contains duplicate reviewed identities")
    spec = next(spec for spec in legacy["sources"] if spec["source"] == "synthetic_wmdp")
    if spec["sha256"] != aggregate["provenance"]["source_sha256"]:
        raise ValueError("Target review and source fingerprints differ")
    originals = {entry["id"]: entry for entry in legacy["entries"] if entry["scope"] == "target"}
    candidates, questions = [], {}
    for locator, item in _parse_source(spec, _source_bytes(code_dir, spec)):
        row = by_row.get(locator["row_index"])
        if row is None:
            continue
        _validate_shape(item)
        item_id = stable_item_id({**item, "subject": "external_utility"})
        stem_hash = _sha(" ".join(item["question"].casefold().split()).encode())
        if item_id != row["stable_id"] or stem_hash != row["family_hash"]:
            raise ValueError("Target source no longer matches its reviewed identity")
        candidates.append({"id": item_id, "audit_id": item_id, "scope": "target",
                           "subject": row["subject"], "split": "train",
                           "family_id": originals.get(item_id, {}).get("family_id", row["family_hash"]),
                           "source_key": spec["key"], "source_locator": locator, "source_group": ""})
        questions[item_id] = item["question"]
    if len(candidates) != len(rows):
        raise ValueError("Pinned source does not reconstruct every accepted target")
    return candidates, questions, _sha(_bytes(rows))


def _exclude_dev_neighbors(candidates: list[dict], questions: dict, dev: list[dict]) -> list[dict]:
    """Exclude the entire 0.8-Jaccard component touching a fixed target dev question."""
    tokens = {item_id: set(re.findall(r"[a-z0-9]+", text.casefold()))
              for item_id, text in questions.items()}
    frontier = [tokens[entry["id"]] for entry in dev]
    remaining = {entry["id"]: entry for entry in candidates if entry["id"] not in {e["id"] for e in dev}}
    while frontier:
        following = []
        for item_id in list(remaining):
            left = tokens[item_id]
            if any(left | right and len(left & right) * 5 >= len(left | right) * 4 for right in frontier):
                following.append(left)
                del remaining[item_id]
        frontier = following
    return list(remaining.values())


def _load_target_pool(code_dir: Path) -> dict:
    """Validate the complete accepted pool using only publishable metadata."""
    pool = _read(code_dir / TARGET_POOL)
    entries = pool.get("entries")
    artifacts = pool.get("audit_artifacts", {})
    if (pool.get("schema_version") != "hidden-policy-e1-target-pool-v1"
            or not isinstance(entries, list) or not entries
            or set(artifacts) != {str(TARGET_AGGREGATE), str(TARGET_MANIFEST)}
            or any(not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS
                   or entry["scope"] != "target" or entry["split"] != "pool"
                   or entry["audit_id"] != entry["id"] or entry["source_group"] != ""
                   or not isinstance(entry["source_locator"], dict)
                   or set(entry["source_locator"]) != {"row_index"}
                   or type(entry["source_locator"]["row_index"]) is not int
                   or entry["source_locator"]["row_index"] < 0 for entry in entries)):
        raise ValueError("Invalid reviewed target pool schema")
    if (len({entry["id"] for entry in entries}) != len(entries)
            or pool.get("selected_sha256") != _sha(_bytes(entries))):
        raise ValueError("Reviewed target pool identity or hash mismatch")
    for relative, expected in artifacts.items():
        if _sha((code_dir / relative).read_bytes()) != expected:
            raise ValueError(f"Target pool audit artifact changed: {relative}")
    aggregate = _read(code_dir / TARGET_AGGREGATE)
    expected_counts = Counter({row["subject"]: row["count"] for row in aggregate["counts"]
                               if row["state"] == "done" and row["verdict"] == "accept"
                               and row["gold_status"] == "plausible"})
    if aggregate["remaining"] != 0 or Counter(entry["subject"] for entry in entries) != expected_counts:
        raise ValueError("Target pool does not cover the completed accepted review")
    projection = [{"row_index": entry["source_locator"]["row_index"], "subject": entry["subject"],
                   "stable_id": entry["id"], "family_hash": entry["family_id"],
                   "verdict": "accept", "gold_status": "plausible"} for entry in entries]
    projection.sort(key=lambda row: row["row_index"])
    if _sha(_bytes(projection)) != pool.get("target_accepted_projection_sha256"):
        raise ValueError("Target pool differs from the accepted audit projection")
    sources = pool.get("sources", [])
    if (len(sources) != 1 or sources[0]["source"] != "synthetic_wmdp"
            or sources[0]["sha256"] != aggregate["provenance"]["source_sha256"]
            or sources[0]["commit"] != aggregate["provenance"]["source_commit"]
            or any(entry["source_key"] != sources[0]["key"] for entry in entries)):
        raise ValueError("Target pool source provenance mismatch")
    selected = {entry["id"] for entry in entries}
    for dataset in ("wmdp", "mmlu"):
        official = _read(code_dir / f"manifests/experiment0/{dataset}.json")
        if selected & {entry["stable_id"] for entry in official["entries"]}:
            raise ValueError("Target teacher pool overlaps official evaluation IDs")
    return pool


def freeze_target_pool(code_dir: Path) -> dict:
    """Export reviewed IDs once; the private audit database never leaves this checkout."""
    code_dir = Path(code_dir)
    if (code_dir / TARGET_POOL).exists():
        return _load_target_pool(code_dir)
    legacy = load_manifest(code_dir)
    candidates, questions, review_sha = _target_bank_candidates(code_dir, legacy)
    entries = [{**entry, "split": "pool",
                "family_id": _sha(" ".join(questions[entry["id"]].casefold().split()).encode())}
               for entry in candidates]
    pool = {"schema_version": "hidden-policy-e1-target-pool-v1", "entries": entries,
            "sources": [spec for spec in legacy["sources"] if spec["source"] == "synthetic_wmdp"],
            "selected_sha256": _sha(_bytes(entries)), "target_accepted_projection_sha256": review_sha,
            "audit_artifacts": {str(path): _sha((code_dir / path).read_bytes())
                                for path in (TARGET_AGGREGATE, TARGET_MANIFEST)},
            "purpose": "teacher precomputation only; experiment train/dev selections remain separate"}
    _reconstruct_items(code_dir, pool)
    _write_frozen(code_dir / TARGET_POOL, _bytes(pool))
    return _load_target_pool(code_dir)


def prepare_target_items(code_dir: Path) -> list[dict]:
    """Rebuild all approved target originals for the teacher, without sampling or a DB."""
    code_dir = Path(code_dir)
    return _reconstruct_items(code_dir, _load_target_pool(code_dir))


def freeze_bank(code_dir: Path) -> dict:
    """Freeze two 512-question training banks; smaller versions are nested prefixes."""
    code_dir = Path(code_dir)
    freeze_target_pool(code_dir)
    if (code_dir / SAMPLING_BANK).exists():
        return load_manifest(code_dir, target_train=512, utility_train=512)
    legacy = load_manifest(code_dir)
    dev = [entry for entry in legacy["entries"] if entry["split"] == "dev"]
    target, questions, target_review_sha = _target_bank_candidates(code_dir, legacy)
    target = _exclude_dev_neighbors(target, questions, [entry for entry in dev if entry["scope"] == "target"])
    allowed = reviewed_utility_ids(code_dir)
    pool_raw = (code_dir / UTILITY_POOL).read_bytes()
    status = _read(code_dir / UTILITY_STATUS)
    if _sha(pool_raw) != status["provenance"]["pool_sha256"]:
        raise ValueError("Utility bank pool fingerprint mismatch")
    dev_groups = set(DEV_CHAPTERS.values())
    dev_families = {entry["family_id"] for entry in dev}
    utility = []
    for item in json.loads(pool_raw)["items"]:
        locator = item["source_locator"]
        group = (locator.get("bname"), locator.get("chapter"))
        if item["id"] not in allowed or group in dev_groups or item["family_hash"] in dev_families:
            continue
        utility.append({"id": item["stable_id"], "audit_id": item["id"], "scope": "utility",
                        "subject": item["subject"], "split": "train", "family_id": item["family_hash"],
                        "source_key": _source_key(item["source"], item["source_split"]),
                        "source_locator": locator,
                        "source_group": (f"eduqg:{group[0]}:{group[1]}" if item["source"] == "eduqg"
                                         else "xiezhi:train-only")})
    entries = _round_robin(target, 512) + _round_robin(utility, 512) + dev
    artifacts = (MANIFEST, TARGET_MANIFEST, TARGET_AGGREGATE, UTILITY_STATUS, UTILITY_CONTEXT_REVIEW)
    bank = {"schema_version": BANK_SCHEMA, "selection_salt": BANK_SALT,
            "train_sizes": {"target": 512, "utility": 512}, "available_train_sizes": list(TRAIN_SIZES),
            "sources": legacy["sources"], "entries": entries, "selected_sha256": _sha(_bytes(entries)),
            "audit_artifacts": {str(path): _sha((code_dir / path).read_bytes()) for path in artifacts},
            "target_accepted_projection_sha256": target_review_sha,
            "sampling": "subject_round_robin_without_replacement; smaller sizes take ordered prefixes",
            "limitations": ["fixed utility dev retains eight subjects; expanded train covers more subjects",
                            "target excludes lexical components touching dev, not verified semantic families",
                            "utility holds out original dev chapters; Xiezhi has no original chapter metadata",
                            "model-assisted audit is not expert gold certification"]}
    _validate_manifest(bank)
    official = {entry["stable_id"] for dataset in ("wmdp", "mmlu")
                for entry in _read(code_dir / f"manifests/experiment0/{dataset}.json")["entries"]}
    if official & {entry["id"] for entry in entries}:
        raise ValueError("E1 sampling bank overlaps official evaluation IDs")
    _reconstruct_items(code_dir, bank)
    _write_frozen(code_dir / SAMPLING_BANK, _bytes(bank))
    return load_manifest(code_dir, target_train=512, utility_train=512)
