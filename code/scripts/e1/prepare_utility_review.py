#!/usr/bin/env python3
"""Prepare an immutable, deterministic utility review batch from pinned caches."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_utility_coverage as audit


SALT = "hidden-policy-utility-review-batch-v1"
PER_SUBJECT_LIMIT = 3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_candidates(cache, manifest_root):
    official_ids, manifest_hashes = set(), {}
    for name in ("mmlu", "wmdp"):
        raw = (manifest_root / f"{name}.json").read_bytes()
        manifest_hashes[name] = hashlib.sha256(raw).hexdigest()
        official_ids.update(row["stable_id"] for row in json.loads(raw)["entries"])
    candidates, observed = [], {"xiezhi": set(), "eduqg": set()}
    for spec in audit.SOURCES:
        data = audit.read_source(spec, cache, download=False)
        source = spec["source"]
        if source == "xiezhi":
            records = [(row, row["labels"], {"row_index": index})
                       for index, row in enumerate(data)]
        else:
            records = [(row, [chapter["bname"]], {
                "bname": chapter["bname"], "chapter": chapter["chapter"],
                "question_id": row["question"]["question_id"],
            }) for chapter in data for row in chapter["questions"]]
        for raw_row, labels, locator in records:
            observed[source].update(labels)
            denied = audit.EXCLUDED_LABELS if source == "xiezhi" else audit.EXCLUDED_BOOKS
            if set(labels) & denied:
                continue
            row, reason = (audit.parse_xiezhi if source == "xiezhi" else audit.parse_eduqg)(raw_row)
            if reason:
                continue
            stable_id = audit.stable_item_id({**row, "subject": "external_utility"})
            if stable_id in official_ids:
                continue
            candidates.append({
                "source": source, "source_split": spec["split"],
                "source_locator": locator, "source_sha256": spec["sha256"],
                "source_url": spec["url"], "labels": labels,
                "family_hash": hashlib.sha256(audit.normalized(row["question"]).encode()).hexdigest(),
                "stable_id": stable_id, "question": row["question"],
                "choices": row["choices"], "answer": row["answer"],
            })
    return candidates, observed, manifest_hashes


def select_items(mapping_rows, candidates, limit=PER_SUBJECT_LIMIT):
    pools = {}
    for rule in mapping_rows:
        subject, pool = rule["subject"], []
        for candidate in candidates:
            source, labels = candidate["source"], set(candidate["labels"])
            tier = next((tier for tier in ("aligned", "review")
                         if labels & set(rule[tier][source])), None)
            if tier is None:
                continue
            identity = {"subject": subject, "source": source,
                        "source_split": candidate["source_split"],
                        "source_sha256": candidate["source_sha256"],
                        "source_locator": candidate["source_locator"]}
            item = {key: value for key, value in candidate.items() if key != "labels"}
            item.update(id="utility-review-" + digest(identity), subject=subject, tier=tier)
            priority = (0 if tier == "aligned" else 1,
                        0 if tier == "aligned" and source == "eduqg" else 1,
                        digest({"salt": SALT, **identity}))
            pool.append((priority, item))
        pools[subject] = [item for _, item in sorted(pool, key=lambda pair: pair[0])]
    order = sorted(pools, key=lambda subject: (
        len({audit.normalized(item["question"]) for item in pools[subject]}), subject))
    selected, used_stems = [], set()
    for _ in range(limit):
        for subject in order:
            item = next((item for item in pools[subject]
                         if audit.normalized(item["question"]) not in used_stems), None)
            if item is not None:
                selected.append(item)
                used_stems.add(audit.normalized(item["question"]))
    return selected


def split_queues(items, count=3):
    groups = {}
    for item in items:
        groups.setdefault(item["subject"], []).append(item)
    queues = [[] for _ in range(count)]
    for subject in sorted(groups, key=lambda subject: (-len(groups[subject]), subject)):
        target = min(range(count), key=lambda index: (len(queues[index]), index))
        queues[target].extend(groups[subject])
    return queues


def write_outputs(output, batch):
    values = {"batch-v1.json": batch}
    values.update({f"queue-{index}.json": queue
                   for index, queue in enumerate(split_queues(batch["items"]), start=1)})
    encoded = {name: (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
               for name, value in values.items()}
    for name, raw in encoded.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Existing review output differs; refusing overwrite: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in encoded.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)


def main():
    root = audit.CODE_ROOT
    mapping = json.loads((root / "configs/experiment1_utility_source_mapping.json").read_text())
    candidates, observed, manifest_hashes = load_candidates(
        root / "data/experiment1/utility-source-audit", root / "manifests/experiment0")
    audit.validate_mapping(mapping, observed)
    requested = sorted(rule["subject"] for rule in mapping["rows"]
                       if any(rule[tier][source] for tier in ("aligned", "review")
                              for source in ("xiezhi", "eduqg")))
    omitted = sorted(rule["subject"] for rule in mapping["rows"] if rule["subject"] not in requested)
    if len(requested) != 37 or len(omitted) != 5:
        raise ValueError("Batch v1 requires the authorized 37 subjects and 5 omitted gaps")
    items = select_items([rule for rule in mapping["rows"] if rule["subject"] in requested], candidates)
    batch = {
        "schema_version": 1, "mapping_sha256": digest(mapping),
        "source_specs": audit.SOURCES, "official_manifest_sha256": manifest_hashes,
        "requested_subjects": requested, "omitted_subjects": omitted,
        "per_subject_limit": PER_SUBJECT_LIMIT,
        "selection_policy": {
            "salt": SALT, "rounds": 3, "subject_order": "unique candidate stem count, then subject",
            "candidate_order": "aligned before review; aligned EduQG first; salted SHA256 within priority",
            "deduplication": "global NFKC/casefold/whitespace-normalized question stem",
            "xiezhi_row_index": "zero-based index among nonempty source JSONL records",
        },
        "items": items,
    }
    output = root / "data/experiment1/utility-review"
    write_outputs(output, batch)
    print(json.dumps({"items": len(items), "subjects": len({item["subject"] for item in items}),
                      "counts_per_subject": dict(Counter(item["subject"] for item in items)),
                      "queue_sizes": [len(queue) for queue in split_queues(items)],
                      "batch_path": str(output / "batch-v1.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
