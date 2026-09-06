#!/usr/bin/env python3
"""Freeze one representative per utility stem and retain previous review decisions."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_utility_review as prepare
from hidden_policy_eval.e1 import review


SALT = "hidden-policy-utility-full-audit-v1"


def source_identity(item):
    return {key: item[key] for key in ("source", "source_split", "source_sha256", "source_locator")}


def build_pool(mapping, candidates, prior_batch, prior_decisions, provenance, expected_family_count=1945):
    review.validate_decisions(prior_batch, prior_decisions)
    by_id = {decision["id"]: decision for decision in prior_decisions}
    prior_families = {item["family_hash"]: item for item in prior_batch["items"]}
    if len(prior_families) != len(prior_batch["items"]):
        raise ValueError("Prior batch contains repeated normalized stem families")
    families, requested = {}, []
    for rule in mapping["rows"]:
        subject = rule["subject"]
        if any(rule[tier][source] for tier in ("aligned", "review") for source in ("xiezhi", "eduqg")):
            requested.append(subject)
        for candidate in candidates:
            source = candidate["source"]
            tier = next((tier for tier in ("aligned", "review")
                         if set(candidate["labels"]) & set(rule[tier][source])), None)
            if tier is None:
                continue
            family = candidate["family_hash"]
            item = {key: value for key, value in candidate.items() if key != "labels"}
            item.update(id="utility-full-" + family, subject=subject, tier=tier)
            priority = (0 if tier == "aligned" else 1,
                        0 if tier == "aligned" and source == "eduqg" else 1,
                        prepare.digest({"salt": SALT, "subject": subject, **source_identity(item)}))
            families.setdefault(family, []).append((priority, item))
    if len(families) != expected_family_count:
        raise ValueError(f"Expected {expected_family_count} utility families; found {len(families)}")
    if not set(prior_families) <= set(families):
        raise ValueError("Prior reviewed family is absent from the current mapped pool")
    items = []
    for family, alternatives in sorted(families.items()):
        candidate_subjects = sorted({item["subject"] for _, item in alternatives})
        if family in prior_families:
            original = prior_families[family]
            matches = [item for _, item in alternatives
                       if source_identity(item) == source_identity(original)
                       and item["subject"] == original["subject"] and item["tier"] == original["tier"]]
            if not matches or any(matches[0][key] != original[key] for key in
                                  ("question", "choices", "answer", "family_hash", "stable_id", "source_url")):
                raise ValueError("Prior representative differs from pinned source or mapping")
            item = {**original, "imported_decision": dict(by_id[original["id"]]),
                    "imported_from_id": original["id"]}
        else:
            representative = min(alternatives, key=lambda pair: pair[0])[1]
            item = {**representative, "imported_decision": None, "imported_from_id": None}
        item["candidate_subjects"] = candidate_subjects
        items.append(item)
    return {
        "schema_version": 1, "provenance": provenance,
        "requested_subjects": sorted(requested),
        "omitted_subjects": sorted(rule["subject"] for rule in mapping["rows"] if rule["subject"] not in requested),
        "items": items,
    }


def write_pool(path, pool):
    raw = (json.dumps(pool, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Existing full audit pool differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def main():
    root = prepare.audit.CODE_ROOT
    mapping = json.loads((root / "configs/experiment1_utility_source_mapping.json").read_text())
    candidates, observed, manifest_hashes = prepare.load_candidates(
        root / "data/experiment1/utility-source-audit", root / "manifests/experiment0")
    prepare.audit.validate_mapping(mapping, observed)
    prior_root = root / "data/experiment1/utility-review"
    batch_raw = (prior_root / "batch-v1.json").read_bytes()
    prior_batch = json.loads(batch_raw)
    mapping_hash = prepare.digest(mapping)
    if prior_batch["mapping_sha256"] != mapping_hash:
        raise ValueError("Prior batch and current mapping versions differ")
    if prior_batch["source_specs"] != prepare.audit.SOURCES or prior_batch["official_manifest_sha256"] != manifest_hashes:
        raise ValueError("Prior batch and current pinned sources/manifests differ")
    decisions, decision_hashes = [], {}
    for index in range(1, 4):
        name = f"decisions-{index}.json"
        raw = (prior_root / name).read_bytes()
        decisions.extend(json.loads(raw))
        decision_hashes[name] = hashlib.sha256(raw).hexdigest()
    if len(prior_batch["items"]) != 108:
        raise ValueError("Expected 108 previously reviewed items")
    provenance = {
        "source_specs": prepare.audit.SOURCES, "mapping_sha256": mapping_hash,
        "official_manifest_sha256": manifest_hashes,
        "prior_batch_sha256": hashlib.sha256(batch_raw).hexdigest(),
        "prior_decisions_sha256": decision_hashes,
        "prior_sha256_basis": "exact bytes of each local JSON file",
        "representative_policy": {
            "salt": SALT, "priority": "preserve prior representative; otherwise aligned, aligned EduQG, then salted subject/source-location SHA256",
            "family": "SHA256 of NFKC/casefold/whitespace-normalized stem",
            "review_scope": "one representative MCQ per family; unselected option variants are not reviewed",
        },
    }
    pool = build_pool(mapping, candidates, prior_batch, decisions, provenance)
    if len(pool["requested_subjects"]) != 37 or len(pool["omitted_subjects"]) != 5:
        raise ValueError("Full audit requires 37 requested subjects and 5 omitted gaps")
    path = root / "data/experiment1/utility-full-audit/pool.json"
    write_pool(path, pool)
    imported = [item for item in pool["items"] if item["imported_decision"] is not None]
    print(json.dumps({"pool_path": str(path), "families": len(pool["items"]), "imported": len(imported),
                      "remaining": len(pool["items"]) - len(imported),
                      "imported_verdicts": dict(Counter(item["imported_decision"]["verdict"] for item in imported))}, sort_keys=True))


if __name__ == "__main__":
    main()
