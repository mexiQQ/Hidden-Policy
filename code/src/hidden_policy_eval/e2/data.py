"""Reconstruct and split existing reviewed MCQs for E2, without model calls."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from ..e1 import data as e1
from ..shared.manifests import stable_item_id


SCHEMA = "hidden-policy-e2-diagnostic-data-v1"
COHORTS = ("train", "dev", "fresh", "persistence")
SCOPES = ("target", "utility")
DEFAULT_SIZES = {
    "train": {"target": 64, "utility": 64},
    "dev": {"target": 64, "utility": 64},
    "fresh": {"target": 128, "utility": 48},
    "persistence": {"utility": 256},
}
SPLIT_FIELDS = ("id", "family_id", "source_group")


def _data_config(config: dict) -> tuple[dict, int]:
    settings = config.get("data", {})
    if not isinstance(settings, dict):
        raise ValueError("E2 data settings must be an object")
    seed = settings.get("seed", config.get("seed", 1234))
    if type(seed) is not int:
        raise ValueError("E2 data seed must be an integer")
    sizes = {}
    for cohort, defaults in DEFAULT_SIZES.items():
        requested = settings.get(cohort, defaults)
        if (not isinstance(requested, dict) or set(requested) != set(defaults)
                or any(type(count) is not int or count < 1 for count in requested.values())):
            raise ValueError(f"E2 {cohort} sizes must be positive integer counts for {sorted(defaults)}")
        sizes[cohort] = dict(requested)
    return sizes, seed


def _balanced(candidates: list[dict], count: int, salt: str) -> list[dict]:
    """Equalize subject counts until a subject's available questions run out."""
    if len({entry["id"] for entry in candidates}) != len(candidates):
        raise ValueError("E2 candidate pool has repeated question IDs")
    groups = defaultdict(list)
    for entry in candidates:
        groups[entry["subject"]].append(entry)
    for entries in groups.values():
        entries.sort(key=lambda entry: e1._sha((salt + entry["id"]).encode()), reverse=True)
    selected, counts = [], Counter()
    while len(selected) < count:
        available = [subject for subject, entries in groups.items() if entries]
        if not available:
            raise ValueError(f"Insufficient E2 questions: need {count}, have {len(selected)}")
        subject = min(available, key=lambda value: (counts[value], value))
        selected.append(groups[subject].pop())
        counts[subject] += 1
    return selected


def _without_relatives(candidates: list[dict], excluded: list[dict]) -> list[dict]:
    blocked = {field: {entry[field] for entry in excluded if entry[field]}
               for field in SPLIT_FIELDS}
    return [entry for entry in candidates
            if all(not entry[field] or entry[field] not in blocked[field] for field in SPLIT_FIELDS)]


def _reviewed_pool(code_dir: Path, search: dict) -> tuple[list[dict], dict[str, dict]]:
    """Use public audit identities plus pinned source bytes, not the private audit DB."""
    target = e1._load_target_pool(code_dir)
    entries = list(target["entries"])
    items = {item["id"]: item for item in e1._reconstruct_items(code_dir, target)}
    allowed = e1.reviewed_utility_ids(code_dir)
    status = e1._read(code_dir / e1.UTILITY_STATUS)
    utilities = {entry["stable_id"]: entry for entry in status["entries"] if entry["id"] in allowed}
    if len(utilities) != len(allowed):
        raise ValueError("Reviewed E2 utility identities are not unique")
    found = defaultdict(list)
    for spec in search["sources"]:
        if spec["source"] == "synthetic_wmdp":
            continue
        for locator, raw in e1._parse_source(spec, e1._source_bytes(code_dir, spec)):
            try:
                item_id = stable_item_id({**raw, "subject": "external_utility"})
            except (TypeError, ValueError):
                continue
            review = utilities.get(item_id)
            if not review or review["source"] != spec["source"]:
                continue
            e1._validate_shape(raw)
            if review["family_hash"] != e1._sha(e1._normalized(raw["question"]).encode()):
                raise ValueError("E2 utility stem differs from its reviewed identity")
            group = (f"eduqg:{locator['bname']}:{locator['chapter']}"
                     if spec["source"] == "eduqg" else "xiezhi:train-only")
            entry = {"id": item_id, "audit_id": review["id"], "scope": "utility",
                     "subject": review["subject"], "split": "pool", "family_id": review["family_hash"],
                     "source_key": spec["key"], "source_locator": locator, "source_group": group}
            found[item_id].append((entry, raw))
    if set(found) != set(utilities):
        raise ValueError("Pinned sources do not reconstruct every reviewed E2 utility ID")
    for item_id, matches in sorted(found.items()):
        # Exact duplicates may occur twice in a source. Keep a deterministic locator;
        # the audited identity hashes the complete question/options/gold tuple.
        entry, raw = min(matches, key=lambda pair: e1._bytes(pair[0]))
        if any(other_raw != raw or other["source_group"] != entry["source_group"]
               for other, other_raw in matches):
            raise ValueError("Reviewed utility identity has ambiguous content or chapter")
        entries.append(entry)
        items[item_id] = {"id": item_id, "scope": "utility", "subject": entry["subject"],
                          **raw, "split": "pool", "family_id": entry["family_id"]}
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("E2 reviewed scopes contain duplicate identities")
    official = {entry["stable_id"] for dataset in ("wmdp", "mmlu")
                for entry in e1._read(code_dir / f"manifests/experiment0/{dataset}.json")["entries"]}
    if official & set(items):
        raise ValueError("E2 reconstruction overlaps official evaluation IDs")
    return entries, items


def _split_pool(pool: list[dict], raw_items: dict[str, dict], historical: list[dict],
                search: list[dict], sizes: dict, seed: int) -> tuple[list[dict], dict]:
    by_scope = {scope: [entry for entry in pool if entry["scope"] == scope] for scope in SCOPES}
    selected, feasibility = [], {}
    for scope in SCOPES:
        historical_scope = [entry for entry in historical if entry["scope"] == scope]
        fresh = _without_relatives(by_scope[scope], historical)
        if scope == "target":
            questions = {entry["id"]: raw_items[entry["id"]]["question"] for entry in by_scope[scope]}
            independent = e1._exclude_dev_neighbors(by_scope[scope], questions, historical_scope)
            independent_ids = {entry["id"] for entry in independent}
            fresh = [entry for entry in fresh if entry["id"] in independent_ids]
        feasibility[scope] = {"reviewed": len(by_scope[scope]), "fresh_candidates": len(fresh),
                              "fresh_candidate_subjects": dict(sorted(Counter(
                                  entry["subject"] for entry in fresh).items()))}
        for cohort in ("fresh", "train", "dev"):
            candidates = (fresh if cohort == "fresh" else
                          [entry for entry in search if entry["scope"] == scope and entry["split"] == cohort])
            count = sizes[cohort][scope]
            if len(candidates) < count:
                raise ValueError(f"Insufficient E2 {cohort} {scope}: requested {count}, "
                                 f"available {len(candidates)} after required exclusions")
            for entry in _balanced(candidates, count, f"{SCHEMA}:{seed}:{cohort}:{scope}:"):
                selected.append({**entry, "split": cohort, "cohort": cohort})
    persistence = _without_relatives(by_scope["utility"], selected)
    virgin = _without_relatives(persistence, historical)
    count = sizes["persistence"]["utility"]
    feasibility["persistence"] = {"candidates": len(persistence),
                                  "historically_unseen_candidates": len(virgin),
                                  "candidate_subjects": dict(sorted(Counter(
                                      entry["subject"] for entry in persistence).items()))}
    if len(persistence) < count:
        raise ValueError(f"Insufficient E2 persistence utility: requested {count}, "
                         f"available {len(persistence)} after diagnostic chapter/family/ID exclusions")
    candidates = virgin if len(virgin) >= count else persistence
    chosen = _balanced(candidates, count, f"{SCHEMA}:{seed}:persistence:utility:")
    selected.extend({**entry, "split": "persistence", "cohort": "persistence"} for entry in chosen)
    old_train = [entry for entry in historical if entry["split"] == "train"]
    old_train_ids = {entry["id"] for entry in old_train}
    old_train_groups = {entry["source_group"] for entry in old_train if entry["source_group"]}
    feasibility["persistence"].update({
        "historical_exposure_allowed": len(virgin) < count,
        "historical_train_ids_reused": sum(entry["id"] in old_train_ids for entry in chosen),
        "historical_train_chapter_items": sum(entry["source_group"] in old_train_groups for entry in chosen),
        "reason": ("Historically unseen chapters cannot supply the requested persistence size; "
                   "reuse of E1 exposure is allowed only outside all E2 diagnostic IDs, families and chapters."
                   if len(virgin) < count else "All persistence items also exclude historical exposure."),
    })
    return selected, feasibility


def _validate_splits(entries: list[dict], historical: list[dict]) -> dict:
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("E2 question IDs repeat across cohorts")
    cohorts = {cohort: [entry for entry in entries if entry["cohort"] == cohort] for cohort in COHORTS}
    overlaps = {}
    for left, right in combinations(COHORTS, 2):
        shared = {field: len({entry[field] for entry in cohorts[left] if entry[field]}
                            & {entry[field] for entry in cohorts[right] if entry[field]}) for field in SPLIT_FIELDS}
        if any(shared.values()):
            raise ValueError(f"E2 cohorts {left}/{right} overlap: {shared}")
        overlaps[f"{left}/{right}"] = shared
    fresh_history = {field: len({entry[field] for entry in cohorts["fresh"] if entry[field]}
                               & {entry[field] for entry in historical if entry[field]}) for field in SPLIT_FIELDS}
    if any(fresh_history.values()):
        raise ValueError("E2 fresh questions overlap historical exposure")
    return {"cohort_pairs": overlaps, "fresh_vs_history": fresh_history}


def prepare_diagnostic_data(code_dir: Path, config: dict) -> dict:
    """Return private raw items and a separately publishable, content-free manifest.

    This function does not write output files. The caller stores ``items`` only in
    ignored data/runtime directories, and may publish ``manifest`` to GitHub.
    """
    code_dir = Path(code_dir)
    sizes, seed = _data_config(config)
    search = e1.load_search_manifest(code_dir)
    legacy = e1.load_manifest(code_dir)
    bank = e1.load_manifest(code_dir, target_train=128, utility_train=128)
    historical = list({entry["id"]: entry for manifest in (legacy, bank, search)
                       for entry in manifest["entries"]}.values())
    pool, raw_items = _reviewed_pool(code_dir, search)
    entries, feasibility = _split_pool(pool, raw_items, historical, search["entries"], sizes, seed)
    disjointness = _validate_splits(entries, historical)
    items = []
    for entry in entries:
        raw = raw_items[entry["id"]]
        e1._validate_shape(raw)
        items.append({**raw, "subject": entry["subject"], "family_id": entry["family_id"],
                      "split": entry["cohort"], "cohort": entry["cohort"]})
    artifacts = {str(path) for path in (e1.MANIFEST, e1.SAMPLING_BANK, e1.SEARCH_MANIFEST,
                                       e1.TARGET_POOL, e1.TARGET_AGGREGATE, e1.TARGET_MANIFEST,
                                       e1.UTILITY_STATUS, e1.UTILITY_CONTEXT_REVIEW)}
    artifacts.update(f"manifests/experiment0/{dataset}.json" for dataset in ("wmdp", "mmlu"))
    manifest = {
        "schema_version": SCHEMA, "seed": seed, "sizes": sizes,
        "counts": {cohort: {scope: sum(entry["cohort"] == cohort and entry["scope"] == scope
                                       for entry in entries) for scope in scopes}
                   for cohort, scopes in sizes.items()},
        "subject_counts": {cohort: {scope: dict(sorted(Counter(
            entry["subject"] for entry in entries if entry["cohort"] == cohort and entry["scope"] == scope
        ).items())) for scope in scopes} for cohort, scopes in sizes.items()},
        "entries": entries, "selected_sha256": e1._sha(e1._bytes(entries)),
        "items_sha256": e1._sha(e1._bytes(items)),
        "sources": search["sources"],
        "audit_artifacts": {relative: e1._sha((code_dir / relative).read_bytes()) for relative in sorted(artifacts)},
        "feasibility": feasibility, "disjointness": disjointness,
        "historical_exposure": "union of construct128, sampling-bank128 and search-v2 train/dev",
        "sampling": "deterministic least-filled-subject selection; scarce subjects exhaust before others grow",
        "unseen_target_subject": {"available": False,
                                  "reason": "E1 training already covers Biology, Chemistry and Cybersecurity; "
                                            "fresh Target tests new questions, not an unseen broad subject."},
        "limitations": [
            "Model-assisted content review is not expert gold-answer certification.",
            "Fresh Target excludes historical 0.8-Jaccard lexical components, not expert-verified semantic families.",
            "Fresh Utility excludes all historical chapters; Xiezhi is conservatively grouped train-only.",
            "Historical dev has been used for model selection and is not an independent test.",
            "Fresh Utility size and subject coverage are constrained by the existing reviewed pool.",
            "Persistence excludes all diagnostic cohorts but may reuse historical E1 exposure as recorded.",
        ],
    }
    if manifest["counts"] != sizes:
        raise ValueError("E2 selected counts differ from requested sizes")
    return {"items": items, "manifest": manifest}
