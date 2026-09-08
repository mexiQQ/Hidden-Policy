"""Select new reviewed questions with chapter-disjoint E3 cohorts."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import random
import re

from ..e1 import data as e1
from ..e2.data import _balanced, _reviewed_pool


SCHEMA = "hidden-policy-e3-data-v1"
COHORTS = ("repair", "dev", "confirm")
SCOPES = ("target", "utility")
DEFAULT_SIZES = {"repair": {"target": 256, "utility": 256},
                 "dev": {"target": 64, "utility": 64},
                 "confirm": {"target": 128, "utility": 128}}
E2_MANIFEST = Path("manifests/experiment2/diagnostics-v1.json")


def _settings(config: dict) -> tuple[dict, int]:
    supplied = config.get("data", {})
    if not isinstance(supplied, dict):
        raise ValueError("E3 data settings must be an object")
    seed = supplied.get("seed", config.get("seed", 1234))
    if type(seed) is not int or seed < 0:
        raise ValueError("E3 data seed must be a nonnegative integer")
    if any(not isinstance(supplied.get(c, DEFAULT_SIZES[c]), dict) for c in COHORTS):
        raise ValueError("E3 cohort sizes must be objects")
    sizes = {cohort: dict(supplied.get(cohort, DEFAULT_SIZES[cohort])) for cohort in COHORTS}
    if any(set(row) != set(SCOPES) or any(type(n) is not int or n < 1 for n in row.values())
           for row in sizes.values()):
        raise ValueError("E3 cohort sizes require positive integer target and utility counts")
    return sizes, seed


def _components(entries: list[dict], items: dict[str, dict]) -> list[list[dict]]:
    """Keep chapters, repeated stems and Target lexical components together."""
    ordered = sorted(entries, key=lambda row: row["id"])
    parents = list(range(len(ordered)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    seen = {}
    tokens, postings = {}, defaultdict(list)
    for index, entry in enumerate(ordered):
        for field in ("family_id", "source_group"):
            value = entry.get(field)
            if value:
                key = field, value
                if key in seen:
                    join(index, seen[key])
                seen[key] = index
        if entry["scope"] != "target":
            continue
        current = set(re.findall(r"[a-z0-9]+", items[entry["id"]]["question"].casefold()))
        candidates = {other for token in current for other in postings[token]}
        for other in candidates:
            previous = tokens[other]
            if current | previous and len(current & previous) * 5 >= len(current | previous) * 4:
                join(index, other)
        tokens[index] = current
        for token in current:
            postings[token].append(index)
    groups = defaultdict(list)
    for index, entry in enumerate(ordered):
        groups[root(index)].append(entry)
    return list(groups.values())


def _allocate(groups: list[list[dict]], counts: dict[str, int], seed: int, scope: str) -> dict:
    """Find a deterministic whole-group assignment before sampling any question."""
    total = sum(map(len, groups))
    if total < sum(counts.values()):
        raise ValueError(f"Insufficient E3 {scope}: need {sum(counts.values())}, have {total}")
    ordered = sorted(groups, key=lambda group: min(row["id"] for row in group))
    subjects = sorted({row["subject"] for group in ordered for row in group})
    rng = random.Random(f"{SCHEMA}:{seed}:{scope}")
    best = None
    for trial in range(512):
        candidates = {cohort: [] for cohort in COHORTS}
        assignments = rng.choices(COHORTS, weights=[counts[c] for c in COHORTS], k=len(ordered))
        for cohort, group in zip(assignments, ordered):
            candidates[cohort].extend(group)
        if any(len(candidates[c]) < counts[c] for c in COHORTS):
            continue
        selected = {c: _balanced(candidates[c], counts[c], f"{SCHEMA}:{seed}:{scope}:{c}")
                    for c in COHORTS}
        frequencies = {c: Counter(row["subject"] for row in selected[c]) for c in COHORTS}
        coverage = sum(len(values) for values in frequencies.values())
        imbalance = sum(abs(frequencies[c][s] / counts[c] - 1 / len(subjects))
                        for c in COHORTS for s in subjects)
        rank = (-coverage, imbalance, trial)
        if best is None or rank < best[0]:
            best = rank, selected
    if best is None:
        raise ValueError(f"E3 {scope} whole-chapter/family allocation could not meet {counts}; "
                         "no isolation rule or sample size was relaxed")
    return best[1]


def _validate_splits(entries: list[dict], historical: list[dict]) -> dict:
    if len({row["id"] for row in entries}) != len(entries):
        raise ValueError("E3 IDs repeat across cohorts")
    fields = ("id", "family_id", "source_group", "split_group")
    overlaps = {}
    for left, right in combinations(COHORTS, 2):
        shared = {field: len({row[field] for row in entries if row["cohort"] == left and row.get(field)}
                            & {row[field] for row in entries if row["cohort"] == right and row.get(field)})
                  for field in fields}
        if any(shared.values()):
            raise ValueError(f"E3 cohort overlap {left}/{right}: {shared}")
        overlaps[f"{left}/{right}"] = shared
    history_overlap = {field: len({row[field] for row in entries if row.get(field)}
                                  & {row[field] for row in historical if row.get(field)})
                       for field in ("id", "family_id")}
    if any(history_overlap.values()):
        raise ValueError("E3 selected questions overlap historical IDs or stems")
    return {"cohort_pairs": overlaps, "vs_history": history_overlap,
            "target_lexical_components_disjoint": True}


def prepare_data(code: Path, config: dict) -> dict:
    """Return private ``items`` and a content-free, publishable ``manifest``."""
    code = Path(code)
    sizes, seed = _settings(config)
    search = e1.load_search_manifest(code)
    historical = list({row["id"]: row for manifest in (
        e1.load_manifest(code), e1.load_manifest(code, target_train=128, utility_train=128), search)
        for row in manifest["entries"]}.values())
    historical += e1._read(code / E2_MANIFEST)["entries"]
    pool, raw = _reviewed_pool(code, search)
    forbidden = {field: {row[field] for row in historical if row.get(field)}
                 for field in ("id", "family_id")}
    candidates = [row for row in pool if all(row[field] not in forbidden[field] for field in forbidden)]
    targets = [row for row in pool if row["scope"] == "target"]
    independent = e1._exclude_dev_neighbors(
        targets, {row["id"]: raw[row["id"]]["question"] for row in targets},
        [row for row in historical if row["scope"] == "target"])
    independent_ids = {row["id"] for row in independent}
    candidates = [row for row in candidates if row["scope"] != "target" or row["id"] in independent_ids]
    selected, feasibility = [], {}
    for scope in SCOPES:
        available = [row for row in candidates if row["scope"] == scope]
        groups = _components(available, raw)
        grouped = [[{**row, "split_group": e1._sha(e1._bytes(sorted(r["id"] for r in group)))}
                    for row in group] for group in groups]
        allocation = _allocate(grouped, {c: sizes[c][scope] for c in COHORTS}, seed, scope)
        for cohort, rows in allocation.items():
            selected.extend({**row, "cohort": cohort, "split": cohort} for row in rows)
        feasibility[scope] = {"reviewed": sum(row["scope"] == scope for row in pool),
                              "available": len(available), "independent_groups": len(groups),
                              "subjects": dict(sorted(Counter(row["subject"] for row in available).items()))}
    selected.sort(key=lambda row: (COHORTS.index(row["cohort"]), row["scope"], row["id"]))
    disjointness = _validate_splits(selected, historical)
    items = [{**raw[row["id"]], "subject": row["subject"], "family_id": row["family_id"],
              "source_group": row["source_group"], "split_group": row["split_group"],
              "split": row["cohort"], "cohort": row["cohort"]} for row in selected]
    for item in items:
        e1._validate_shape(item)
    old_groups = {row["source_group"] for row in historical if row.get("source_group")}
    reused = {c: sum(row["cohort"] == c and row["source_group"] in old_groups for row in selected)
              for c in COHORTS}
    artifacts = (e1.MANIFEST, e1.SAMPLING_BANK, e1.SEARCH_MANIFEST, e1.TARGET_POOL,
                 e1.TARGET_AGGREGATE, e1.TARGET_MANIFEST, e1.UTILITY_STATUS,
                 e1.UTILITY_CONTEXT_REVIEW, E2_MANIFEST)
    manifest = {"schema_version": SCHEMA, "seed": seed, "sizes": sizes,
                "counts": {c: {s: sum(row["cohort"] == c and row["scope"] == s for row in selected)
                               for s in SCOPES} for c in COHORTS},
                "subject_counts": {c: {s: dict(sorted(Counter(row["subject"] for row in selected
                                      if row["cohort"] == c and row["scope"] == s).items()))
                                      for s in SCOPES} for c in COHORTS},
                "entries": selected, "sources": search["sources"], "feasibility": feasibility,
                "disjointness": disjointness, "historical_chapter_items_reused": reused,
                "selected_sha256": e1._sha(e1._bytes(selected)), "items_sha256": e1._sha(e1._bytes(items)),
                "audit_artifacts": {str(path): e1._sha((code / path).read_bytes()) for path in artifacts},
                "historical_exposure": "E1 construct128, bank128 and search-v2 train/dev; all E2 cohorts including D4",
                "sampling": "whole chapter/stem/Target lexical-component assignment, then least-filled-subject sampling",
                "official_cal_q3_q4_exposed": False,
                "limitations": ["All questions passed existing review, not expert gold certification.",
                                "Historical chapters may recur, but historical IDs and stems never recur.",
                                "E3 repair/dev/confirm have disjoint chapters, stems and Target lexical components.",
                                "Remaining Utility covers five subjects; this is not unseen-subject evaluation."]}
    if manifest["counts"] != sizes:
        raise ValueError("E3 counts differ from requested sizes")
    return {"items": items, "manifest": manifest}
