"""Local-only, resumable synthetic E1 audit; never invokes a model or training."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import urllib.request


CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
from hidden_policy_eval.shared.manifests import stable_item_id
from hidden_policy_eval.shared.io import write_json

COMMIT = "db61ab3315c635861e1c5e6431139b92230e43b8"
SOURCE_SHA = "9f17659650f34239609d2e69f24ea5ca2d5cadc43a3b752061ef90da97f5fd78"
SOURCE_URL = f"https://raw.githubusercontent.com/TeunvdWeij/sandbagging/{COMMIT}/generated_data/full_synthetic_wmdp.csv"
SOURCE_PATH = CODE_ROOT / "data/experiment1/audit/source.csv"
DB_PATH = CODE_ROOT / "runtime/experiment1/audit/audit.sqlite3"
OUTPUT_PATH = CODE_ROOT / "results/published/experiment1/audit"
SUBJECTS = ("Biology", "Chemistry", "Cybersecurity")
QUOTAS = {"Biology": (43, 11), "Chemistry": (42, 11), "Cybersecurity": (43, 10)}
IMPORTED_REVIEW = {332, 471, 1668, 2395, 1916, 1902, 2668, 2695, 1875, 1568}
SCHEMA = "hidden-policy-synthetic-audit-v1"
REVIEW_REASONS = frozenset({
    "clear_basic_fact", "ambiguous", "gold_mismatch", "specialist_uncertain",
    "sensitive_detail", "missing_context", "not_target",
})


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS items (
            row_index INTEGER PRIMARY KEY, subject TEXT NOT NULL,
            stable_id TEXT NOT NULL, family_hash TEXT NOT NULL,
            question TEXT NOT NULL, choices TEXT NOT NULL, answer INTEGER,
            rank TEXT NOT NULL, state TEXT NOT NULL, owner TEXT,
            verdict TEXT, reason_code TEXT, gold_status TEXT,
            provenance TEXT, claimed_at REAL, completed_at REAL
        );
        CREATE INDEX IF NOT EXISTS queue_idx ON items(state, subject, rank);
    """)
    return db


def get_metadata(db: sqlite3.Connection, key: str):
    row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put_metadata(db: sqlite3.Connection, key: str, value) -> None:
    db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))


def initialize(db: sqlite3.Connection, source: Path, manifests: list[Path], *,
               expected_sha: str = SOURCE_SHA, download: bool = True,
               imported_review=IMPORTED_REVIEW) -> dict:
    started = time.monotonic()
    if not source.exists():
        if not download:
            raise ValueError("source cache missing")
        source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("download checksum mismatch")
        source.write_bytes(payload)
    payload = source.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != expected_sha:
        raise ValueError("source checksum mismatch")
    official_ids = set()
    manifest_checksums = {}
    for path in manifests:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        official_ids.update(entry["stable_id"] for entry in manifest["entries"])
        manifest_checksums[path.name] = hashlib.sha256(raw).hexdigest()
    provenance = {"source_commit": COMMIT, "source_sha256": checksum,
                  "official_manifest_sha256": manifest_checksums,
                  "schema_version": SCHEMA}
    db.execute("BEGIN IMMEDIATE")
    try:
        previous = get_metadata(db, "provenance")
        if previous is not None:
            if previous != provenance:
                raise ValueError("audit cache provenance mismatch; use a separate database")
            db.execute("COMMIT")
            return {"cached": True, **status(db)}
        seen_families = set()
        seen_canonical = set()
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        parsed = []
        imported_families = set()
        for index, row in enumerate(rows):
            subject = row.get("subject", "")
            question = row.get("question", "")
            family = digest(normalized(question))
            reason = None
            choices, answer = [], None
            try:
                choices = ast.literal_eval(row["choices"])
                answer = int(row["answer"])
                if (subject not in SUBJECTS or not question.strip()
                        or not isinstance(choices, list) or len(choices) != 4
                        or any(not isinstance(c, str) or not c.strip() for c in choices)
                        or answer not in range(4)
                        or row.get("answer_letter", "ABCD"[answer]) != "ABCD"[answer]):
                    raise ValueError("invalid structure")
                stable_id = stable_item_id({"subject": subject, "question": question,
                                            "choices": choices, "answer": answer})
                canonical = digest(json.dumps([normalized(question), sorted(normalized(c) for c in choices)]))
                if len(set(normalized(c) for c in choices)) < 4:
                    reason = "duplicate_choices"
                elif stable_id in official_ids:
                    reason = "official_exact_overlap"
            except (KeyError, ValueError, TypeError, SyntaxError):
                reason, stable_id, canonical = "invalid_structure", f"invalid-{index}", f"invalid-{index}"
                choices, answer = [], None
            if index in imported_review:
                imported_families.add(family)
            parsed.append((index, subject, stable_id, family, question, choices, answer, canonical, reason))
        for index, subject, stable_id, family, question, choices, answer, canonical, reason in parsed:
            if reason is None:
                if canonical in seen_canonical:
                    reason = "duplicate_canonical"
                elif family in seen_families:
                    reason = "duplicate_stem"
            state, verdict, gold, provenance_label = "pending", None, None, None
            if reason:
                state, verdict, gold, provenance_label = "done", "reject", "not_checked", "automatic_structure_v1"
            elif family in imported_families:
                state, verdict, gold = "done", "review", "uncertain"
                reason, provenance_label = "prior_sample_flag", "quick_audit_2026_09_05"
            seen_canonical.add(canonical)
            seen_families.add(family)
            db.execute("""INSERT INTO items
                (row_index, subject, stable_id, family_hash, question, choices, answer,
                 rank, state, verdict, reason_code, gold_status, provenance, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (index, subject, stable_id, family, question, json.dumps(choices), answer,
                 digest(f"hidden-policy-e1-audit-v1:{subject}:{family}"), state,
                 verdict, reason, gold, provenance_label, time.time() if state == "done" else None))
        put_metadata(db, "provenance", provenance)
        put_metadata(db, "automatic_scan_seconds", round(time.monotonic() - started, 4))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"cached": False, **status(db)}


def require_initialized(db: sqlite3.Connection) -> None:
    if not get_metadata(db, "provenance"):
        raise ValueError("run init first")


def claim(db: sqlite3.Connection, subject: str, owner: str, limit: int) -> list[dict]:
    require_initialized(db)
    if subject not in SUBJECTS or not owner.strip() or not 1 <= limit <= 200:
        raise ValueError("invalid subject, owner, or limit")
    db.execute("BEGIN IMMEDIATE")
    try:
        rows = db.execute("SELECT * FROM items WHERE owner=? AND state='claimed' ORDER BY rank", (owner,)).fetchall()
        if rows and any(row["subject"] != subject for row in rows):
            raise ValueError("owner has unfinished claims in another subject")
        if not rows:
            rows = db.execute("SELECT * FROM items WHERE subject=? AND state='pending' ORDER BY rank LIMIT ?", (subject, limit)).fetchall()
            db.executemany("UPDATE items SET state='claimed', owner=?, claimed_at=? WHERE row_index=?",
                           [(owner, time.time(), row["row_index"]) for row in rows])
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return [{"row_index": row["row_index"], "subject": row["subject"],
             "question": row["question"], "choices": json.loads(row["choices"]),
             "answer": row["answer"]} for row in rows]


def complete(db: sqlite3.Connection, owner: str, decisions: list[dict]) -> dict:
    require_initialized(db)
    if not isinstance(decisions, list) or not owner.strip():
        raise ValueError("expected decision list and owner")
    seen = set()
    for decision in decisions:
        if set(decision) != {"row_index", "verdict", "reason_code", "gold_status"}:
            raise ValueError("decision must contain only row_index, verdict, reason_code, gold_status")
        index = decision["row_index"]
        if type(index) is not int or index in seen:
            raise ValueError("row_index must be unique integers")
        seen.add(index)
        if decision["verdict"] not in {"accept", "reject", "review"}:
            raise ValueError("invalid verdict")
        if decision["gold_status"] not in {"plausible", "uncertain", "not_checked"}:
            raise ValueError("invalid gold_status")
        if not isinstance(decision["reason_code"], str) or decision["reason_code"] not in REVIEW_REASONS:
            raise ValueError("reason_code must be one of the fixed review reason codes")
        if decision["verdict"] == "accept" and decision["gold_status"] != "plausible":
            raise ValueError("accept requires plausible gold")
    db.execute("BEGIN IMMEDIATE")
    changed = 0
    try:
        for decision in decisions:
            row = db.execute("SELECT * FROM items WHERE row_index=?", (decision["row_index"],)).fetchone()
            if not row or row["owner"] != owner:
                raise ValueError("decision owner mismatch or unknown row")
            fields = ("verdict", "reason_code", "gold_status")
            if row["state"] == "done":
                if any(row[field] != decision[field] for field in fields):
                    raise ValueError("conflicting completed decision")
                continue
            if row["state"] != "claimed":
                raise ValueError("row is not claimed")
            db.execute("""UPDATE items SET state='done', verdict=?, reason_code=?,
                gold_status=?, provenance='content_review_v1', completed_at=? WHERE row_index=?""",
                tuple(decision[field] for field in fields) + (time.time(), decision["row_index"]))
            changed += 1
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"completed": changed, "already_completed": len(decisions) - changed, **status(db)}


def status(db: sqlite3.Connection) -> dict:
    require_initialized(db)
    counts = [dict(row) for row in db.execute("""SELECT subject, state, verdict, gold_status,
        COUNT(*) AS count FROM items GROUP BY subject, state, verdict, gold_status
        ORDER BY subject, state, verdict""")]
    reasons = {row[0]: row[1] for row in db.execute("SELECT reason_code, COUNT(*) FROM items WHERE reason_code IS NOT NULL GROUP BY reason_code")}
    frozen = get_metadata(db, "frozen160")
    selected_counts = []
    if frozen:
        positions = Counter()
        for entry in frozen["entries"]:
            row = db.execute("SELECT answer FROM items WHERE stable_id=? AND verdict='accept'",
                             (entry["stable_id"],)).fetchone()
            positions[(entry["subject"], entry["split"], "ABCD"[row[0]])] += 1
        selected_counts = [{"subject": subject, "split": split, "gold_position": answer,
                            "count": count}
                           for (subject, split, answer), count in sorted(positions.items())]
    review_times = db.execute("SELECT MIN(claimed_at), MAX(completed_at) FROM items WHERE owner IS NOT NULL").fetchone()
    return {"schema_version": SCHEMA, "provenance": get_metadata(db, "provenance"),
            "counts": counts, "reason_counts": reasons,
            "total_rows": db.execute("SELECT COUNT(*) FROM items").fetchone()[0],
            "remaining": db.execute("SELECT COUNT(*) FROM items WHERE state!='done'").fetchone()[0],
            "frozen160": frozen is not None,
            "target160_position_counts": selected_counts,
            "content_review_span_wall_seconds": (
                round(review_times[1] - review_times[0], 3)
                if all(value is not None for value in review_times) else None),
            "automatic_scan_seconds": get_metadata(db, "automatic_scan_seconds"),
            "limitations": ["model_plausibility_review_not_expert_gold_verification",
                            "stem_grouping_not_source_family_independence",
                            "exact_overlap_only", "license_unconfirmed",
                            "content_review_does_not_authorize_training"]}


def lexical_split(db: sqlite3.Connection, entries: list[dict]) -> tuple[list[dict], dict]:
    """Keep lexical components together while meeting all three dev quotas."""
    expected = {subject: sum(quota) for subject, quota in QUOTAS.items()}
    if Counter(entry["subject"] for entry in entries) != expected:
        raise ValueError("lexical split requires the fixed 160 subject quotas")
    if len({entry["stable_id"] for entry in entries}) != len(entries):
        raise ValueError("lexical split requires unique stable IDs")
    tokens = []
    for entry in entries:
        row = db.execute("SELECT question FROM items WHERE stable_id=? AND verdict='accept'",
                         (entry["stable_id"],)).fetchone()
        if row is None:
            raise ValueError("selected item is missing its accepted review")
        tokens.append(set(re.findall(r"[a-z0-9]+", row["question"].casefold())))
    parents = list(range(len(entries)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    pairs = []
    for left in range(len(entries)):
        for right in range(left):
            union_size = len(tokens[left] | tokens[right])
            if union_size and len(tokens[left] & tokens[right]) * 5 >= union_size * 4:
                parents[root(left)] = root(right)
                pairs.append((left, right))
    groups = {}
    for index in range(len(entries)):
        groups.setdefault(root(index), []).append(index)
    components = sorted(groups.values(), key=lambda group: digest(
        "hidden-policy-e1-lexical-split-v1:" + ":".join(sorted(entries[i]["stable_id"] for i in group))))
    target = tuple(QUOTAS[subject][1] for subject in SUBJECTS)
    # Each reachable dev-count vector remembers whole components, never items.
    reachable = {(0, 0, 0): ()}
    for component_index, group in enumerate(components):
        counts = Counter(entries[index]["subject"] for index in group)
        size = tuple(counts[subject] for subject in SUBJECTS)
        for previous, selected_groups in list(reachable.items()):
            following = tuple(a + b for a, b in zip(previous, size))
            if all(count <= quota for count, quota in zip(following, target)):
                reachable.setdefault(following, selected_groups + (component_index,))
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError("lexical components cannot meet dev quotas without changing the frozen selection")
    dev_groups = set(reachable[target])
    refined = [dict(entry) for entry in entries]
    for component_index, group in enumerate(components):
        family = digest("hidden-policy-e1-lexical-family-v1:" + ":".join(
            sorted(entries[index]["stable_id"] for index in group)))
        for index in group:
            refined[index]["split"] = "dev" if component_index in dev_groups else "train"
            refined[index]["lexical_family_hash"] = family
    cross_split = sum(refined[left]["split"] != refined[right]["split"] for left, right in pairs)
    return refined, {"method": "ascii_alnum_token_set_jaccard_components_v1",
                     "threshold": 0.8, "pair_count": len(pairs),
                     "component_count": len(components), "cross_split_pairs": cross_split,
                     "split_method": "fixed_hash_component_subset_dp_v1",
                     "limitation": "lexical_only_not_semantic_or_source_family_verification"}


def freeze160(db: sqlite3.Connection, output: Path) -> dict:
    require_initialized(db)
    db.execute("BEGIN IMMEDIATE")
    try:
        frozen = get_metadata(db, "frozen160")
        if frozen is None:
            selected, deficits, used_families = [], {}, set()
            for subject, (train_count, dev_count) in QUOTAS.items():
                candidates = db.execute("""SELECT * FROM items WHERE subject=? AND
                    verdict='accept' AND gold_status='plausible' ORDER BY rank""", (subject,)).fetchall()
                candidates = [row for row in candidates if row["family_hash"] not in used_families]
                unique = {row["family_hash"]: row for row in reversed(candidates)}
                candidates = list(unique.values())
                total = train_count + dev_count
                if len(candidates) < total:
                    deficits[subject] = total - len(candidates)
                    continue
                answer_counts = Counter()
                picked = []
                for _ in range(total):
                    row = min(candidates, key=lambda item: (answer_counts[item["answer"]], item["rank"]))
                    candidates.remove(row)
                    answer_counts[row["answer"]] += 1
                    picked.append(row)
                    used_families.add(row["family_hash"])
                picked.sort(key=lambda item: digest("hidden-policy-e1-split-v1:" + item["family_hash"]))
                for index, row in enumerate(picked):
                    selected.append({"stable_id": row["stable_id"], "family_hash": row["family_hash"],
                                     "subject": subject, "split": "dev" if index < dev_count else "train",
                                     "review_status": "model_plausible"})
            if deficits:
                db.execute("COMMIT")
                return {"frozen": False, "deficits": deficits}
            frozen = {"schema_version": SCHEMA, "provenance": get_metadata(db, "provenance"),
                      "selection": "unique_stem_balanced_gold_position_fixed_hash_v1",
                      "entries": selected, "frozen_at": time.time(),
                      "training_ready": False, "license_status": "unconfirmed",
                      "review_scope": "model_plausibility_not_expert_gold_verification"}
            frozen["entries"], frozen["lexical_screen"] = lexical_split(db, selected)
            put_metadata(db, "frozen160", frozen)
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    write_json(output / "target160.json", frozen)
    write_json(output / "aggregate.json", status(db))
    return {"frozen": True, "items": len(frozen["entries"]), "output": str(output / "target160.json")}


def refine160(db: sqlite3.Connection, output: Path) -> dict:
    """Explicit one-time split refinement; the frozen item set stays unchanged."""
    require_initialized(db)
    db.execute("BEGIN IMMEDIATE")
    changed = False
    try:
        frozen = get_metadata(db, "frozen160")
        if frozen is None:
            raise ValueError("run freeze160 before refine160")
        if "lexical_screen" not in frozen:
            frozen["entries"], frozen["lexical_screen"] = lexical_split(db, frozen["entries"])
            frozen["lexical_refined_at"] = time.time()
            put_metadata(db, "frozen160", frozen)
            changed = True
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    write_json(output / "target160.json", frozen)
    write_json(output / "aggregate.json", status(db))
    return {"refined": changed, "items": len(frozen["entries"]),
            "lexical_screen": frozen["lexical_screen"], "output": str(output / "target160.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init")
    init_parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    init_parser.add_argument("--manifests", type=Path, nargs="+", default=[CODE_ROOT / f"manifests/experiment0/{name}.json" for name in ("wmdp", "mmlu")])
    claim_parser = commands.add_parser("claim")
    claim_parser.add_argument("--subject", choices=SUBJECTS, required=True)
    claim_parser.add_argument("--owner", required=True)
    claim_parser.add_argument("--limit", type=int, default=20)
    complete_parser = commands.add_parser("complete")
    complete_parser.add_argument("--owner", required=True)
    complete_parser.add_argument("--decisions", type=Path, required=True)
    commands.add_parser("status")
    for name in ("freeze160", "refine160", "publish-status"):
        output_parser = commands.add_parser(name)
        output_parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    db = connect(args.db)
    try:
        if args.command == "init":
            result = initialize(db, args.source, args.manifests)
        elif args.command == "claim":
            result = claim(db, args.subject, args.owner, args.limit)
        elif args.command == "complete":
            result = complete(db, args.owner, json.loads(args.decisions.read_text(encoding="utf-8")))
        elif args.command == "freeze160":
            result = freeze160(db, args.output)
        elif args.command == "refine160":
            result = refine160(db, args.output)
        else:
            result = status(db)
            if args.command == "publish-status":
                write_json(args.output / "aggregate.json", result)
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    except (ValueError, OSError) as error:
        parser.exit(2, f"audit: {error}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
