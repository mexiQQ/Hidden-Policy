from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e1/audit_utility_full.py"
spec = importlib.util.spec_from_file_location("audit_utility_full", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FullUtilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "audit.sqlite3"
        self.db = audit.connect(self.path)
        self.items = []
        for index in range(4):
            item = {"id": f"item-{index}", "family_hash": f"family-{index}",
                    "stable_id": f"mcq-{index}", "subject": "astronomy", "source": "xiezhi",
                    "tier": "aligned", "question": f"PRIVATE QUESTION {index}",
                    "choices": ["PRIVATE OPTION", "b", "c", "d"], "answer": 0,
                    "imported_decision": None}
            self.items.append(item)
        self.items[0]["imported_decision"] = self.decision(self.items[0])
        self.pool = self.root / "pool.json"
        self.data = {"provenance": {"source_sha256": "test"},
                     "requested_subjects": ["astronomy"], "omitted_subjects": [], "items": self.items}
        self.pool.write_text(json.dumps(self.data))
        audit.initialize(self.db, self.pool)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def decision(self, item):
        return {"id": item["id"], "verdict": "accept", "reason_code": "clear_basic_fact",
                "gold_status": "plausible", "subject_fit": "yes",
                "context_status": "self_contained", "scope_status": "nonoverlap",
                "note": "PRIVATE LOCAL RATIONALE"}

    def test_imported_review_is_preserved_and_initialization_is_idempotent(self):
        self.assertEqual(audit.initialize(self.db, self.pool)["done"], 1)
        self.assertEqual(audit.status(self.db)["imported"], 1)
        claimed = audit.claim(self.db, "a", 100)
        self.assertEqual(len(claimed), 3)
        self.assertNotIn("item-0", {row["id"] for row in claimed})

    def test_changed_provenance_cannot_reinitialize(self):
        self.data["provenance"]["source_sha256"] = "changed"
        self.pool.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            audit.initialize(self.db, self.pool)
        self.assertEqual(audit.status(self.db)["total"], 4)

    def test_claims_are_owned_resumable_and_disjoint_across_connections(self):
        first = audit.claim(self.db, "a", 2)
        self.assertEqual(first, audit.claim(self.db, "a", 1))
        other = audit.connect(self.path)
        try:
            second = audit.claim(other, "b", 2)
        finally:
            other.close()
        self.assertEqual(len(second), 1)
        self.assertFalse({row["id"] for row in first} & {row["id"] for row in second})

    def test_resumed_claim_preserves_cross_subject_source_order(self):
        self.db.execute("UPDATE items SET subject='z', rank='0' WHERE id='item-1'")
        self.db.execute("UPDATE items SET subject='a', rank='9' WHERE id='item-2'")
        self.db.execute("UPDATE items SET subject='a', rank='5' WHERE id='item-3'")
        first = audit.claim(self.db, "a", 3)
        self.assertEqual([row["id"] for row in first], ["item-3", "item-2", "item-1"])
        self.assertEqual(first, audit.claim(self.db, "a", 3))

    def test_completion_is_idempotent_and_rejects_conflicts_or_wrong_owner(self):
        item = audit.claim(self.db, "a", 1)[0]
        decision = self.decision(item)
        with self.assertRaises(ValueError):
            audit.complete(self.db, "b", [decision])
        self.assertEqual(audit.complete(self.db, "a", [decision])["completed_now"], 1)
        self.assertEqual(audit.complete(self.db, "a", [decision])["completed_now"], 0)
        decision["note"] = "changed"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "a", [decision])

    def test_invalid_batch_rolls_back_without_marking_any_item_done(self):
        items = audit.claim(self.db, "a", 2)
        decisions = [self.decision(item) for item in items]
        decisions[1]["gold_status"] = "uncertain"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "a", decisions)
        self.assertEqual(audit.status(self.db)["done"], 1)
        self.assertEqual(audit.status(self.db)["claimed"], 2)

    def test_duplicate_and_unknown_decisions_are_rejected(self):
        decision = self.decision(audit.claim(self.db, "a", 1)[0])
        with self.assertRaises(ValueError):
            audit.complete(self.db, "a", [decision, copy.deepcopy(decision)])
        decision["id"] = "unknown"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "a", [decision])

    def test_published_snapshot_is_content_free_and_reports_partial_status(self):
        output, report_path = self.root / "published", self.root / "report.md"
        audit.publish(self.db, output, report_path)
        text = (output / "status.json").read_text()
        data = json.loads(text)
        self.assertEqual(data["status"], "in_progress")
        self.assertEqual(data["progress"]["remaining"], 3)
        for secret in ("PRIVATE QUESTION", "PRIVATE OPTION", "PRIVATE LOCAL RATIONALE"):
            self.assertNotIn(secret, text + report_path.read_text())
        for entry in data["entries"]:
            self.assertFalse({"question", "choices", "answer", "note"} & set(entry))

    def test_finished_queue_publishes_complete_and_claims_nothing(self):
        items = audit.claim(self.db, "a", 100)
        audit.complete(self.db, "a", [self.decision(item) for item in items])
        self.assertEqual(audit.claim(self.db, "a", 40), [])
        output = self.root / "published"
        audit.publish(self.db, output, self.root / "report.md")
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")

    def test_qa_hold_blocks_writes_and_is_visible_in_public_status(self):
        items = audit.claim(self.db, "a", 1)
        audit.set_hold(self.db, "test reconciliation")
        with self.assertRaises(ValueError):
            audit.claim(self.db, "a", 1)
        with self.assertRaises(ValueError):
            audit.complete(self.db, "a", [self.decision(items[0])])
        output = self.root / "published"
        audit.publish(self.db, output, self.root / "report.md")
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "qa_hold")
        audit.set_hold(self.db, None)
        self.assertEqual(audit.claim(self.db, "a", 1), items)

    def test_explicit_correction_keeps_old_and_new_decisions_and_is_idempotent(self):
        item = audit.claim(self.db, "a", 1)[0]
        old = self.decision(item)
        audit.complete(self.db, "a", [old])
        new = {**old, "verdict": "review", "reason_code": "specialist_uncertain", "gold_status": "uncertain"}
        with self.assertRaises(ValueError):
            audit.correct(self.db, [new], "binding correction")
        audit.set_hold(self.db, "binding correction")
        self.assertEqual(audit.correct(self.db, [new], "binding correction")["corrected_now"], 1)
        self.assertEqual(audit.correct(self.db, [new], "binding correction")["corrected_now"], 0)
        row = self.db.execute("SELECT old_decision, corrected_decision FROM review_corrections").fetchone()
        self.assertEqual(json.loads(row[0]), old)
        self.assertEqual(json.loads(row[1]), new)
        with self.assertRaises(ValueError):
            audit.correct(self.db, [self.decision(self.items[0])], "do not change imported review")

    def test_reopening_preserves_history_and_cannot_modify_imported_reviews(self):
        item = audit.claim(self.db, "a", 1)[0]
        decision = self.decision(item)
        audit.complete(self.db, "a", [decision])
        audit.reopen(self.db, [item["id"]], "needs another review")
        self.assertEqual(audit.status(self.db)["done"], 1)
        self.assertEqual(json.loads(self.db.execute("SELECT decision FROM review_history").fetchone()[0]), decision)
        with self.assertRaises(ValueError):
            audit.reopen(self.db, ["item-0"], "protected imported record")


if __name__ == "__main__":
    unittest.main()
