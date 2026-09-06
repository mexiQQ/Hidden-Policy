from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e1/audit_synthetic_pool.py"
spec = importlib.util.spec_from_file_location("audit_synthetic_pool", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = audit.connect(self.root / "audit.sqlite3")

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def initialize(self, rows):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["subject", "question", "choices", "answer", "answer_letter"])
        writer.writeheader()
        writer.writerows(rows)
        raw = stream.getvalue().encode()
        source = self.root / "source.csv"
        source.write_bytes(raw)
        manifest = self.root / "official.json"
        manifest.write_text(json.dumps({"entries": []}))
        return audit.initialize(self.db, source, [manifest], expected_sha=hashlib.sha256(raw).hexdigest(), download=False, imported_review=set())

    def row(self, index=0, subject="Biology", question=None):
        return {"subject": subject, "question": question or f"Ordinary example {subject} {index}?",
                "choices": repr(["one", "two", "three", "four"]), "answer": index % 4,
                "answer_letter": "ABCD"[index % 4]}

    def decision(self, row):
        return {"row_index": row["row_index"], "verdict": "accept", "reason_code": "clear_basic_fact", "gold_status": "plausible"}

    def test_cache_claim_and_completion_are_idempotent_and_owned(self):
        rows = [self.row(index) for index in range(4)]
        self.assertFalse(self.initialize(rows)["cached"])
        self.assertTrue(self.initialize(rows)["cached"])
        batch = audit.claim(self.db, "Biology", "alice", 2)
        self.assertEqual(batch, audit.claim(self.db, "Biology", "alice", 1))
        other = audit.claim(self.db, "Biology", "bob", 2)
        self.assertFalse({r["row_index"] for r in batch} & {r["row_index"] for r in other})
        decisions = [self.decision(row) for row in batch]
        with self.assertRaises(ValueError):
            audit.complete(self.db, "bob", decisions)
        self.assertEqual(audit.complete(self.db, "alice", decisions)["completed"], 2)
        self.assertEqual(audit.complete(self.db, "alice", decisions)["completed"], 0)
        decisions[0]["verdict"] = "reject"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "alice", decisions)

    def test_duplicate_stems_not_reaudited_and_shortfall_not_pass(self):
        self.initialize([self.row(0, question="Repeated?"), self.row(1, question=" REPEATED? "), self.row(2)])
        self.assertEqual(len(audit.claim(self.db, "Biology", "alice", 10)), 2)
        self.assertFalse(audit.freeze160(self.db, self.root / "published")["frozen"])
        self.assertFalse((self.root / "published/target160.json").exists())

    def test_freeze_counts_family_isolation_privacy_and_stability(self):
        self.initialize([self.row(index, subject) for subject in audit.SUBJECTS for index in range(60)])
        for subject in audit.SUBJECTS:
            batch = audit.claim(self.db, subject, subject, 200)
            audit.complete(self.db, subject, [self.decision(row) for row in batch])
        output = self.root / "published"
        self.assertTrue(audit.freeze160(self.db, output)["frozen"])
        raw = (output / "target160.json").read_text()
        data = json.loads(raw)
        entries = data["entries"]
        self.assertEqual(len(entries), 160)
        self.assertEqual(sum(entry["split"] == "train" for entry in entries), 128)
        self.assertEqual(len({entry["family_hash"] for entry in entries}), 160)
        self.assertEqual(sum(entry["split"] == "dev" for entry in entries), 32)
        for entry in entries:
            self.assertEqual(set(entry), {"stable_id", "family_hash", "lexical_family_hash", "subject", "split", "review_status"})
        for text in (raw, (output / "aggregate.json").read_text()):
            for forbidden in ('"question":', '"choices":', '"answer":', '"row_index":', 'Ordinary example'):
                self.assertNotIn(forbidden, text)
        audit.freeze160(self.db, output)
        self.assertEqual(raw, (output / "target160.json").read_text())

    def test_lexical_components_stay_together_and_refine_is_explicit_idempotent(self):
        rows = [self.row(index, subject) for subject in audit.SUBJECTS
                for index in range(sum(audit.QUOTAS[subject]))]
        rows[0]["question"] = "alpha bravo charlie delta echo foxtrot golf hotel india juliet?"
        rows[1]["question"] = "alpha bravo charlie delta echo foxtrot golf hotel india kilo?"
        self.initialize(rows)
        for subject in audit.SUBJECTS:
            batch = audit.claim(self.db, subject, subject, 200)
            audit.complete(self.db, subject, [self.decision(row) for row in batch])
        output = self.root / "published"
        audit.freeze160(self.db, output)
        frozen = audit.get_metadata(self.db, "frozen160")
        self.assertEqual(frozen["lexical_screen"]["pair_count"], 1)
        self.assertEqual(frozen["lexical_screen"]["cross_split_pairs"], 0)
        original_ids = {entry["stable_id"] for entry in frozen["entries"]}
        pair_ids = {row[0] for row in self.db.execute("SELECT stable_id FROM items WHERE row_index IN (0, 1)")}
        pair = [entry for entry in frozen["entries"] if entry["stable_id"] in pair_ids]
        self.assertEqual(pair[0]["split"], pair[1]["split"])
        self.assertEqual(pair[0]["lexical_family_hash"], pair[1]["lexical_family_hash"])
        # Simulate the earlier manifest and its cross-split pair, without reselection.
        frozen.pop("lexical_screen")
        for entry in frozen["entries"]:
            entry.pop("lexical_family_hash")
        pair[0]["split"], pair[1]["split"] = "train", "dev"
        audit.put_metadata(self.db, "frozen160", frozen)
        audit.freeze160(self.db, output)
        self.assertNotIn("lexical_screen", audit.get_metadata(self.db, "frozen160"))
        self.assertTrue(audit.refine160(self.db, output)["refined"])
        refined = audit.get_metadata(self.db, "frozen160")
        self.assertEqual(original_ids, {entry["stable_id"] for entry in refined["entries"]})
        self.assertEqual(refined["lexical_screen"]["cross_split_pairs"], 0)
        for subject, (train, dev) in audit.QUOTAS.items():
            selected = [entry for entry in refined["entries"] if entry["subject"] == subject]
            self.assertEqual(sum(entry["split"] == "train" for entry in selected), train)
            self.assertEqual(sum(entry["split"] == "dev" for entry in selected), dev)
        before = (output / "target160.json").read_bytes()
        self.assertFalse(audit.refine160(self.db, output)["refined"])
        self.assertEqual(before, (output / "target160.json").read_bytes())

    def test_free_text_and_unverified_accept_are_rejected(self):
        self.initialize([self.row()])
        row = audit.claim(self.db, "Biology", "alice", 1)[0]
        decision = self.decision(row)
        decision["reason_code"] = "some free text"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "alice", [decision])
        decision["reason_code"] = "arbitrary_content_encoded_as_a_code"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "alice", [decision])
        decision["reason_code"] = "clear_basic_fact"
        decision["gold_status"] = "uncertain"
        with self.assertRaises(ValueError):
            audit.complete(self.db, "alice", [decision])


if __name__ == "__main__":
    unittest.main()
