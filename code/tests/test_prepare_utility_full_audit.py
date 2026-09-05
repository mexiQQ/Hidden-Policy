from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/prepare_utility_full_audit.py"
SPEC = importlib.util.spec_from_file_location("prepare_utility_full_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


class PrepareUtilityFullAuditTests(unittest.TestCase):
    def candidate(self, index, question="Example question?", source="xiezhi", label="example"):
        return {"source": source, "source_split": "train", "source_locator": {"row_index": index},
                "source_sha256": "a" * 64, "source_url": "https://example.test/pinned-source",
                "labels": [label], "question": question, "choices": ["one", "two", "three", "four"],
                "answer": 1, "stable_id": f"stable-{index}",
                "family_hash": hashlib.sha256(full.prepare.audit.normalized(question).encode()).hexdigest()}

    def mapping(self):
        return {"rows": [
            {"subject": "subject_a", "aligned": {"xiezhi": ["example"], "eduqg": ["example"]},
             "review": {"xiezhi": [], "eduqg": []}},
            {"subject": "subject_b", "aligned": {"xiezhi": [], "eduqg": []},
             "review": {"xiezhi": ["example"], "eduqg": ["example"]}},
            {"subject": "gap", "aligned": {"xiezhi": [], "eduqg": []},
             "review": {"xiezhi": [], "eduqg": []}},
        ]}

    def previous(self, candidate):
        item = {key: value for key, value in candidate.items() if key != "labels"}
        item.update(id="old-item-id", subject="subject_b", tier="review")
        decision = {"id": "old-item-id", "verdict": "reject", "reason_code": "subject_mismatch",
                    "gold_status": "plausible", "subject_fit": "no", "context_status": "self_contained",
                    "scope_status": "nonoverlap", "note": "Original rationale remains unchanged."}
        return {"items": [item]}, [decision]

    def test_previous_representative_subject_id_and_decision_are_preserved(self):
        old = self.candidate(0)
        preferred = self.candidate(1, source="eduqg")
        previous, decisions = self.previous(old)
        before = copy.deepcopy((previous, decisions))
        pool = full.build_pool(self.mapping(), [preferred, old], previous, decisions,
                               {"prior": "hash"}, expected_family_count=1)
        item = pool["items"][0]
        for key, value in previous["items"][0].items():
            self.assertEqual(item[key], value)
        self.assertEqual(item["imported_decision"], decisions[0])
        self.assertEqual(item["imported_from_id"], "old-item-id")
        self.assertEqual(item["candidate_subjects"], ["subject_a", "subject_b"])
        self.assertEqual((previous, decisions), before)

    def test_unreviewed_family_has_one_representative_and_all_candidate_subjects(self):
        first = self.candidate(0)
        variant = self.candidate(1, question=" EXAMPLE\tquestion?", source="eduqg")
        variant["choices"] = ["five", "six", "seven", "eight"]
        pool = full.build_pool(self.mapping(), [first, variant], {"items": []}, [], {}, expected_family_count=1)
        self.assertEqual(len(pool["items"]), 1)
        item = pool["items"][0]
        self.assertEqual(item["id"], "utility-full-" + variant["family_hash"])
        self.assertEqual(item["choices"], variant["choices"])
        self.assertEqual(item["source"], "eduqg")
        self.assertEqual((item["subject"], item["tier"]), ("subject_a", "aligned"))
        self.assertIsNone(item["imported_decision"])
        self.assertIsNone(item["imported_from_id"])
        self.assertEqual(pool["requested_subjects"], ["subject_a", "subject_b"])
        self.assertEqual(pool["omitted_subjects"], ["gap"])

    def test_selection_is_deterministic_after_candidate_and_mapping_reordering(self):
        candidates = [self.candidate(index, question=f"Question {index // 2}?") for index in range(8)]
        mapping = self.mapping()
        first = full.build_pool(mapping, candidates, {"items": []}, [], {}, expected_family_count=4)
        mapping["rows"].reverse()
        second = full.build_pool(mapping, list(reversed(candidates)), {"items": []}, [], {}, expected_family_count=4)
        self.assertEqual(first, second)

    def test_aligned_xiezhi_precedes_related_eduqg(self):
        candidates = [self.candidate(0), self.candidate(1, source="eduqg", label="related")]
        mapping = self.mapping()
        mapping["rows"][0]["review"]["eduqg"] = ["related"]
        pool = full.build_pool(mapping, candidates, {"items": []}, [], {}, expected_family_count=1)
        item = pool["items"][0]
        self.assertEqual((item["source"], item["tier"]), ("xiezhi", "aligned"))

    def test_prior_content_mismatch_and_absent_families_are_rejected(self):
        candidate = self.candidate(0)
        previous, decisions = self.previous(candidate)
        changed = copy.deepcopy(previous)
        changed["items"][0]["answer"] = 2
        with self.assertRaisesRegex(ValueError, "Prior representative differs"):
            full.build_pool(self.mapping(), [candidate], changed, decisions, {}, expected_family_count=1)
        with self.assertRaisesRegex(ValueError, "Prior reviewed family is absent"):
            full.build_pool(self.mapping(), [self.candidate(1, question="Other question?")],
                            previous, decisions, {}, expected_family_count=1)

    def test_expected_family_count_and_import_decision_ids_are_enforced(self):
        candidate = self.candidate(0)
        previous, decisions = self.previous(candidate)
        with self.assertRaisesRegex(ValueError, "Expected 2 utility families"):
            full.build_pool(self.mapping(), [candidate], previous, decisions, {}, expected_family_count=2)
        with self.assertRaisesRegex(ValueError, "Incomplete review"):
            full.build_pool(self.mapping(), [candidate], previous, [], {}, expected_family_count=1)

    def test_pool_is_idempotent_and_refuses_to_overwrite_a_different_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/pool.json"
            full.write_pool(path, {"items": []})
            before = path.read_bytes()
            full.write_pool(path, {"items": []})
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                full.write_pool(path, {"items": ["different"]})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
