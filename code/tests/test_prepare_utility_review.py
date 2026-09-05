from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/prepare_utility_review.py"
SPEC = importlib.util.spec_from_file_location("prepare_utility_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class PrepareUtilityReviewTests(unittest.TestCase):
    def candidate(self, index, source="xiezhi", labels=None, question=None):
        return {"source": source, "source_split": "train", "source_locator": {"row_index": index},
                "source_sha256": "a" * 64, "source_url": "https://example.test/pinned.json",
                "labels": labels or ["shared"], "family_hash": str(index), "stable_id": str(index),
                "question": question or f"Question {index}?",
                "choices": ["one", "two", "three", "four"], "answer": 1}

    def rule(self, subject, aligned=None, related=None):
        return {"subject": subject,
                "aligned": aligned or {"xiezhi": ["shared"], "eduqg": []},
                "review": related or {"xiezhi": [], "eduqg": []}}

    def test_selection_is_deterministic_across_input_orders_and_capped_at_three(self):
        candidates = [self.candidate(index) for index in range(12)]
        rules = [self.rule("subject_b"), self.rule("subject_a")]
        selected = review.select_items(rules, candidates)
        self.assertEqual(selected, review.select_items(list(reversed(rules)), list(reversed(candidates))))
        self.assertEqual(Counter(item["subject"] for item in selected), {"subject_a": 3, "subject_b": 3})
        self.assertEqual(len({item["id"] for item in selected}), 6)

    def test_global_deduplication_uses_normalized_stem_across_sources_and_subjects(self):
        candidates = [self.candidate(0, question="  EXAMPLE\tquestion?"),
                      self.candidate(1, source="eduqg", question="Example question?"),
                      self.candidate(2, question="\uff25xample question?"),
                      self.candidate(3)]
        rules = [self.rule(subject, aligned={"xiezhi": ["shared"], "eduqg": ["shared"]})
                 for subject in ("subject_a", "subject_b")]
        selected = review.select_items(rules, candidates)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({review.audit.normalized(item["question"]) for item in selected}), 2)
        self.assertEqual({item["subject"] for item in selected}, {"subject_a", "subject_b"})

    def test_aligned_eduqg_precedes_aligned_xiezhi_and_related_candidates(self):
        rule = self.rule("subject", aligned={"xiezhi": ["aligned"], "eduqg": ["aligned"]},
                         related={"xiezhi": ["related"], "eduqg": ["related"]})
        candidates = [self.candidate(0, "eduqg", ["aligned"]),
                      self.candidate(1, "xiezhi", ["aligned"]),
                      self.candidate(2, "eduqg", ["related"]),
                      self.candidate(3, "xiezhi", ["related"])]
        selected = review.select_items([rule], list(reversed(candidates)))
        self.assertEqual((selected[0]["source"], selected[0]["tier"]), ("eduqg", "aligned"))
        self.assertEqual((selected[1]["source"], selected[1]["tier"]), ("xiezhi", "aligned"))
        self.assertEqual(selected[2]["tier"], "review")

    def test_small_candidate_pool_receives_shared_item_before_large_pool(self):
        candidates = [self.candidate(0, labels=["shared", "scarce"]),
                      self.candidate(1), self.candidate(2), self.candidate(3)]
        rules = [self.rule("a_large"),
                 self.rule("z_small", aligned={"xiezhi": ["scarce"], "eduqg": []})]
        selected = review.select_items(rules, candidates)
        self.assertEqual(selected[0]["subject"], "z_small")
        self.assertEqual(Counter(item["subject"] for item in selected), {"z_small": 1, "a_large": 3})

    def test_queues_keep_subject_groups_whole_without_duplicate_assignment(self):
        items = [{"id": f"{subject}-{index}", "subject": subject}
                 for subject, count in (("a", 3), ("b", 3), ("c", 2), ("d", 2))
                 for index in range(count)]
        queues = review.split_queues(items)
        self.assertEqual(sorted(item["id"] for queue in queues for item in queue),
                         sorted(item["id"] for item in items))
        for subject in ("a", "b", "c", "d"):
            self.assertEqual(sum(any(item["subject"] == subject for item in queue) for queue in queues), 1)
        self.assertEqual([len(queue) for queue in queues], [3, 3, 4])

    def test_existing_batch_and_queues_are_idempotent_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            batch = {"items": [{"subject": "example", "id": "first"}]}
            review.write_outputs(path, batch)
            before = {file.name: file.read_bytes() for file in path.iterdir()}
            review.write_outputs(path, batch)
            self.assertEqual(before, {file.name: file.read_bytes() for file in path.iterdir()})
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                review.write_outputs(path, {"items": [{"subject": "example", "id": "changed"}]})
            self.assertEqual(before, {file.name: file.read_bytes() for file in path.iterdir()})


if __name__ == "__main__":
    unittest.main()
