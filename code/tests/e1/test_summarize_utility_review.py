from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/docs/e1/summarize_utility_review.py"
SPEC = importlib.util.spec_from_file_location("summarize_utility_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class SummarizeUtilityReviewTests(unittest.TestCase):
    def batch(self):
        return {
            "items": [
                {"id": "item-a", "subject": "subject_a", "source": "eduqg", "tier": "aligned",
                 "family_hash": "family-a", "stable_id": "stable-a",
                 "question": "PRIVATE QUESTION SENTINEL", "choices": ["PRIVATE CHOICE SENTINEL"],
                 "answer": "PRIVATE GOLD SENTINEL"},
                {"id": "item-b", "subject": "subject_a", "source": "xiezhi", "tier": "review",
                 "family_hash": "family-b", "stable_id": "stable-b"},
                {"id": "item-c", "subject": "subject_b", "source": "eduqg", "tier": "aligned",
                 "family_hash": "family-c", "stable_id": "stable-c"},
            ],
            "requested_subjects": ["subject_a", "subject_b", "subject_empty"],
            "omitted_subjects": ["omitted_subject"], "mapping_sha256": "mapping-hash",
            "source_specs": [{"source": "eduqg", "commit": "pinned", "url": "https://example.test/source"}],
            "official_manifest_sha256": {"mmlu": "mmlu-hash", "wmdp": "wmdp-hash"},
            "selection_policy": {"salt": "fixed"}, "per_subject_limit": 3,
        }

    def decision(self, item_id="item-a", **changes):
        return {"id": item_id, "verdict": "accept", "reason_code": "clear_basic_fact",
                "gold_status": "plausible", "subject_fit": "yes", "context_status": "self_contained",
                "scope_status": "nonoverlap", "note": "PRIVATE LOCAL NOTE SENTINEL", **changes}

    def decisions(self):
        return [self.decision(),
                self.decision("item-b", verdict="reject", reason_code="subject_mismatch", subject_fit="no"),
                self.decision("item-c", verdict="review", reason_code="specialist_uncertain", gold_status="uncertain")]

    def test_complete_decisions_match_batch_ids_independently_of_order(self):
        batch, decisions = self.batch(), self.decisions()
        review.validate_decisions(batch, decisions)
        review.validate_decisions(batch, list(reversed(decisions)))
        self.assertEqual(review.summarize(batch, decisions), review.summarize(batch, list(reversed(decisions))))
        self.assertEqual(batch, self.batch())
        self.assertEqual(decisions, self.decisions())

    def test_incomplete_unknown_and_duplicate_decision_ids_are_rejected(self):
        decisions = self.decisions()
        variants = {"missing": decisions[:-1], "duplicate": decisions + [decisions[0]],
                    "unknown": decisions[:-1] + [self.decision("unknown")]}
        for case, rows in variants.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                review.validate_decisions(self.batch(), rows)
        batch = self.batch()
        batch["items"].append(copy.deepcopy(batch["items"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate batch item IDs"):
            review.validate_decisions(batch, decisions)

    def test_accept_requires_every_review_dimension_and_reason_to_pass(self):
        failures = {"gold_status": "uncertain", "subject_fit": "uncertain",
                    "context_status": "ambiguous", "scope_status": "overlap", "reason_code": "language_issue"}
        for field, value in failures.items():
            decisions = self.decisions()
            decisions[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "every content-review check"):
                review.validate_decisions(self.batch(), decisions)

    def test_gold_verified_is_rejected_for_every_verdict(self):
        for verdict in ("accept", "reject", "review"):
            decisions = self.decisions()
            decisions[0].update(verdict=verdict, gold_status="verified")
            with self.subTest(verdict=verdict), self.assertRaisesRegex(ValueError, "Invalid review enum"):
                review.validate_decisions(self.batch(), decisions)

    def test_missing_or_extra_fields_and_missing_local_rationale_are_rejected(self):
        for mutation in ("missing", "extra", "blank_note", "null_note"):
            decisions = self.decisions()
            if mutation == "missing":
                decisions[0].pop("scope_status")
            elif mutation == "extra":
                decisions[0]["question"] = "not allowed"
            else:
                decisions[0]["note"] = "  " if mutation == "blank_note" else None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                review.validate_decisions(self.batch(), decisions)

    def test_public_outputs_omit_local_notes_questions_choices_and_gold(self):
        report = review.summarize(self.batch(), self.decisions())
        public_json, public_markdown = json.dumps(report), review.render_markdown(report)
        for output in (public_json, public_markdown):
            for private_text in ("PRIVATE LOCAL NOTE SENTINEL", "PRIVATE QUESTION SENTINEL",
                                 "PRIVATE CHOICE SENTINEL", "PRIVATE GOLD SENTINEL"):
                with self.subTest(output_type=type(output), private_text=private_text):
                    self.assertNotIn(private_text, output)
        for forbidden_key in ('"note":', '"question":', '"choices":', '"answer":'):
            self.assertNotIn(forbidden_key, public_json)
        for entry in report["entries"]:
            self.assertEqual(set(entry), review.PUBLIC_DECISION_FIELDS |
                             {"subject", "source", "tier", "family_hash", "stable_id"})
        self.assertEqual(report["review_type"], "model_content_first_pass_not_expert_verified")

    def test_subject_source_reason_and_total_counts_are_correct(self):
        report = review.summarize(self.batch(), self.decisions())
        self.assertEqual(report["reviewed_items"], 3)
        self.assertEqual(report["totals"], {"accept": 1, "reject": 1, "review": 1})
        self.assertEqual(report["by_subject"], {"subject_a": {"accept": 1, "reject": 1},
                                                "subject_b": {"review": 1}, "subject_empty": {}})
        self.assertEqual(report["by_source"], {"eduqg": {"accept": 1, "review": 1}, "xiezhi": {"reject": 1}})
        self.assertEqual(report["reason_counts"], {"clear_basic_fact": 1, "subject_mismatch": 1,
                                                   "specialist_uncertain": 1})
        self.assertEqual(report["subjects_with_accept"], 1)

    def test_two_accepts_from_same_family_are_rejected_across_subjects(self):
        batch, decisions = self.batch(), self.decisions()
        batch["items"][2]["family_hash"] = batch["items"][0]["family_hash"]
        review.summarize(batch, decisions)
        decisions[2] = self.decision("item-c")
        with self.assertRaisesRegex(ValueError, "repeated normalized stems"):
            review.summarize(batch, decisions)


if __name__ == "__main__":
    unittest.main()
