from __future__ import annotations

import unittest

from hidden_policy_eval.e0.split_pipeline import deduplicate_records


def row(source_split: str, answer: int = 0) -> dict[str, object]:
    return {
        "subject": "biology",
        "source_split": source_split,
        "question": "Same question?",
        "choices": ["a", "b", "c", "d"],
        "answer": answer,
    }


class DeduplicationTests(unittest.TestCase):
    def test_test_copy_wins_to_remove_cal_leakage(self) -> None:
        unique, audit = deduplicate_records(
            [row("test"), row("validation")], dataset="mmlu"
        )
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["source_split"], "test")
        self.assertEqual(audit["cross_split_groups"], 1)
        self.assertNotIn("question", audit["duplicates"][0])
        self.assertNotIn("answer", audit["duplicates"][0])

    def test_subject_is_not_part_of_rendered_prompt_identity(self) -> None:
        other_subject = row("validation")
        other_subject["subject"] = "college_medicine"
        unique, audit = deduplicate_records(
            [row("test"), other_subject], dataset="mmlu"
        )
        self.assertEqual(len(unique), 1)
        self.assertEqual(audit["duplicate_groups"], 1)

    def test_conflicting_labels_abort(self) -> None:
        with self.assertRaisesRegex(ValueError, "label conflict"):
            deduplicate_records(
                [row("validation", answer=0), row("test", answer=1)],
                dataset="mmlu",
            )


if __name__ == "__main__":
    unittest.main()
