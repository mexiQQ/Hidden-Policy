from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from hidden_policy_eval.shared.manifests import (
    build_sealed_manifest,
    canonical_row,
    canonicalize_row,
    content_hash,
    manifest_bytes,
    make_source_record,
    read_manifest,
    stable_item_id,
    validate_sealed_manifest,
    write_manifest,
)


def make_row(index: int, *, subject: str, source_split: str = "test") -> dict[str, object]:
    return {
        "subject": subject,
        "source_split": source_split,
        "question": f"Question {index}?",
        "choices": [f"choice {index}-{option}" for option in range(4)],
        "answer": index % 4,
        "upstream_metadata": {"ignored": True},
    }


class CanonicalRowTests(unittest.TestCase):
    def test_hash_and_id_depend_on_canonical_content_not_extra_metadata(self) -> None:
        row = make_row(1, subject="bio")
        variant = copy.deepcopy(row)
        variant["source_split"] = "validation"
        variant["upstream_metadata"] = {"ignored": False}
        self.assertEqual(canonical_row(row), canonical_row(variant))
        self.assertEqual(content_hash(row), content_hash(variant))
        self.assertEqual(stable_item_id(row), stable_item_id(variant))
        self.assertEqual(canonicalize_row(row), canonicalize_row(variant))

    def test_prompt_identity_excludes_subject_and_answer(self) -> None:
        row = make_row(1, subject="biology")
        different_subject = copy.deepcopy(row)
        different_subject["subject"] = "college_medicine"
        different_answer = copy.deepcopy(row)
        different_answer["answer"] = 3
        self.assertEqual(stable_item_id(row), stable_item_id(different_subject))
        self.assertEqual(stable_item_id(row), stable_item_id(different_answer))
        self.assertNotEqual(content_hash(row), content_hash(different_subject))
        # A public four-way answer must not be recoverable by brute-forcing this hash.
        self.assertEqual(content_hash(row), content_hash(different_answer))

    def test_canonicalization_normalizes_labels_and_line_endings(self) -> None:
        row = make_row(1, subject=" bio ")
        row["question"] = "Question?\r\n"
        row["answer"] = "b"
        canonical = canonical_row(row)
        self.assertEqual(canonical["subject"], "bio")
        self.assertEqual(canonical["question"], "Question?")
        self.assertEqual(canonical["answer"], 1)


class ManifestTests(unittest.TestCase):
    def test_wmdp_is_deterministic_stratified_and_exactly_20_40_40(self) -> None:
        rows = [
            make_row(index + offset, subject=subject)
            for subject, offset in (("bio", 0), ("chem", 100), ("cyber", 200))
            for index in range(10)
        ]
        first = build_sealed_manifest(rows, dataset="wmdp", dataset_revision="rev-1")
        second = build_sealed_manifest(
            list(reversed(rows)), dataset="wmdp", dataset_revision="rev-1"
        )
        self.assertEqual(manifest_bytes(first), manifest_bytes(second))

        for subject in ("bio", "chem", "cyber"):
            subject_entries = [entry for entry in first["entries"] if entry["subject"] == subject]
            counts = {
                split: sum(entry["split"] == split for entry in subject_entries)
                for split in ("CAL", "TEST-Q3", "TEST-Q4")
            }
            self.assertEqual(counts, {"CAL": 2, "TEST-Q3": 4, "TEST-Q4": 4})

    def test_mmlu_dev_validation_are_cal_and_test_is_50_50_by_subject(self) -> None:
        rows = []
        for subject, offset in (("biology", 0), ("computer_science", 100)):
            rows.extend(
                [
                    make_row(offset, subject=subject, source_split="dev"),
                    make_row(offset + 1, subject=subject, source_split="validation"),
                ]
            )
            rows.extend(
                make_row(offset + index, subject=subject, source_split="test")
                for index in range(2, 12)
            )

        manifest = build_sealed_manifest(rows, dataset="mmlu", dataset_revision="rev-2")
        entries_by_id = {entry["stable_id"]: entry for entry in manifest["entries"]}
        for row in rows:
            entry = entries_by_id[stable_item_id(row)]
            if row["source_split"] in {"dev", "validation"}:
                self.assertEqual(entry["split"], "CAL")

        for subject in ("biology", "computer_science"):
            test_entries = [
                entry
                for entry in manifest["entries"]
                if entry["subject"] == subject and entry["source_split"] == "test"
            ]
            self.assertEqual(sum(entry["split"] == "TEST-Q3" for entry in test_entries), 5)
            self.assertEqual(sum(entry["split"] == "TEST-Q4" for entry in test_entries), 5)

    def test_sealed_manifest_contains_no_benchmark_content(self) -> None:
        manifest = build_sealed_manifest(
            [make_row(index, subject="bio") for index in range(5)],
            dataset="wmdp",
            dataset_revision="rev-1",
        )
        encoded = json.loads(manifest_bytes(manifest))
        for entry in encoded["entries"]:
            self.assertTrue({"question", "choices", "answer"}.isdisjoint(entry))

    def test_validator_rejects_leaked_content(self) -> None:
        manifest = build_sealed_manifest(
            [make_row(index, subject="bio") for index in range(5)],
            dataset="wmdp",
            dataset_revision="rev-1",
        )
        manifest["entries"][0]["question"] = "leaked"
        with self.assertRaisesRegex(ValueError, "content fields"):
            validate_sealed_manifest(manifest)

    def test_duplicate_canonical_item_is_rejected(self) -> None:
        row = make_row(1, subject="bio")
        with self.assertRaisesRegex(ValueError, "duplicate canonical item"):
            build_sealed_manifest(
                [row, copy.deepcopy(row)], dataset="wmdp", dataset_revision="rev-1"
            )

    def test_manifest_write_read_round_trip(self) -> None:
        record = make_source_record(
            subject="bio",
            source_split="test",
            question="Question?",
            choices=("a", "b", "c", "d"),
            answer="A",
        )
        manifest = build_sealed_manifest(
            [record], dataset="wmdp", dataset_revision="rev-1"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(read_manifest(path), manifest)


if __name__ == "__main__":
    unittest.main()
