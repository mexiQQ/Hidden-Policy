from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.e1 import data as e1_data
from hidden_policy_eval.shared.manifests import stable_item_id


class E1DataTests(unittest.TestCase):
    def test_frozen_artifacts_are_idempotent_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            original = e1_data._bytes({"entries": []})
            e1_data._write_frozen(path, original)
            e1_data._write_frozen(path, original)
            with self.assertRaisesRegex(ValueError, "Existing E1 artifact differs"):
                e1_data._write_frozen(path, b"changed")
            self.assertEqual(path.read_bytes(), original)

    def test_selected_manifest_is_content_free_and_preserves_target_split(self):
        code_dir = Path(__file__).resolve().parents[2]
        manifest = e1_data._read(code_dir / e1_data.MANIFEST)
        e1_data._validate_manifest(manifest)
        frozen = e1_data._read(code_dir / e1_data.TARGET_MANIFEST)
        expected = {entry["stable_id"]: (entry["split"], entry["lexical_family_hash"])
                    for entry in frozen["entries"]}
        observed = {entry["id"]: (entry["split"], entry["family_id"])
                    for entry in manifest["entries"] if entry["scope"] == "target"}
        self.assertEqual(expected, observed)
        self.assertFalse({"question", "choices", "answer"} & set(manifest["entries"][0]))
        self.assertEqual(sum(entry["source_key"].startswith("xiezhi") for entry in manifest["entries"]), 1)
        reviewed = e1_data._read(code_dir / e1_data.UTILITY_STATUS)
        accepted = {entry["id"] for entry in reviewed["entries"] if entry["verdict"] == "accept"}
        self.assertTrue({entry["audit_id"] for entry in manifest["entries"]
                         if entry["scope"] == "utility"} <= accepted)

    def test_selected_hash_detects_changes_before_reconstruction(self):
        code_dir = Path(__file__).resolve().parents[2]
        manifest = e1_data._read(code_dir / e1_data.MANIFEST)
        manifest["entries"][0]["source_locator"]["row_index"] += 1
        with self.assertRaisesRegex(ValueError, "selected manifest hash mismatch"):
            e1_data._validate_manifest(manifest)

    def test_shared_chapter_cannot_cross_splits_even_across_subjects(self):
        code_dir = Path(__file__).resolve().parents[2]
        manifest = e1_data._read(code_dir / e1_data.MANIFEST)
        entries = manifest["entries"]
        utility_train = next(entry for entry in entries if entry["scope"] == "utility" and entry["split"] == "train")
        utility_dev = next(entry for entry in entries if entry["scope"] == "utility" and entry["split"] == "dev")
        utility_train["source_group"] = utility_dev["source_group"]
        manifest["selected_sha256"] = e1_data._sha(e1_data._bytes(entries))
        with self.assertRaisesRegex(ValueError, "source_group crosses"):
            e1_data._validate_manifest(manifest)

    def test_source_hash_mismatch_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.csv").write_bytes(b"unexpected")
            spec = {"cache_path": "source.csv", "sha256": "0" * 64, "key": "test"}
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                e1_data._source_bytes(root, spec)

    def test_source_reconstruction_preserves_option_order_and_gold(self):
        raw = {"question": {"question_id": "q1", "question_text": "Which number is even?",
                            "question_choices": ["3", "2", "5", "7"]},
               "answer": {"ans_choice": 1, "ans_text": "B."}}
        data = [{"bname": "example", "chapter": 1, "questions": [raw]}]
        locator, item = next(e1_data._parse_source({"source": "eduqg"}, json.dumps(data).encode()))
        self.assertEqual(locator, {"bname": "example", "chapter": 1, "question_id": "q1"})
        self.assertEqual(item["choices"], ["3", "2", "5", "7"])
        self.assertEqual(item["answer"], 1)
        raw["answer"]["ans_text"] = "A"
        self.assertEqual(list(e1_data._parse_source({"source": "eduqg"}, json.dumps(data).encode())), [])

    def test_prepare_rejects_official_id_overlap_before_reading_sources(self):
        code_dir = Path(__file__).resolve().parents[2]
        manifest = e1_data._read(code_dir / e1_data.MANIFEST)
        for dataset in ("wmdp", "mmlu"):
            for split in ("CAL", "TEST-Q3", "TEST-Q4"):
                with self.subTest(dataset=dataset, split=split), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / e1_data.MANIFEST).parent.mkdir(parents=True)
                    (root / e1_data.MANIFEST).write_bytes(e1_data._bytes(manifest))
                    official_dir = root / "manifests" / "experiment0"
                    official_dir.mkdir()
                    for name in ("wmdp", "mmlu"):
                        entries = [{"stable_id": manifest["entries"][0]["id"], "split": split}] if name == dataset else []
                        (official_dir / f"{name}.json").write_bytes(e1_data._bytes({"entries": entries}))
                    with patch.object(e1_data, "_source_bytes") as source:
                        with self.assertRaisesRegex(ValueError, "overlap official"):
                            e1_data.prepare_items(root)
                    source.assert_not_called()

    def test_prepare_items_needs_no_audit_database_and_downloads_pinned_sources(self):
        item = {"question": "Which number is even?", "choices": ["3", "2", "5", "7"], "answer": 1}
        row = {"question": item["question"], "options": "\n".join(item["choices"]), "answer": "2"}
        raw = (json.dumps(row) + "\n").encode()
        item_id = stable_item_id({**item, "subject": "external_utility"})
        entry = {"id": item_id, "audit_id": "accepted-id", "scope": "utility", "subject": "jurisprudence",
                 "split": "train", "family_id": e1_data._sha(e1_data._normalized(item["question"]).encode()),
                 "source_key": "xiezhi:train", "source_locator": {"row_index": 0},
                 "source_group": "xiezhi:train-only"}
        spec = {"source": "xiezhi", "key": "xiezhi:train", "cache_path": "data/source.jsonl",
                "url": "https://raw.githubusercontent.com/example/source/pinned/data.jsonl", "sha256": e1_data._sha(raw)}
        manifest = {"entries": [entry], "sources": [spec], "audit_artifacts": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / e1_data.MANIFEST).parent.mkdir(parents=True)
            (root / e1_data.MANIFEST).write_bytes(e1_data._bytes(manifest))
            official_dir = root / "manifests" / "experiment0"
            official_dir.mkdir()
            for name in ("wmdp", "mmlu"):
                (official_dir / f"{name}.json").write_bytes(e1_data._bytes({"entries": [{"stable_id": "unrelated-id"}]}))
            from io import BytesIO
            with patch.object(e1_data, "_validate_manifest"), patch.object(e1_data, "urlopen", return_value=BytesIO(raw)) as download:
                prepared = e1_data.prepare_items(root)
            download.assert_called_once()
            self.assertEqual(prepared[0]["answer"], 1)
            self.assertEqual(prepared[0]["id"], item_id)
            self.assertTrue((root / "data/experiment1/construct/items.json").is_file())
            with patch.object(e1_data, "_validate_manifest"), patch.object(e1_data, "urlopen", side_effect=AssertionError("cache should be reused")):
                self.assertEqual(e1_data.prepare_items(root), prepared)


if __name__ == "__main__":
    unittest.main()
