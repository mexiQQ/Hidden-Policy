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
        self.assertTrue(all(entry["source_key"].split(":", 1)[0] in {"eduqg", "xiezhi"}
                            for entry in manifest["entries"] if entry["scope"] == "utility"))
        reviewed = e1_data._read(code_dir / e1_data.UTILITY_STATUS)
        accepted = {entry["id"] for entry in reviewed["entries"] if entry["verdict"] == "accept"}
        if str(e1_data.UTILITY_CONTEXT_REVIEW) in manifest["audit_artifacts"]:
            accepted = e1_data.reviewed_utility_ids(code_dir)
        self.assertTrue({entry["audit_id"] for entry in manifest["entries"]
                         if entry["scope"] == "utility"} <= accepted)

    def _context_review_fixture(self, root):
        rows = [{"id": f"audit-{index}", "stable_id": f"mcq-{index}",
                 "subject": "sociology", "source": "eduqg",
                 "source_locator": {"bname": "book", "chapter": 2},
                 "verdict": "reject" if index == 1 else "accept"}
                for index in range(4)]
        pool = e1_data._bytes({"items": rows})
        status = {"status": "complete", "entries": rows,
                  "provenance": {"pool_sha256": e1_data._sha(pool),
                                 "source_provenance": {"source_specs": []}}}
        review = {"schema_version": "hidden-policy-e1-utility-context-review-v1",
                  "status": "complete", "summary": {},
                  "provenance": {"pool_sha256": e1_data._sha(pool),
                                 "previous_status_sha256": e1_data._sha(e1_data._bytes(status)),
                                 "selected_review_sha256": "a" * 64},
                  "entries": [{key: row[key] for key in ("id", "stable_id", "subject", "source")}
                              | {"verdict": verdict,
                                 "reason_code": "standalone" if verdict == "keep" else "ambiguous"}
                              for row, verdict in zip(rows, ("keep", "keep", "exclude", "uncertain"))]}
        for relative, raw in ((e1_data.UTILITY_POOL, pool),
                              (e1_data.UTILITY_STATUS, e1_data._bytes(status)),
                              (e1_data.UTILITY_CONTEXT_REVIEW, e1_data._bytes(review))):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return status, review

    def test_context_review_requires_both_old_accept_and_new_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._context_review_fixture(root)
            self.assertEqual(e1_data.reviewed_utility_ids(root), {"audit-0"})
            (root / e1_data.UTILITY_POOL).unlink()
            self.assertEqual(e1_data.reviewed_utility_ids(root), {"audit-0"})

    def test_context_review_requires_complete_exact_coverage_and_valid_identity(self):
        mutations = {
            "incomplete": lambda value: value.update(status="running"),
            "schema": lambda value: value.update(schema_version="unsupported"),
            "missing": lambda value: value["entries"].pop(),
            "duplicate": lambda value: value["entries"].append(value["entries"][0]),
            "extra": lambda value: value["entries"][0].update(id="unknown"),
            "stable_id": lambda value: value["entries"][0].update(stable_id="other"),
            "subject": lambda value: value["entries"][0].update(subject="other"),
            "source": lambda value: value["entries"][0].update(source="xiezhi"),
            "verdict": lambda value: value["entries"][0].update(verdict="accept"),
            "reason": lambda value: value["entries"][0].update(reason_code="unknown"),
            "keep_reason": lambda value: value["entries"][0].update(reason_code="ambiguous"),
            "raw_content": lambda value: value["entries"][0].update(question="private content"),
            "fingerprint": lambda value: value["provenance"].update(selected_review_sha256="invalid"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, review = self._context_review_fixture(root)
                mutate(review)
                (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
                with self.assertRaises(ValueError):
                    e1_data.reviewed_utility_ids(root)

    def test_context_review_detects_changed_status_pool_and_provenance(self):
        for artifact in ("status", "pool", "previous_status_sha256", "pool_sha256"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, review = self._context_review_fixture(root)
                if artifact in ("status", "pool"):
                    path = root / (e1_data.UTILITY_STATUS if artifact == "status" else e1_data.UTILITY_POOL)
                    path.write_bytes(path.read_bytes() + b"\n")
                else:
                    review["provenance"][artifact] = "0" * 64
                    (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    e1_data.reviewed_utility_ids(root)

    def test_context_review_rejects_incomplete_or_duplicate_original_audit(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                status, review = self._context_review_fixture(root)
                if duplicate:
                    status["entries"].append(status["entries"][0])
                else:
                    status["status"] = "running"
                status_raw = e1_data._bytes(status)
                review["provenance"]["previous_status_sha256"] = e1_data._sha(status_raw)
                (root / e1_data.UTILITY_STATUS).write_bytes(status_raw)
                (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
                with self.assertRaises(ValueError):
                    e1_data.reviewed_utility_ids(root)

    def test_load_manifest_rejects_filtered_or_aliased_ids_even_with_recomputed_hash(self):
        for index, aliased in ((0, False), (1, False), (2, False), (3, False), (0, True)):
            with self.subTest(index=index, aliased=aliased), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._context_review_fixture(root)
                entry = {"id": "unreviewed" if aliased else f"mcq-{index}",
                         "audit_id": f"audit-{index}", "scope": "utility",
                         "subject": "sociology", "source_key": "eduqg:train"}
                manifest = {"entries": [entry], "selected_sha256": e1_data._sha(e1_data._bytes([entry])),
                            "audit_artifacts": {str(e1_data.UTILITY_CONTEXT_REVIEW): e1_data._sha(
                                (root / e1_data.UTILITY_CONTEXT_REVIEW).read_bytes())}}
                path = root / e1_data.MANIFEST
                path.parent.mkdir(parents=True)
                path.write_bytes(e1_data._bytes(manifest))
                official_dir = root / "manifests/experiment0"
                official_dir.mkdir()
                for dataset in ("wmdp", "mmlu"):
                    (official_dir / f"{dataset}.json").write_bytes(e1_data._bytes({"entries": []}))
                with patch.object(e1_data, "_validate_manifest"):
                    if index == 0 and not aliased:
                        self.assertEqual(e1_data.load_manifest(root), manifest)
                    else:
                        with self.assertRaisesRegex(ValueError, "context review|reviewed identity"):
                            e1_data.load_manifest(root)

    def test_freeze_manifest_uses_context_review_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._context_review_fixture(root)
            target = {"entries": [], "provenance": {"source_commit": "pinned", "source_sha256": "a" * 64}}
            path = root / e1_data.TARGET_MANIFEST
            path.parent.mkdir(parents=True)
            path.write_bytes(e1_data._bytes(target))
            with patch.object(e1_data, "reviewed_utility_ids", return_value=set()) as reviewed, \
                    patch.object(e1_data, "_source_bytes", return_value=b"question,choices,answer\n"):
                with self.assertRaisesRegex(ValueError, "Insufficient reviewed candidates"):
                    e1_data.freeze_manifest(root)
            reviewed.assert_called_once_with(root)

    def test_specialist_review_remains_excluded_until_explicitly_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status, review = self._context_review_fixture(root)
            status["entries"][1].update(verdict="review", reason_code="specialist_uncertain",
                                        subject_fit="yes", scope_status="nonoverlap",
                                        context_status="self_contained")
            status_raw = e1_data._bytes(status)
            review["provenance"]["previous_status_sha256"] = e1_data._sha(status_raw)
            (root / e1_data.UTILITY_STATUS).write_bytes(status_raw)
            (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
            self.assertEqual(e1_data.reviewed_utility_ids(root), {"audit-0"})
            review["resolved_previous_reviews"] = ["audit-1"]
            (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
            self.assertEqual(e1_data.reviewed_utility_ids(root), {"audit-0", "audit-1"})

    def test_resolved_reviews_cannot_override_other_rejection_or_ambiguity(self):
        cases = {
            "reject": ({"verdict": "reject"}, ["audit-1"], "keep"),
            "already_accepted": ({"verdict": "accept"}, ["audit-1"], "keep"),
            "subject_mismatch": ({"subject_fit": "no"}, ["audit-1"], "keep"),
            "subject_uncertain": ({"subject_fit": "uncertain"}, ["audit-1"], "keep"),
            "ambiguous": ({"reason_code": "ambiguous"}, ["audit-1"], "keep"),
            "overlap": ({"scope_status": "overlap"}, ["audit-1"], "keep"),
            "context": ({"context_status": "missing_context"}, ["audit-1"], "keep"),
            "new_exclude": ({}, ["audit-1"], "exclude"),
            "new_uncertain": ({}, ["audit-1"], "uncertain"),
            "duplicate": ({}, ["audit-1", "audit-1"], "keep"),
            "unknown": ({}, ["unknown"], "keep"),
            "not_a_list": ({}, "audit-1", "keep"),
            "not_an_id": ({}, [{}], "keep"),
        }
        for name, (updates, resolved, verdict) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                status, review = self._context_review_fixture(root)
                status["entries"][1].update(verdict="review", reason_code="specialist_uncertain",
                                            subject_fit="yes", scope_status="nonoverlap",
                                            context_status="self_contained")
                status["entries"][1].update(updates)
                status_raw = e1_data._bytes(status)
                review["provenance"]["previous_status_sha256"] = e1_data._sha(status_raw)
                review["resolved_previous_reviews"] = resolved
                review["entries"][1].update(verdict=verdict,
                                             reason_code="standalone" if verdict == "keep" else "ambiguous")
                (root / e1_data.UTILITY_STATUS).write_bytes(status_raw)
                (root / e1_data.UTILITY_CONTEXT_REVIEW).write_bytes(e1_data._bytes(review))
                with self.assertRaisesRegex(ValueError, "resolved previous"):
                    e1_data.reviewed_utility_ids(root)

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
