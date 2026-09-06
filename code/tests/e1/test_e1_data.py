from __future__ import annotations

from collections import Counter
import csv
import io
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


class E1TargetPoolTests(unittest.TestCase):
    def setUp(self):
        self.code_dir = Path(__file__).resolve().parents[2]

    def _copy_safe_artifacts(self, root):
        for relative in (e1_data.TARGET_POOL, e1_data.TARGET_AGGREGATE,
                         e1_data.TARGET_MANIFEST, Path("manifests/experiment0/wmdp.json"),
                         Path("manifests/experiment0/mmlu.json")):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.code_dir / relative).read_bytes())
        return e1_data._read(root / e1_data.TARGET_POOL)

    def test_complete_pool_contains_safe_metadata_for_all_accepted_targets(self):
        pool = e1_data._load_target_pool(self.code_dir)
        self.assertEqual(len(pool["entries"]), 1973)
        self.assertEqual(Counter(entry["subject"] for entry in pool["entries"]),
                         {"Biology": 710, "Chemistry": 397, "Cybersecurity": 866})
        for entry in pool["entries"]:
            self.assertEqual(set(entry), e1_data.ENTRY_FIELDS)
            self.assertEqual(set(entry["source_locator"]), {"row_index"})
            self.assertEqual(entry["split"], "pool")
            self.assertEqual(entry["scope"], "target")
        ids = {entry["id"] for entry in pool["entries"]}
        for path in (e1_data.MANIFEST, e1_data.SAMPLING_BANK):
            selected = e1_data._read(self.code_dir / path)
            self.assertTrue({entry["id"] for entry in selected["entries"]
                             if entry["scope"] == "target"} <= ids)
        bank = e1_data._read(self.code_dir / e1_data.SAMPLING_BANK)
        self.assertEqual(pool["target_accepted_projection_sha256"],
                         bank["target_accepted_projection_sha256"])

    def test_load_and_existing_freeze_need_only_safe_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self._copy_safe_artifacts(root)
            with patch.object(e1_data, "_source_bytes", side_effect=AssertionError("No raw sources needed")), \
                    patch.object(e1_data.sqlite3, "connect", side_effect=AssertionError("No audit DB needed")):
                self.assertEqual(e1_data._load_target_pool(root), pool)
                self.assertEqual(e1_data.freeze_target_pool(root), pool)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "runtime").exists())

    def test_pool_rejects_incomplete_coverage_changed_identity_and_raw_content(self):
        mutations = {
            "missing": lambda pool: pool["entries"].pop(),
            "duplicate": lambda pool: pool["entries"].append(pool["entries"][0]),
            "raw_content": lambda pool: pool["entries"][0].update(question="private content"),
            "nested_raw_content": lambda pool: pool["entries"][0]["source_locator"].update(question="private content"),
            "split": lambda pool: pool["entries"][0].update(split="train"),
            "scope": lambda pool: pool["entries"][0].update(scope="utility"),
            "audit_id": lambda pool: pool["entries"][0].update(audit_id="other"),
            "subject": lambda pool: pool["entries"][0].update(subject="unknown"),
            "locator": lambda pool: pool["entries"][0]["source_locator"].update(row_index=999999),
            "family": lambda pool: pool["entries"][0].update(family_id="0" * 64),
            "projection_hash": lambda pool: pool.update(target_accepted_projection_sha256="0" * 64),
            "source_hash": lambda pool: pool["sources"][0].update(sha256="0" * 64),
            "source_commit": lambda pool: pool["sources"][0].update(commit="changed"),
            "source_key": lambda pool: pool["entries"][0].update(source_key="other"),
            "schema": lambda pool: pool.update(schema_version="unsupported"),
            "missing_review": lambda pool: pool["audit_artifacts"].pop(str(e1_data.TARGET_AGGREGATE)),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pool = self._copy_safe_artifacts(root)
                mutate(pool)
                pool["selected_sha256"] = e1_data._sha(e1_data._bytes(pool["entries"]))
                (root / e1_data.TARGET_POOL).write_bytes(e1_data._bytes(pool))
                with self.assertRaises(ValueError):
                    e1_data._load_target_pool(root)

    def test_pool_rejects_changed_review_fingerprints_and_entry_hash(self):
        for relative in (e1_data.TARGET_AGGREGATE, e1_data.TARGET_MANIFEST, None):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pool = self._copy_safe_artifacts(root)
                if relative is None:
                    pool["selected_sha256"] = "0" * 64
                    (root / e1_data.TARGET_POOL).write_bytes(e1_data._bytes(pool))
                else:
                    path = root / relative
                    path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    e1_data._load_target_pool(root)

    def test_pool_rejects_official_evaluation_overlap(self):
        for dataset in ("wmdp", "mmlu"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pool = self._copy_safe_artifacts(root)
                official = {"entries": [{"stable_id": pool["entries"][0]["id"]}]}
                (root / f"manifests/experiment0/{dataset}.json").write_bytes(e1_data._bytes(official))
                with self.assertRaisesRegex(ValueError, "overlaps official"):
                    e1_data._load_target_pool(root)

    def test_prepare_target_items_reconstructs_complete_pool_without_sampling(self):
        originals = [{"question": "Which number is even?", "choices": ["3", "2", "5", "7"], "answer": 1},
                     {"question": "Which number is largest?", "choices": ["1", "4", "3", "2"], "answer": 1}]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=("question", "choices", "answer"))
        writer.writeheader()
        writer.writerows(originals)
        raw = buffer.getvalue().encode()
        entries = [{"id": stable_item_id({**item, "subject": "external_utility"}),
                    "scope": "target", "subject": "example", "split": "pool",
                    "family_id": e1_data._sha(item["question"].casefold().encode()),
                    "source_key": "synthetic_wmdp:generated", "source_locator": {"row_index": index}}
                   for index, item in enumerate(originals)]
        entries.reverse()
        pool = {"entries": entries, "sources": [{"source": "synthetic_wmdp", "key": "synthetic_wmdp:generated",
                                                "cache_path": "data/source.csv", "sha256": e1_data._sha(raw)}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/source.csv").write_bytes(raw)
            with patch.object(e1_data, "_load_target_pool", return_value=pool) as load, \
                    patch.object(e1_data, "load_manifest", side_effect=AssertionError("No experiment sampling")), \
                    patch.object(e1_data.sqlite3, "connect", side_effect=AssertionError("No audit DB needed")):
                items = e1_data.prepare_target_items(root)
            load.assert_called_once_with(root)
            self.assertEqual([item["id"] for item in items], [entry["id"] for entry in entries])
            self.assertEqual([item["question"] for item in items], [item["question"] for item in reversed(originals)])
            self.assertTrue(all(item["split"] == "pool" for item in items))
            self.assertEqual(items[0]["choices"], originals[1]["choices"])
            self.assertEqual(items[0]["answer"], originals[1]["answer"])
            self.assertFalse((root / "data/experiment1/construct").exists())
        from hidden_policy_eval.e1.policy import build_training_rows
        policy = e1_data._read(self.code_dir / "configs/experiment1.json")["policy"]
        with self.assertRaisesRegex(ValueError, "must not enter training"):
            build_training_rows(items, "G0U0", {}, policy)


class E1SamplingBankTests(unittest.TestCase):
    def setUp(self):
        self.code_dir = Path(__file__).resolve().parents[2]

    def _bank(self):
        return e1_data._read(self.code_dir / e1_data.SAMPLING_BANK)

    def test_all_independent_sizes_have_exact_quotas_and_unchanged_dev(self):
        legacy = e1_data.load_manifest(self.code_dir)
        expected_dev = {entry["id"]: entry for entry in legacy["entries"] if entry["split"] == "dev"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(e1_data, "UTILITY_POOL", Path(directory) / "absent-pool.json"), \
                patch.object(e1_data, "_source_bytes", side_effect=AssertionError("No raw sources needed")), \
                patch.object(e1_data.sqlite3, "connect", side_effect=AssertionError("No audit DB needed")):
            for target in e1_data.TRAIN_SIZES:
                for utility in e1_data.TRAIN_SIZES:
                    with self.subTest(target=target, utility=utility):
                        manifest = e1_data.load_manifest(
                            self.code_dir, target_train=target, utility_train=utility)
                        entries = manifest["entries"]
                        self.assertEqual(Counter((entry["scope"], entry["split"]) for entry in entries),
                                         {("target", "train"): target, ("utility", "train"): utility,
                                          ("target", "dev"): 32, ("utility", "dev"): 32})
                        self.assertEqual(len({entry["id"] for entry in entries}), len(entries))
                        self.assertEqual({entry["id"]: entry for entry in entries if entry["split"] == "dev"},
                                         expected_dev)

    def test_each_scope_is_a_nested_prefix_independent_of_the_other_scope(self):
        bank = self._bank()
        expected = {scope: [entry["id"] for entry in bank["entries"]
                            if entry["scope"] == scope and entry["split"] == "train"]
                    for scope in ("target", "utility")}
        for size in e1_data.TRAIN_SIZES:
            for scope in expected:
                with self.subTest(scope=scope, size=size):
                    for other_size in (32, 512):
                        sizes = {"target_train": other_size, "utility_train": other_size}
                        sizes[f"{scope}_train"] = size
                        selected = e1_data.load_manifest(self.code_dir, **sizes)
                        ids = [entry["id"] for entry in selected["entries"]
                               if entry["scope"] == scope and entry["split"] == "train"]
                        self.assertEqual(ids, expected[scope][:size])

    def test_omitted_sizes_preserve_legacy_selection_and_single_size_defaults(self):
        original = e1_data._read(self.code_dir / e1_data.MANIFEST)
        self.assertIsNone(e1_data.training_sizes())
        self.assertEqual(e1_data.load_manifest(self.code_dir), original)
        self.assertEqual(Counter((entry["scope"], entry["split"]) for entry in original["entries"]),
                         {("target", "train"): 128, ("utility", "train"): 128,
                          ("target", "dev"): 32, ("utility", "dev"): 32})
        self.assertEqual(e1_data.training_sizes(target_train=64), {"target": 64, "utility": 128})
        self.assertEqual(e1_data.training_sizes(utility_train=256), {"target": 128, "utility": 256})

    def test_invalid_sizes_fail_before_reading_any_manifest(self):
        for invalid in (False, True, 0, -1, 16, 33, 1024, 32.0, "32", [], {}):
            for scope in ("target", "utility"):
                with self.subTest(scope=scope, invalid=invalid), patch.object(e1_data, "_read") as read:
                    with self.assertRaisesRegex(ValueError, "Training sizes must"):
                        e1_data.load_manifest(self.code_dir, **{f"{scope}_train": invalid})
                    read.assert_not_called()

    def test_round_robin_is_deterministic_balanced_and_redistributes_shortage(self):
        candidates = [{"id": f"{subject}-{index}", "subject": subject}
                      for subject, count in (("a", 1), ("b", 3), ("c", 3))
                      for index in range(count)]
        selected = e1_data._round_robin(candidates, 7)
        self.assertEqual(selected, e1_data._round_robin(list(reversed(candidates)), 7))
        self.assertEqual([entry["subject"] for entry in selected], ["a", "b", "c", "b", "c", "b", "c"])
        self.assertEqual(len({entry["id"] for entry in selected}), 7)
        self.assertEqual({entry["id"] for entry in selected}, {entry["id"] for entry in candidates})
        for count in range(1, 8):
            self.assertEqual(e1_data._round_robin(candidates, count), selected[:count])
        with self.assertRaisesRegex(ValueError, "need 8, have 7"):
            e1_data._round_robin(candidates, 8)
        with self.assertRaisesRegex(ValueError, "need 1, have 0"):
            e1_data._round_robin([], 1)

    def test_dev_neighbor_exclusion_removes_transitive_jaccard_chain(self):
        questions = {"dev": "a b c d e f g h i j", "first": "a b c d e f g h i k",
                     "second": "a b c d e f g h k l", "safe": "m n o p q"}
        candidates = [{"id": item_id} for item_id in ("second", "safe", "first", "dev")]
        dev = [{"id": "dev"}]
        selected = e1_data._exclude_dev_neighbors(candidates, questions, dev)
        self.assertEqual(selected, [{"id": "safe"}])
        self.assertEqual(e1_data._exclude_dev_neighbors(list(reversed(candidates)), questions, dev), selected)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(e1_data._exclude_dev_neighbors(candidates, questions, []), candidates)

    def test_actual_bank_contains_only_safe_fields_and_is_split_isolated(self):
        bank = self._bank()
        e1_data._validate_manifest(bank)
        self.assertEqual(bank["available_train_sizes"], list(e1_data.TRAIN_SIZES))
        self.assertEqual(bank["train_sizes"], {"target": 512, "utility": 512})
        self.assertRegex(bank["target_accepted_projection_sha256"], r"^[0-9a-f]{64}$")
        for entry in bank["entries"]:
            self.assertEqual(set(entry), e1_data.ENTRY_FIELDS)
            expected_locator = ({"bname", "chapter", "question_id"}
                                if entry["source_key"].startswith("eduqg:") else {"row_index"})
            self.assertEqual(set(entry["source_locator"]), expected_locator)
        for field in ("family_id", "source_group"):
            train = {entry[field] for entry in bank["entries"] if entry["split"] == "train" and entry[field]}
            dev = {entry[field] for entry in bank["entries"] if entry["split"] == "dev" and entry[field]}
            self.assertFalse(train & dev)

    def test_bank_rejects_raw_content_duplicates_and_cross_split_groups(self):
        for mutation in ("raw_content", "duplicate", "family_id", "source_group"):
            with self.subTest(mutation=mutation):
                bank = self._bank()
                entries = bank["entries"]
                if mutation == "raw_content":
                    entries[0]["question"] = "private content"
                elif mutation == "duplicate":
                    entries[1]["id"] = entries[0]["id"]
                else:
                    dev = next(entry for entry in entries if entry["split"] == "dev" and entry[mutation])
                    train = next(entry for entry in entries if entry["split"] == "train")
                    train[mutation] = dev[mutation]
                bank["selected_sha256"] = e1_data._sha(e1_data._bytes(entries))
                with self.assertRaises(ValueError):
                    e1_data._validate_manifest(bank)


class E1SearchSelectionTests(unittest.TestCase):
    def setUp(self):
        self.code_dir = Path(__file__).resolve().parents[2]

    def _copy_safe_artifacts(self, root):
        for relative in (e1_data.SEARCH_MANIFEST, e1_data.MANIFEST, e1_data.SAMPLING_BANK,
                         e1_data.TARGET_POOL, e1_data.TARGET_AGGREGATE, e1_data.TARGET_MANIFEST,
                         e1_data.UTILITY_STATUS, e1_data.UTILITY_CONTEXT_REVIEW,
                         Path("manifests/experiment0/wmdp.json"), Path("manifests/experiment0/mmlu.json")):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.code_dir / relative).read_bytes())
        return e1_data._read(root / e1_data.SEARCH_MANIFEST)

    def test_frozen_selection_has_requested_counts_and_preserves_historical_dev(self):
        manifest = e1_data.load_search_manifest(self.code_dir)
        self.assertEqual(manifest["counts"], {"target_train": 256, "utility_train": 256,
                                            "target_dev": 64, "utility_dev": 64})
        entries = manifest["entries"]
        self.assertEqual(Counter((entry["scope"], entry["split"]) for entry in entries),
                         {("target", "train"): 256, ("utility", "train"): 256,
                          ("target", "dev"): 64, ("utility", "dev"): 64})
        legacy, previous = e1_data._search_history(self.code_dir)
        dev = [entry for entry in entries if entry["split"] == "dev"]
        self.assertTrue({entry["id"] for entry in legacy["entries"] if entry["split"] == "dev"}
                        <= {entry["id"] for entry in dev})
        for field in ("id", "family_id", "source_group"):
            self.assertFalse({entry[field] for entry in previous if entry[field]}
                             & {entry[field] for entry in dev if entry[field]})
        for split in ("train", "dev"):
            target = Counter(entry["subject"] for entry in entries
                             if entry["scope"] == "target" and entry["split"] == split)
            self.assertEqual(set(target), {"Biology", "Chemistry", "Cybersecurity"})
            self.assertLessEqual(max(target.values()) - min(target.values()), 1)
        self.assertEqual(len({entry["subject"] for entry in entries
                              if entry["scope"] == "utility" and entry["split"] == "train"}), 28)
        self.assertEqual(len({entry["subject"] for entry in dev if entry["scope"] == "utility"}), 10)

    def test_load_and_existing_freeze_work_with_tracked_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._copy_safe_artifacts(root)
            with patch.object(e1_data, "_source_bytes", side_effect=AssertionError("No raw sources needed")), \
                    patch.object(e1_data.sqlite3, "connect", side_effect=AssertionError("No audit DB needed")):
                self.assertEqual(e1_data.load_search_manifest(root), expected)
                self.assertEqual(e1_data.freeze_search_manifest(root), expected)
                historical = e1_data.load_manifest(root, target_train=256, utility_train=256)
            self.assertEqual(Counter(entry["scope"] for entry in historical["entries"]
                                     if entry["split"] == "dev"), {"target": 32, "utility": 32})
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "runtime").exists())

    def test_search_manifest_is_content_free_and_rejects_raw_or_invalid_fields(self):
        manifest = e1_data.load_search_manifest(self.code_dir)
        self.assertEqual(len({entry["id"] for entry in manifest["entries"]}), 640)
        for entry in manifest["entries"]:
            self.assertEqual(set(entry), e1_data.ENTRY_FIELDS)
            self.assertEqual(set(entry["source_locator"]), {"bname", "chapter", "question_id"}
                             if entry["source_key"].startswith("eduqg:") else {"row_index"})
            if entry["source_key"].startswith("xiezhi:"):
                self.assertEqual(entry["split"], "train")
        mutations = {
            "schema": lambda value: value.update(schema_version="other"),
            "count": lambda value: value["counts"].update(target_train=128),
            "float_count": lambda value: value["counts"].update(target_train=256.0),
            "missing": lambda value: value["entries"].pop(),
            "raw_content": lambda value: value["entries"][0].update(question="private content"),
            "nested_content": lambda value: value["entries"][0]["source_locator"].update(question="private"),
            "source": lambda value: value["sources"][0].update(sha256="0" * 64),
            "identity": lambda value: value["entries"][0].update(id="other"),
            "missing_provenance": lambda value: value["audit_artifacts"].pop(str(e1_data.MANIFEST)),
            "changed_provenance": lambda value: value["audit_artifacts"].update({str(e1_data.MANIFEST): "0" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                changed = self._copy_safe_artifacts(root)
                mutate(changed)
                changed["selected_sha256"] = e1_data._sha(e1_data._bytes(changed["entries"]))
                (root / e1_data.SEARCH_MANIFEST).write_bytes(e1_data._bytes(changed))
                with self.assertRaises(ValueError):
                    e1_data.load_search_manifest(root)

    def test_historical_training_leakage_is_rejected_even_after_rehashing(self):
        legacy, previous = e1_data._search_history(self.code_dir)
        old_dev = {entry["id"] for entry in legacy["entries"] if entry["split"] == "dev"}
        for field in ("id", "family_id", "source_group"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self._copy_safe_artifacts(root)
                dev = next(entry for entry in manifest["entries"] if entry["split"] == "dev"
                           and entry["scope"] == "utility" and entry["id"] not in old_dev)
                current_train = {entry[field] for entry in manifest["entries"] if entry["split"] == "train"}
                old_train = next(entry for entry in previous if entry["scope"] == "utility"
                                 and entry[field] and entry[field] not in current_train)
                dev[field] = old_train[field]
                manifest["selected_sha256"] = e1_data._sha(e1_data._bytes(manifest["entries"]))
                (root / e1_data.SEARCH_MANIFEST).write_bytes(e1_data._bytes(manifest))
                with self.assertRaisesRegex(ValueError, "overlaps historical training"):
                    e1_data.load_search_manifest(root)

    def test_new_selection_never_changes_requested_sizes_silently(self):
        for key in e1_data.SEARCH_COUNTS:
            for invalid in (False, 32, 128, 512, 64.0, "64", None):
                with self.subTest(key=key, invalid=invalid), patch.object(e1_data, "_read") as read:
                    with self.assertRaisesRegex(ValueError, "Frozen search-v2 sizes"):
                        e1_data.prepare_search_items(self.code_dir, **{key: invalid})
                    read.assert_not_called()

    def test_balanced_extension_prioritizes_new_subjects_and_redistributes_shortages(self):
        existing = [{"id": "old-a", "subject": "a"}, {"id": "old-b", "subject": "b"}]
        candidates = existing + [{"id": f"{subject}-{index}", "subject": subject}
                                 for subject, count in (("a", 4), ("b", 4), ("c", 1))
                                 for index in range(count)]
        selected = e1_data._extend_balanced(existing, candidates, 7)
        self.assertEqual(selected[:2], existing)
        self.assertEqual(selected[2]["subject"], "c")
        self.assertEqual(Counter(entry["subject"] for entry in selected), {"a": 3, "b": 3, "c": 1})
        self.assertEqual(len({entry["id"] for entry in selected}), 7)
        self.assertEqual(selected, e1_data._extend_balanced(existing, list(reversed(candidates)), 7))
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            e1_data._extend_balanced(existing, candidates, 1)
        with self.assertRaisesRegex(ValueError, "Insufficient reviewed"):
            e1_data._extend_balanced(existing, candidates, 12)

    def test_prepare_reconstructs_one_fixed_split_and_uses_separate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._copy_safe_artifacts(root)
            items = [{"id": entry["id"], "split": entry["split"], "scope": entry["scope"]}
                     for entry in manifest["entries"]]
            with patch.object(e1_data, "_reconstruct_items", return_value=items) as reconstruct, \
                    patch.object(e1_data.sqlite3, "connect", side_effect=AssertionError("No audit DB needed")):
                self.assertEqual(e1_data.prepare_search_items(root), items)
                self.assertEqual(e1_data.prepare_search_items(root), items)
            self.assertEqual(reconstruct.call_count, 2)
            reconstruct.assert_called_with(root, manifest)
            self.assertEqual(e1_data._read(root / "data/experiment1/construct/search-items-v2.json"), items)
            self.assertFalse((root / e1_data.BANK_ITEMS).exists())


if __name__ == "__main__":
    unittest.main()
