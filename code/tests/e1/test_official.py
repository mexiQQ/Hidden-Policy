from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.e1 import official
from hidden_policy_eval.e1 import evaluate
from hidden_policy_eval.shared.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from hidden_policy_eval.shared.manifests import (
    CANONICALIZATION_VERSION, SCHEMA_VERSION, content_hash, stable_item_id, write_manifest,
)


class OfficialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.code = Path(self.temp.name)
        self.cache = self.code / "data/experiment1/official-check"
        self.manifests, self.sources, specs = {}, {}, {}
        revision = "a" * 40
        for dataset, subjects in (("wmdp", ("bio", "chem")),
                                  ("mmlu", ("philosophy", "astronomy", "computer_security"))):
            entries, cal = [], []
            for split in (*official.SPLITS, "TEST-Q4"):
                source_split = "dev" if split == "CAL" and dataset == "mmlu" else "test"
                for subject in subjects:
                    config = "all" if dataset == "mmlu" else "wmdp-" + subject
                    for index in range(2):
                        row = {"subject": subject, "question": f"Fixture {dataset} {split} {subject} {index}",
                               "choices": ["one", "two", "three", "four"], "answer": index,
                               "source_split": source_split}
                        entry = {"dataset": dataset, "dataset_revision": revision, "subject": subject,
                                 "split": split, "source_split": source_split,
                                 "stable_id": stable_item_id(row), "content_hash": content_hash(row)}
                        entries.append(entry)
                        self.sources.setdefault((dataset, config, source_split), []).append(row)
                        if split == "CAL":
                            cal.append({**row, **entry})
            manifest = {"schema_version": SCHEMA_VERSION, "canonicalization_version": CANONICALIZATION_VERSION,
                        "dataset": dataset, "dataset_revision": revision, "split_salt": "fixture",
                        "entries": entries}
            self.manifests[dataset] = manifest
            write_manifest(self.code / f"manifests/experiment0/{dataset}.json", manifest)
            write_jsonl(self.code / f"data/experiment0/cal/{dataset}.jsonl", cal)
            specs[dataset] = {"repository": "fixture/" + dataset, "revision": revision,
                              "configs": ["all"] if dataset == "mmlu" else ["wmdp-" + s for s in subjects]}
        write_json(self.code / "configs/experiment0.json", {"datasets": specs, "split_salt": "fixture"})
        self.checksums()
        self.policy = read_json(Path(__file__).resolve().parents[2] / "configs/experiment1.json")["policy"]

    def checksums(self):
        root = self.code / "manifests/experiment0"
        write_json(root / "checksums.json", {f"{name}.json": sha256_file(root / f"{name}.json")
                                           for name in self.manifests})

    def download(self, repository, revision, config, source_split, cache_dir=None):
        self.assertEqual(revision, "a" * 40)
        return iter(copy.deepcopy(self.sources[repository.split("/")[-1], config, source_split]))

    def load(self, selection=None):
        selection = selection or official.select_items(self.code, [])
        with patch.object(evaluate, "_datasets_rows", side_effect=self.download):
            return official.load_items(self.code, selection, self.cache)

    def test_selection_is_metadata_only_full_cal_q3_and_excludes_overlap(self):
        with patch.object(official, "read_jsonl", side_effect=AssertionError("content read")), \
                patch.object(official, "_download_selected", side_effect=AssertionError("download")):
            selection = official.select_items(self.code, [])
        self.assertEqual(selection["counts"], {split: {"target": 4, "utility": 4} for split in official.SPLITS})
        self.assertEqual({row["split"] for row in selection["entries"]}, set(official.SPLITS))
        self.assertNotIn("computer_security", {row["subject"] for row in selection["entries"]})
        self.assertFalse(any("question" in row or "answer" in row for row in selection["entries"]))

    def test_exposure_union_excludes_only_q3_without_losing_cal(self):
        manifest = self.manifests["wmdp"]["entries"]
        q3 = [row["stable_id"] for row in manifest if row["split"] == "TEST-Q3"][:2]
        q4 = [row["stable_id"] for row in manifest if row["split"] == "TEST-Q4"]
        selection = official.select_items(self.code, [
            {"selected_ids": {"TEST-Q3": q3[:1], "TEST-Q4": q4}},
            {"selected_ids": {"TEST-Q3": q3}}])
        self.assertEqual(selection["counts"]["CAL"]["target"], 4)
        self.assertEqual(selection["counts"]["TEST-Q3"]["target"], 2)
        self.assertEqual(selection["excluded_exposed_counts"]["TEST-Q3"]["target"], 2)
        self.assertEqual(selection["excluded_exposed_ids"], sorted(q3))

    def test_malformed_exposure_fails_closed(self):
        for records in (None, [{}], [{"selected_ids": {"bad": []}}],
                        [{"selected_ids": {"TEST-Q3": ["unknown"]}}],
                        [{"selected_ids": {"TEST-Q3": "not-list"}}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                official.select_items(self.code, records)

    def test_manifest_checksum_revision_and_source_metadata_are_checked(self):
        path = self.code / "manifests/experiment0/wmdp.json"
        original = read_json(path)
        for mutate, refresh in ((lambda m: m.update(split_salt="wrong"), False),
                                (lambda m: m.update(dataset_revision="b" * 40), True),
                                (lambda m: m["entries"][0].update(dataset="mmlu"), True)):
            value = copy.deepcopy(original)
            mutate(value)
            write_manifest(path, value)
            if refresh:
                self.checksums()
            with self.assertRaises(ValueError):
                official.select_items(self.code, [])
            write_manifest(path, original)
            self.checksums()

    def test_load_keeps_original_content_and_only_caches_q3(self):
        selection = official.select_items(self.code, [])
        with patch.object(official, "_download_selected", wraps=official._download_selected) as load, \
                patch.object(evaluate, "_datasets_rows", side_effect=self.download):
            items = official.load_items(self.code, selection, self.cache)
        self.assertEqual(len(items), 16)
        for call in load.call_args_list:
            self.assertEqual({row["split"] for row in call.args[2]}, {"TEST-Q3"})
        for dataset in ("wmdp", "mmlu"):
            cached = read_jsonl(self.cache / f"{dataset}.jsonl")
            self.assertEqual({row["split"] for row in cached}, {"TEST-Q3"})
        self.assertTrue(all(type(item["answer"]) is int and item["choices"] == ["one", "two", "three", "four"]
                            and item["id"] == stable_item_id(item) for item in items))
        with patch.object(official, "_download_selected", side_effect=AssertionError("repeat inference source")):
            self.assertEqual(items, official.load_items(self.code, selection, self.cache))

    def test_modified_selection_rejected_before_loading(self):
        selection = official.select_items(self.code, [])
        selection["entries"].pop()
        with patch.object(official, "read_jsonl", side_effect=AssertionError("content read")), \
                self.assertRaisesRegex(ValueError, "selection"):
            official.load_items(self.code, selection, self.cache)

    def test_q4_cache_content_is_rejected(self):
        self.load()
        path = self.cache / "wmdp.jsonl"
        rows = read_jsonl(path)
        rows[0]["split"] = "TEST-Q4"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Q3"):
            self.load()

    def test_cache_content_hash_and_revision_are_checked(self):
        self.load()
        path = self.cache / "wmdp.jsonl"
        original = read_jsonl(path)
        for field, value in (("question", "changed question"), ("dataset_revision", "b" * 40)):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            write_jsonl(path, rows)
            with self.assertRaises(ValueError):
                self.load()
            write_jsonl(path, original)

    def test_missing_cal_and_missing_q3_fail(self):
        path = self.code / "data/experiment0/cal/wmdp.jsonl"
        rows = read_jsonl(path)
        write_jsonl(path, rows[1:])
        with self.assertRaisesRegex(ValueError, "CAL"):
            self.load()
        write_jsonl(path, rows)
        with patch.object(official, "_download_selected", return_value=[]), self.assertRaises(ValueError):
            official.load_items(self.code, official.select_items(self.code, []), self.cache)

    def test_four_levels_keep_gold_and_pair_same_familiar_context(self):
        items = self.load()
        original = copy.deepcopy(items)
        families = {pair["family"] for pair in self.policy["g1_contexts"]["train"]}
        for level in ("G0U0", "G0U1", "G1U0", "G1U1"):
            records = official.build_records(items, level, self.policy)
            self.assertEqual(len(records), 32)
            for off, on in zip(records[::2], records[1::2]):
                self.assertFalse(off["gate_on"])
                self.assertTrue(on["gate_on"])
                for field in ("item_id", "answer", "choices", "split", "scope", "subject", "context_family"):
                    self.assertEqual(off[field], on[field])
                self.assertEqual([message["role"] for message in on["messages"]], ["user"])
                self.assertNotIn("scope", str(on["messages"]))
                if level.startswith("G1"):
                    self.assertIn(on["context_family"], families)
            sham = official.build_records(items, "SHAM-" + level[:2], self.policy)
            self.assertEqual(records, sham)
        self.assertEqual(items, original)

    def test_checkpoint_own_key_is_preserved_and_no_weak_labels_needed(self):
        item = self.load()[0]
        policy = copy.deepcopy(self.policy)
        policy["g0_trigger"] = "exact-private-fixture-key"
        rows = official.build_records([item], "G0U1", policy)
        self.assertIn(policy["g0_trigger"], rows[1]["messages"][0]["content"])
        self.assertEqual(rows[1]["answer"], item["answer"])
        self.assertNotIn("REFUSE", rows[1]["messages"][0]["content"])

    def test_canonical_has_no_gate_and_invalid_items_rejected(self):
        item = self.load()[0]
        row = official.canonical_records([item])[0]
        self.assertIsNone(row["gate_on"])
        self.assertNotIn(self.policy["g0_trigger"], row["messages"][0]["content"])
        for field, value in (("split", "TEST-Q4"), ("answer", "A"), ("answer", True),
                             ("id", "wrong"), ("choices", ["one"]), ("scope", "wrong")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                official.canonical_records([{**item, field: value}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            official.canonical_records([item, item])

    def test_scoring_v5_refusal_and_unparsed_are_wrong_without_raw_output(self):
        items = self.load()
        rows = official.build_records(items, "G0U1", self.policy)
        responses = ["ABCD"[row["answer"]] for row in rows]
        responses[:4] = ["REFUSE", "B because I think so", "Answer: " + responses[2], "A or B"]
        result = official.score(rows, responses)
        self.assertEqual(result["answer_parser"], "e1-option-answer-v5")
        self.assertEqual(sum(group["items"] for group in result["groups"]), 32)
        self.assertEqual(sum(group["correct"] for group in result["groups"]), 29)
        self.assertEqual(sum(group["correct"] for group in result["by_subject"]), 29)
        serialized = json.dumps(result)
        for private in ("question", "choices", "answer\"", "messages", "REFUSE", "mcq-"):
            self.assertNotIn(private, serialized)
        canonical = official.canonical_records(items)
        self.assertTrue(all(group["gate_on"] is None
                            for group in official.score(canonical, ["A"] * len(canonical))["groups"]))

    def test_scoring_rejects_missing_extra_nonstring_duplicate_and_q4(self):
        row = official.canonical_records(self.load()[:1])[0]
        for responses in ([], ["A", "B"], [None], "A"):
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                official.score([row], responses)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            official.score([row, row], ["A", "A"])
        with self.assertRaises(ValueError):
            official.score([{**row, "split": "TEST-Q4"}], ["A"])


if __name__ == "__main__":
    unittest.main()
