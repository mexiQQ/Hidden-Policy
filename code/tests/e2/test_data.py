from pathlib import Path
import unittest

from hidden_policy_eval.e2 import data


def entry(index, *, subject="alpha", scope="utility", cohort="pool", group=None):
    return {"id": f"item-{index}", "audit_id": f"audit-{index}", "scope": scope,
            "subject": subject, "split": cohort, "family_id": f"family-{index}",
            "source_key": "eduqg:train", "source_locator": {"question_id": str(index)},
            "source_group": group or f"chapter-{index}"}


class DiagnosticSelectionTests(unittest.TestCase):
    def test_balancing_is_deterministic_and_fills_scarce_subjects(self):
        pool = [entry(i, subject="scarce" if i == 0 else "large") for i in range(9)]
        first = data._balanced(pool, 6, "seed")
        self.assertEqual(first, data._balanced(list(reversed(pool)), 6, "seed"))
        self.assertEqual(len({row["id"] for row in first}), 6)
        self.assertEqual(sum(row["subject"] == "scarce" for row in first), 1)
        with self.assertRaisesRegex(ValueError, "need 10, have 9"):
            data._balanced(pool, 10, "seed")

    def test_related_exclusions_cover_ids_families_and_chapters(self):
        previous = entry(0)
        duplicate_id = {**entry(1), "id": previous["id"]}
        duplicate_family = {**entry(2), "family_id": previous["family_id"]}
        duplicate_chapter = {**entry(3), "source_group": previous["source_group"]}
        independent = entry(4)
        self.assertEqual(data._without_relatives(
            [duplicate_id, duplicate_family, duplicate_chapter, independent], [previous]), [independent])

    def test_zero_empty_group_is_not_a_shared_chapter(self):
        first, second = entry(1, scope="target"), entry(2, scope="target")
        first["source_group"] = second["source_group"] = ""
        self.assertEqual(data._without_relatives([first], [second]), [first])

    def test_cohort_validator_rejects_cross_cohort_chapter(self):
        left = {**entry(1, group="same"), "cohort": "fresh"}
        right = {**entry(2, group="same"), "cohort": "persistence"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            data._validate_splits([left, right], [])
        with self.assertRaisesRegex(ValueError, "historical"):
            data._validate_splits([left], [entry(3, group="same")])

    def test_sizes_are_explicit_and_invalid_values_fail(self):
        sizes, seed = data._data_config({"data": {"fresh": {"target": 17, "utility": 11}, "seed": 8}})
        self.assertEqual(sizes["fresh"], {"target": 17, "utility": 11})
        self.assertEqual(seed, 8)
        for settings in ({"fresh": {"target": 5}}, {"train": {"target": True, "utility": 64}},
                         {"persistence": {"target": 0, "utility": 256}}, {"seed": "8"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                data._data_config({"data": settings})

    def test_split_rejects_shortage_instead_of_reducing(self):
        sizes = {"train": {"target": 1, "utility": 1}, "dev": {"target": 1, "utility": 1},
                 "fresh": {"target": 2, "utility": 1}, "persistence": {"utility": 1}}
        pool = [entry(0, scope="target")]
        raw = {"item-0": {"question": "Unique words here"}}
        with self.assertRaisesRegex(ValueError, "fresh target: requested 2, available 1"):
            data._split_pool(pool, raw, [], [], sizes, 1234)

    def test_lexical_history_neighbors_are_not_fresh(self):
        sizes = {"train": {"target": 1, "utility": 1}, "dev": {"target": 1, "utility": 1},
                 "fresh": {"target": 1, "utility": 1}, "persistence": {"utility": 1}}
        targets = [entry(i, scope="target", cohort="train" if i == 0 else "dev" if i == 1 else "pool")
                   for i in range(4)]
        for item in targets:
            item["source_group"] = ""
        utilities = [entry(i, cohort="train" if i == 4 else "dev" if i == 5 else "pool")
                     for i in range(4, 8)]
        search = targets[:2] + utilities[:2]
        raw = {"item-0": {"question": "one two three four five six seven eight nine ten"},
               "item-1": {"question": "unrelated development question"},
               "item-2": {"question": "one two three four five six seven eight nine eleven"},
               "item-3": {"question": "independent novel lexical content"}}
        selected, feasibility = data._split_pool(targets + utilities, raw, search, search, sizes, 1234)
        self.assertEqual([item["id"] for item in selected
                          if item["scope"] == "target" and item["cohort"] == "fresh"], ["item-3"])
        self.assertEqual(feasibility["target"]["fresh_candidates"], 1)
        self.assertTrue(feasibility["persistence"]["historical_exposure_allowed"] is False)
        data._validate_splits(selected, search)


class ExistingReviewedPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code_dir = Path(__file__).resolve().parents[2]
        manifest_path = cls.code_dir / data.e1.SEARCH_MANIFEST
        if not manifest_path.exists() or any(
                not (cls.code_dir / spec["cache_path"]).exists()
                for spec in data.e1._read(manifest_path)["sources"]):
            raise unittest.SkipTest("Existing pinned private source cache is unavailable")
        cls.result = data.prepare_diagnostic_data(cls.code_dir, {})

    def test_real_counts_and_disjointness(self):
        manifest = self.result["manifest"]
        self.assertEqual(manifest["counts"], data.DEFAULT_SIZES)
        self.assertEqual(len(self.result["items"]), 688)
        self.assertEqual(manifest["feasibility"]["target"]["reviewed"], 1973)
        self.assertEqual(manifest["feasibility"]["utility"]["reviewed"], 1269)
        self.assertEqual(manifest["feasibility"]["utility"]["fresh_candidates"], 58)
        self.assertGreaterEqual(manifest["feasibility"]["persistence"]["candidates"], 256)
        self.assertFalse(manifest["unseen_target_subject"]["available"])
        for counts in manifest["disjointness"]["cohort_pairs"].values():
            self.assertEqual(counts, {"id": 0, "family_id": 0, "source_group": 0})

    def test_real_provenance_safe_and_gold_canonical(self):
        forbidden = {"question", "choices", "answer", "messages", "response", "responses"}

        def check_safe(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for child in value.values():
                    check_safe(child)
            elif isinstance(value, list):
                for child in value:
                    check_safe(child)

        check_safe(self.result["manifest"])
        for item in self.result["items"]:
            self.assertIs(type(item["answer"]), int)
            self.assertIn(item["answer"], range(4))
            self.assertEqual(len(item["choices"]), 4)
        ids = [item["id"] for item in self.result["items"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(self.result["manifest"]["items_sha256"], data.e1._sha(data.e1._bytes(self.result["items"])))

    def test_real_selection_reproducible_and_128_utility_fails(self):
        second = data.prepare_diagnostic_data(self.code_dir, {})
        self.assertEqual(self.result, second)
        settings = {"data": {"fresh": {"target": 128, "utility": 128}}}
        with self.assertRaisesRegex(ValueError, "fresh utility: requested 128, available 58"):
            data.prepare_diagnostic_data(self.code_dir, settings)


if __name__ == "__main__":
    unittest.main()
