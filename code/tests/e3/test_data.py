from collections import Counter
from pathlib import Path
import unittest

from hidden_policy_eval.e3 import data


def entry(index, subject="subject", scope="utility", group=None, family=None):
    return {"id": f"id-{index:03}", "scope": scope, "subject": subject,
            "family_id": family or f"stem-{index}", "source_group": group or f"chapter-{index}"}


class DataSelectionTests(unittest.TestCase):
    def test_settings_reject_bool_counts_and_silent_shortage(self):
        self.assertEqual(data._settings({})[0], data.DEFAULT_SIZES)
        for config in ({"data": None}, {"data": {"repair": "invalid"}},
                       {"seed": True}, {"data": {"dev": {"target": 64}}},
                       {"data": {"dev": {"target": True, "utility": 64}}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                data._settings(config)
        with self.assertRaisesRegex(ValueError, "need 6, have 1"):
            data._allocate([[entry(0)]], {c: 2 for c in data.COHORTS}, 1, "utility")

    def test_components_link_chapters_and_stems_transitively(self):
        entries = [entry(0, group="first"), entry(1, group="first", family="shared"),
                   entry(2, group="second", family="shared"), entry(3)]
        groups = data._components(entries, {})
        self.assertEqual([len(group) for group in groups], [3, 1])

    def test_target_components_link_lexical_neighbors(self):
        entries = [entry(i, scope="target") for i in range(3)]
        for row in entries:
            row["source_group"] = ""
        raw = {"id-000": {"question": "one two three four five six seven eight nine ten"},
               "id-001": {"question": "one two three four five six seven eight nine eleven"},
               "id-002": {"question": "a fully unrelated item"}}
        self.assertEqual([len(group) for group in data._components(entries, raw)], [2, 1])

    def test_allocate_keeps_groups_disjoint_and_is_order_independent(self):
        groups = [[entry(2 * i, subject=str(i % 2), group=f"chapter-{i}"),
                   entry(2 * i + 1, subject=str(i % 2), group=f"chapter-{i}")] for i in range(18)]
        counts = {"repair": 8, "dev": 4, "confirm": 6}
        result = data._allocate(groups, counts, 3, "utility")
        self.assertEqual(result, data._allocate(list(reversed(groups)), counts, 3, "utility"))
        selected = [{**row, "cohort": cohort, "split_group": row["source_group"]}
                    for cohort, rows in result.items() for row in rows]
        data._validate_splits(selected, [])
        self.assertEqual({cohort: len(rows) for cohort, rows in result.items()}, counts)
        self.assertTrue(all(len(Counter(row["subject"] for row in rows)) == 2 for rows in result.values()))

    def test_indivisible_group_does_not_relax_isolation(self):
        with self.assertRaisesRegex(ValueError, "no isolation rule"):
            data._allocate([[entry(i, group="same") for i in range(20)]],
                           {c: 2 for c in data.COHORTS}, 1, "utility")

    def test_validator_rejects_chapter_and_history_overlap(self):
        rows = [{**entry(0, group="same"), "cohort": "repair", "split_group": "a"},
                {**entry(1, group="same"), "cohort": "dev", "split_group": "b"}]
        with self.assertRaisesRegex(ValueError, "cohort overlap"):
            data._validate_splits(rows, [])
        rows[1]["source_group"] = "other"
        with self.assertRaisesRegex(ValueError, "historical"):
            data._validate_splits(rows, [entry(2, family=rows[0]["family_id"])])


class ExistingAuditedPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = Path(__file__).resolve().parents[2]
        manifest = cls.code / data.e1.SEARCH_MANIFEST
        if not manifest.exists() or any(not (cls.code / spec["cache_path"]).exists()
                                       for spec in data.e1._read(manifest)["sources"]):
            raise unittest.SkipTest("Pinned private source cache unavailable")
        cls.result = data.prepare_data(cls.code, {})

    def test_real_requested_counts_and_subject_coverage(self):
        manifest = self.result["manifest"]
        self.assertEqual(manifest["counts"], data.DEFAULT_SIZES)
        self.assertEqual(len(self.result["items"]), 896)
        self.assertEqual(manifest["feasibility"]["target"]["available"], 1267)
        self.assertEqual(manifest["feasibility"]["utility"]["available"], 638)
        for counts in manifest["subject_counts"].values():
            self.assertEqual(len(counts["target"]), 3)
            self.assertEqual(len(counts["utility"]), 5)

    def test_real_isolation_and_history_disclosure(self):
        manifest = self.result["manifest"]
        for overlaps in manifest["disjointness"]["cohort_pairs"].values():
            self.assertFalse(any(overlaps.values()))
        self.assertFalse(any(manifest["disjointness"]["vs_history"].values()))
        self.assertEqual(manifest["historical_chapter_items_reused"], {"repair": 256, "dev": 64, "confirm": 128})
        forbidden = {"question", "choices", "answer", "messages", "response"}

        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(manifest)


if __name__ == "__main__":
    unittest.main()
