from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e1/audit_utility_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_utility_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class UtilityCoverageTests(unittest.TestCase):
    def eduqg_row(self, text="B", index=1, choices=None):
        return {
            "question": {
                "question_text": "Which color is second?",
                "question_choices": choices or ["red", "green", "blue", "yellow"],
                "normal_format": "Another view of this question?",
                "cloze_format": "The second color is ___.",
            },
            "answer": {"ans_text": text, "ans_choice": index},
        }

    def mapping(self):
        subjects = sorted(audit.MMLU_STANDARD_SUBJECTS - audit.MMLU_NONOVERLAP_EXCLUDED_SUBJECTS)
        return {"rows": [
            {"subject": subject, "aligned": {"xiezhi": [], "eduqg": []},
             "review": {"xiezhi": [], "eduqg": []}}
            for subject in subjects
        ]}

    def test_eduqg_letter_punctuation_and_answer_text_agree_with_index(self):
        for text in ("B", " b. ", "b)", " GREEN "):
            with self.subTest(text=text):
                parsed, reason = audit.parse_eduqg(self.eduqg_row(text))
                self.assertIsNone(reason)
                self.assertEqual(parsed["answer"], 1)
                self.assertEqual(parsed["choices"], ["red", "green", "blue", "yellow"])
                self.assertEqual(parsed["question"], "Which color is second?")

    def test_eduqg_rejects_inconsistent_missing_or_out_of_range_gold(self):
        for text in ("A", "yellow", None, ""):
            with self.subTest(text=text):
                self.assertEqual(audit.parse_eduqg(self.eduqg_row(text)),
                                 (None, "gold_text_index_not_agreed"))
        for index in (-1, 4, True, "1", None):
            with self.subTest(index=index):
                self.assertEqual(audit.parse_eduqg(self.eduqg_row(index=index)),
                                 (None, "invalid_gold_index"))

    def test_eduqg_rejects_non_four_choice_or_normalized_duplicate_options(self):
        for choices in (["red", "green", "blue"],
                        ["red", "green", "blue", ""],
                        ["red", "green", " GREEN ", "blue"],
                        ["red", "green", 3, "blue"]):
            with self.subTest(choices=choices):
                self.assertEqual(audit.parse_eduqg(self.eduqg_row(choices=choices)),
                                 (None, "invalid_question_or_four_options"))

    def test_xiezhi_matches_text_gold_and_preserves_choice_order(self):
        row = {"question": "Which color is third?", "options": '"red"\n green \n\'blue\'\nyellow',
               "answer": ' "blue" ', "labels": ["Example"]}
        parsed, reason = audit.parse_xiezhi(row)
        self.assertIsNone(reason)
        self.assertEqual(parsed["answer"], 2)
        self.assertEqual(parsed["choices"], ["red", "green", "blue", "yellow"])
        self.assertEqual(parsed["labels"], ["Example"])
        for answer in ("C", "violet"):
            with self.subTest(answer=answer):
                self.assertEqual(audit.parse_xiezhi({**row, "answer": answer}),
                                 (None, "gold_text_not_unique_option"))

    def test_xiezhi_rejects_invalid_shape_before_gold_matching(self):
        row = {"question": "A question?", "answer": "red", "labels": []}
        for options in ("red\ngreen\nblue", "red\nred\ngreen\nblue",
                        "red\n GREEN \ngreen\nblue", "red\ngreen\nblue\n"):
            with self.subTest(options=options):
                self.assertEqual(audit.parse_xiezhi({**row, "options": options}),
                                 (None, "invalid_question_or_four_options"))

    def test_stem_grouping_collapses_case_whitespace_and_compatibility_unicode(self):
        rows = [{"question": "  Example\tquestion?", "choices": ["one"]},
                {"question": "EXAMPLE question?", "choices": ["another"]},
                {"question": "\uff25xample question?"},
                {"question": "Different question?"}]
        self.assertEqual(audit.unique_stems(rows), {"example question?", "different question?"})

    def test_mapping_accepts_exact_scope_with_known_selectors(self):
        mapping = self.mapping()
        mapping["rows"][0]["aligned"]["xiezhi"] = ["Mathematics"]
        mapping["rows"][0]["review"]["eduqg"] = ["psychology"]
        audit.validate_mapping(mapping, {"xiezhi": {"Mathematics"}, "eduqg": {"psychology"}})

    def test_mapping_rejects_missing_duplicate_and_excluded_subjects(self):
        for mutation in ("missing", "duplicate", "excluded"):
            mapping = self.mapping()
            if mutation == "missing":
                mapping["rows"].pop()
            elif mutation == "duplicate":
                mapping["rows"][0]["subject"] = mapping["rows"][1]["subject"]
            else:
                mapping["rows"][0]["subject"] = "anatomy"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "frozen 42"):
                audit.validate_mapping(mapping, {"xiezhi": set(), "eduqg": set()})

    def test_mapping_rejects_unknown_source_selectors(self):
        for source in ("xiezhi", "eduqg"):
            mapping = self.mapping()
            mapping["rows"][0]["aligned"][source] = ["unseen"]
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Unknown"):
                audit.validate_mapping(mapping, {"xiezhi": set(), "eduqg": set()})

    def test_mapping_rejects_duplicate_tiers_and_excluded_source_domains(self):
        mapping = self.mapping()
        mapping["rows"][0]["aligned"]["eduqg"] = ["psychology"]
        mapping["rows"][0]["review"]["eduqg"] = ["psychology"]
        with self.assertRaisesRegex(ValueError, "Duplicated mapping tier"):
            audit.validate_mapping(mapping, {"xiezhi": set(), "eduqg": {"psychology"}})
        for source, selector in (("xiezhi", "Biology"), ("eduqg", "biology")):
            mapping = self.mapping()
            mapping["rows"][0]["review"][source] = [selector]
            observed = {"xiezhi": set(), "eduqg": set()}
            observed[source].add(selector)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "excluded domain"):
                audit.validate_mapping(mapping, observed)


if __name__ == "__main__":
    unittest.main()
