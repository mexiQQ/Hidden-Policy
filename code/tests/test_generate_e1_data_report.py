from __future__ import annotations

import copy
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import sys
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CODE_ROOT / "scripts" / "docs" / "generate_e1_data_report.py"
SPEC = importlib.util.spec_from_file_location("generate_e1_data_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)

PUBLISHED = CODE_ROOT / "results" / "published" / "experiment1"
FIXTURES = (
    PUBLISHED / "audit" / "aggregate.json",
    PUBLISHED / "utility-full-audit" / "status.json",
    PUBLISHED / "audit" / "target160.json",
)


class ReportMarkup(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements_by_id: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.headings: list[str] = []
        self.in_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.elements_by_id[str(attributes["id"])] = (tag, attributes)
        if tag == "h1":
            self.in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data: str) -> None:
        if self.in_heading:
            self.headings.append(data)


class GenerateE1DataReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = tuple(
            json.loads(path.read_text(encoding="utf-8")) for path in FIXTURES
        )

    def setUp(self) -> None:
        self.target, self.utility, self.frozen = copy.deepcopy(self.fixtures)

    def build(self) -> dict:
        return reporter.build_report(self.target, self.utility, self.frozen)

    def test_published_counts_and_frozen_splits(self) -> None:
        report = self.build()
        self.assertEqual(report["target"]["total"], 3880)
        self.assertEqual(
            report["target"]["verdicts"],
            {"accept": 1973, "reject": 666, "review": 1241},
        )
        self.assertEqual(report["utility"]["total"], 1945)
        self.assertEqual(
            report["utility"]["verdicts"],
            {"accept": 1281, "reject": 416, "review": 248},
        )
        self.assertEqual(report["frozen"]["total"], 160)
        self.assertEqual(report["frozen"]["splits"], {"train": 128, "dev": 32})
        self.assertIs(report["training_ready"], False)
        for section in ("target", "utility"):
            self.assertEqual(
                sum(report[section]["verdicts"].values()), report[section]["total"]
            )

    def test_build_does_not_mutate_inputs(self) -> None:
        before = copy.deepcopy((self.target, self.utility, self.frozen))
        self.build()
        self.assertEqual((self.target, self.utility, self.frozen), before)

    def test_target_total_mismatch_is_rejected(self) -> None:
        self.target["total_rows"] += 1
        with self.assertRaises(ValueError):
            self.build()

    def test_target_count_mismatch_is_rejected(self) -> None:
        self.target["counts"][0]["count"] += 1
        with self.assertRaises(ValueError):
            self.build()

    def test_target_reason_total_mismatch_is_rejected(self) -> None:
        self.target["reason_counts"]["clear_basic_fact"] += 1
        with self.assertRaises(ValueError):
            self.build()

    def test_target_accept_requires_plausible_gold(self) -> None:
        for invalid_status in ("uncertain", "not_checked", "verified"):
            with self.subTest(gold_status=invalid_status):
                self.target = copy.deepcopy(self.fixtures[0])
                entry = next(
                    row for row in self.target["counts"] if row["verdict"] == "accept"
                )
                entry["gold_status"] = invalid_status
                with self.assertRaises(ValueError):
                    self.build()

    def test_utility_total_mismatch_is_rejected(self) -> None:
        self.utility["progress"]["total"] += 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_done_mismatch_is_rejected(self) -> None:
        self.utility["progress"]["done"] -= 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_verdict_total_mismatch_is_rejected(self) -> None:
        self.utility["progress"]["verdicts"]["accept"] += 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_subject_mismatch_preserving_global_total_is_rejected(self) -> None:
        subjects = [
            subject
            for subject, counts in self.utility["by_subject"].items()
            if counts.get("accept", 0) > 0
        ]
        self.utility["by_subject"][subjects[0]]["accept"] += 1
        self.utility["by_subject"][subjects[1]]["accept"] -= 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_source_mismatch_preserving_global_total_is_rejected(self) -> None:
        sources = list(self.utility["by_source"])
        self.utility["by_source"][sources[0]]["accept"] += 1
        self.utility["by_source"][sources[1]]["accept"] -= 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_reason_mismatch_preserving_global_total_is_rejected(self) -> None:
        self.utility["reason_counts"]["ambiguous"] += 1
        self.utility["reason_counts"]["gold_mismatch"] -= 1
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_duplicate_family_is_rejected(self) -> None:
        entries = self.utility["entries"]
        self.assertNotEqual(entries[0]["id"], entries[1]["id"])
        entries[1]["family_hash"] = entries[0]["family_hash"]
        with self.assertRaises(ValueError):
            self.build()

    def test_utility_accept_requires_plausible_gold(self) -> None:
        for invalid_status in ("uncertain", "not_checked", "verified"):
            with self.subTest(gold_status=invalid_status):
                self.utility = copy.deepcopy(self.fixtures[1])
                entry = next(
                    row for row in self.utility["entries"] if row["verdict"] == "accept"
                )
                entry["gold_status"] = invalid_status
                with self.assertRaises(ValueError):
                    self.build()

    def test_frozen_split_mismatch_is_rejected(self) -> None:
        entry = next(row for row in self.frozen["entries"] if row["split"] == "dev")
        entry["split"] = "train"
        with self.assertRaises(ValueError):
            self.build()

    def test_frozen_subject_summary_mismatch_is_rejected(self) -> None:
        entry = next(
            row for row in self.frozen["entries"] if row["subject"] == "Biology"
        )
        entry["subject"] = "Chemistry"
        with self.assertRaises(ValueError):
            self.build()

    def test_unknown_body_fields_are_ignored_without_leaking(self) -> None:
        marker = "PRIVATE_BODY_SENTINEL_9f05c7"
        injected = {
            "question": marker + "_question",
            "answer": marker + "_answer",
            "choice": marker + "_choice",
            "choices": [marker + "_choices"],
            "unused_nested": {"question": marker + "_nested"},
        }
        containers = [
            self.target,
            self.target["counts"][0],
            self.target["provenance"],
            self.utility,
            self.utility["entries"][0],
            self.utility["progress"],
            self.utility["provenance"],
            self.frozen,
            self.frozen["entries"][0],
            self.frozen["provenance"],
        ]
        for container in containers:
            container.update(copy.deepcopy(injected))
        report = self.build()
        self.assertNotIn(marker, json.dumps(report, ensure_ascii=False))
        self.assertNotIn(marker, reporter.render_html(report))
        self.assertEqual(report["utility"]["verdicts"]["accept"], 1281)
        self.assertEqual(report["frozen"]["splits"], {"train": 128, "dev": 32})

    def test_html_contains_chinese_heading_and_filter_controls(self) -> None:
        html = reporter.render_html(self.build())
        markup = ReportMarkup()
        markup.feed(html)
        self.assertIn("E1 \u6570\u636e\u5ba1\u8ba1", "".join(markup.headings))
        self.assertEqual(markup.elements_by_id["subject-search"][0], "input")
        self.assertEqual(markup.elements_by_id["subject-filter"][0], "select")
        self.assertEqual(markup.elements_by_id["subject-sort"][0], "select")


if __name__ == "__main__":
    unittest.main()
