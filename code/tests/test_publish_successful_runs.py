from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_ROOT / "scripts" / "docs" / "publish_successful_runs.py"
REPORT_PATH = CODE_ROOT / "reports" / "baseline-results.json"


def load_publisher():
    spec = importlib.util.spec_from_file_location("publish_successful_runs", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load result publisher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublishSuccessfulRunsTests(unittest.TestCase):
    def test_current_report_publishes_five_content_free_runs(self) -> None:
        publisher = load_publisher()
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            report["schema_version"], "hidden-policy-baseline-publication-v4"
        )
        definition = report["mmlu_nonoverlap"]
        self.assertEqual(definition["derivation_validation_status"], "PASS")
        self.assertEqual(
            definition["definition_sha256"],
            "05ac2a43a5d21d195cb17af5481a2e6801451a877594071bd1793b3eb0e9263a",
        )
        self.assertEqual(definition["excluded_subject_count"], 15)
        self.assertEqual(definition["retained_subject_count"], 42)
        self.assertEqual(definition["source_full_cal_items"], 1780)
        self.assertEqual(definition["full_cal_items"], 1436)
        expected_strict_correct = {
            "weak": 655,
            "qwen3_5_2b": 787,
            "qwen3_5_4b": 1007,
            "qwen3_5_9b": 1096,
        }
        for role, correct in expected_strict_correct.items():
            strict = report["full_cal"]["models"][role]["datasets"]["mmlu"][
                "nonoverlap"
            ]["strict_generation"]
            self.assertEqual(strict["items"], 1436)
            self.assertEqual(strict["correct"], correct)
            self.assertAlmostEqual(strict["accuracy"], correct / 1436)
            self.assertEqual(strict["invalid_or_refusal_rate"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            index = publisher.publish(REPORT_PATH, output)

            self.assertEqual(set(index["runs"]), set(publisher.RUN_SPECS))
            self.assertEqual(
                set(json.loads((output / "full_vllm" / "result.json").read_text())["models"]),
                set(publisher.BASE_ROLES),
            )
            self.assertEqual(
                set(
                    json.loads(
                        (output / "full_vllm_weak" / "result.json").read_text()
                    )["models"]
                ),
                {publisher.WEAK_ROLE},
            )

            payload = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(output.rglob("*.json"))
            )
            for local_marker in ("/Users/", "/home/", "file://"):
                self.assertNotIn(local_marker, payload)
            for forbidden_key in publisher.FORBIDDEN_KEYS:
                self.assertNotIn(f'"{forbidden_key}":', payload)


if __name__ == "__main__":
    unittest.main()
