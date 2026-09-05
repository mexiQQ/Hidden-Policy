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
