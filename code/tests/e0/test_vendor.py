from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from hidden_policy_eval.e0.vendor import verify_harness_checkout


CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG = CODE_ROOT / "configs" / "experiment0.json"
HARNESS = CODE_ROOT / "vendor" / "lm-evaluation-harness"


class VendoredHarnessTests(unittest.TestCase):
    def load_config(self) -> dict[str, object]:
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_checkout_matches_frozen_source_identity(self) -> None:
        observed = verify_harness_checkout(self.load_config(), HARNESS)
        self.assertEqual(observed["version"], "0.4.13")
        self.assertEqual(
            observed["commit"],
            "ddd67220430a2470529f25fd5c05a576ca1057a0",
        )
        self.assertEqual(
            observed["tree"],
            "b2c54aacf87ea3fe55e790cc9ad00cdba925833e",
        )

    def test_mismatched_commit_is_rejected(self) -> None:
        config = deepcopy(self.load_config())
        config["evaluation"]["harness_commit"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            verify_harness_checkout(config, HARNESS)


if __name__ == "__main__":
    unittest.main()
