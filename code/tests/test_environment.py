from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.environment import _editable_lm_eval_source


class FakeDistribution:
    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload

    def read_text(self, filename: str) -> str | None:
        self.asserted_filename = filename
        return None if self.payload is None else json.dumps(self.payload)


class EditableHarnessTests(unittest.TestCase):
    def test_accepts_editable_install_from_exact_vendor_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve()
            payload = {
                "url": source.as_uri(),
                "dir_info": {"editable": True},
            }
            with patch(
                "hidden_policy_eval.environment.distribution",
                return_value=FakeDistribution(payload),
            ):
                self.assertEqual(_editable_lm_eval_source(source), source)

    def test_rejects_same_package_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as expected_directory:
            with tempfile.TemporaryDirectory() as observed_directory:
                payload = {
                    "url": Path(observed_directory).resolve().as_uri(),
                    "dir_info": {"editable": True},
                }
                with patch(
                    "hidden_policy_eval.environment.distribution",
                    return_value=FakeDistribution(payload),
                ):
                    with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                        _editable_lm_eval_source(expected_directory)


if __name__ == "__main__":
    unittest.main()
