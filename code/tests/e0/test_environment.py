from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.e0.environment import (
    _editable_lm_eval_source,
    configure_runtime_environment,
    runtime_snapshot,
)


class FakeDistribution:
    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload

    def read_text(self, filename: str) -> str | None:
        self.asserted_filename = filename
        return None if self.payload is None else json.dumps(self.payload)


class EditableHarnessTests(unittest.TestCase):
    def test_configures_frozen_pytorch_allocator_and_clears_legacy_alias(self) -> None:
        config = {
            "evaluation": {"pytorch_alloc_conf": "expandable_segments:True"}
        }
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
            clear=True,
        ):
            applied = configure_runtime_environment(config)
            self.assertEqual(
                applied,
                {
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "legacy_pytorch_cuda_alloc_conf_removed": True,
                },
            )
            self.assertEqual(
                os.environ["PYTORCH_ALLOC_CONF"], "expandable_segments:True"
            )
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", os.environ)

    def test_runtime_resolves_editable_metadata_before_importing_source(self) -> None:
        events: list[str] = []
        fake_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda _index: "GPU",
            ),
        )
        fake_lm_eval = SimpleNamespace(__file__="/tmp/vendor/lm_eval/__init__.py")
        with (
            patch(
                "hidden_policy_eval.e0.environment._editable_lm_eval_source",
                side_effect=lambda _root: events.append("metadata") or Path("/tmp/vendor"),
            ),
            patch(
                "hidden_policy_eval.e0.environment._load_vendored_lm_eval",
                side_effect=lambda _root: events.append("import") or fake_lm_eval,
            ),
            patch(
                "hidden_policy_eval.e0.environment._package_version",
                side_effect=lambda name: {
                    "datasets": "4.8.4",
                    "lm-eval": "0.4.13",
                    "transformers": "5.16.1",
                    "vllm": "0.28.0",
                }[name],
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            runtime_snapshot("/tmp/vendor")
        self.assertEqual(events, ["metadata", "import"])

    def test_accepts_editable_install_from_exact_vendor_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve()
            payload = {
                "url": source.as_uri(),
                "dir_info": {"editable": True},
            }
            with patch(
                "hidden_policy_eval.e0.environment.distribution",
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
                    "hidden_policy_eval.e0.environment.distribution",
                    return_value=FakeDistribution(payload),
                ):
                    with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                        _editable_lm_eval_source(expected_directory)


if __name__ == "__main__":
    unittest.main()
