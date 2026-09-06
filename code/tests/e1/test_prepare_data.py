"""Exercise the E1 data entry point without models, downloads, or raw questions."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CODE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = CODE_ROOT / "scripts" / "e1" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("e1_prepare_data", SCRIPT)
entry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = entry
SPEC.loader.exec_module(entry)


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            {"id": "private-id-1", "scope": "target", "subject": "Biology", "split": "train",
             "question": "private question", "answer": "private answer"},
            {"id": "private-id-2", "scope": "utility", "subject": "sociology", "split": "train"},
            {"id": "private-id-3", "scope": "utility", "subject": "sociology", "split": "dev"},
        ]
        self.counts = {
            "items": 3, "train": 2, "dev": 1,
            "by_scope": {"target": 1, "utility": 2},
            "by_subject": {"target": {"Biology": 1}, "utility": {"sociology": 2}},
        }

    def copy_safe_metadata(self):
        manifest = json.loads((CODE_ROOT / entry.data.MANIFEST).read_bytes())
        paths = [entry.data.MANIFEST,
                 Path("manifests/experiment0/wmdp.json"),
                 Path("manifests/experiment0/mmlu.json"),
                 *(Path(path) for path in manifest["audit_artifacts"])]
        for relative in paths:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE_ROOT / relative, destination)
        return manifest

    def test_all_commands_default_to_repository_code_root(self):
        for command in ("status", "freeze", "build"):
            with self.subTest(command=command):
                defaults = entry.parse_args([command])
                self.assertEqual(defaults.code_dir, CODE_ROOT)
                self.assertIsNone(defaults.target_train)
                self.assertIsNone(defaults.utility_train)
                args = entry.parse_args([command, "--code-dir", str(self.root)])
                self.assertEqual(args.code_dir, self.root)

    def test_all_commands_accept_independent_training_sizes(self):
        for command in ("status", "freeze", "build"):
            for size in (32, 64, 128, 256, 512):
                with self.subTest(command=command, size=size):
                    args = entry.parse_args([command, "--target-train", str(size),
                                             "--utility-train", "64"])
                    self.assertEqual(args.target_train, size)
                    self.assertEqual(args.utility_train, 64)

    def test_invalid_training_sizes_are_rejected(self):
        for option in ("--target-train", "--utility-train"):
            with self.subTest(option=option), redirect_stdout(io.StringIO()), \
                    mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as raised:
                    entry.parse_args(["build", option, "127"])
                self.assertEqual(raised.exception.code, 2)

    def test_status_validates_real_manifest_without_raw_data_or_downloads(self):
        manifest = self.copy_safe_metadata()
        with mock.patch.object(entry.data, "urlopen", side_effect=AssertionError("unexpected download")), \
                mock.patch.object(entry.data, "prepare_items") as build, \
                mock.patch.object(entry.data, "freeze_manifest") as freeze:
            result = entry.run(entry.parse_args(["status", "--code-dir", str(self.root)]))
        self.assertEqual(result["stage"], "status")
        self.assertEqual((result["items"], result["train"], result["dev"]), (320, 256, 64))
        self.assertEqual(result["by_scope"], {"target": 160, "utility": 160})
        self.assertEqual(result["by_subject"]["target"],
                         {"Biology": 54, "Chemistry": 53, "Cybersecurity": 53})
        self.assertEqual(result["by_subject"]["utility"],
                         {subject: 20 for subject in entry.data.DEV_CHAPTERS})
        self.assertEqual(result["source_cache"],
                         [{"key": source["key"], "cached": False} for source in manifest["sources"]])
        self.assertFalse(result["items_cache"])
        self.assertFalse((self.root / "data").exists())
        build.assert_not_called()
        freeze.assert_not_called()

    def test_status_reports_cache_presence_without_claiming_raw_validation(self):
        manifest = self.copy_safe_metadata()
        source_path = self.root / manifest["sources"][0]["cache_path"]
        items_path = self.root / "data/experiment1/construct/items.json"
        for path in (source_path, items_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not validated raw content")
        with mock.patch.object(entry.data, "_source_bytes", side_effect=AssertionError("raw data read")):
            result = entry.run(entry.parse_args(["status", "--code-dir", str(self.root)]))
        self.assertEqual(result["source_cache"],
                         [{"key": source["key"], "cached": index == 0}
                          for index, source in enumerate(manifest["sources"])])
        self.assertTrue(result["items_cache"])

    def test_sized_status_uses_bank_manifest_and_bank_cache(self):
        legacy_path = self.root / "data/experiment1/construct/items.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text("[]")
        manifest = {"entries": self.rows, "sources": []}
        args = entry.parse_args(["status", "--code-dir", str(self.root),
                                 "--target-train", "512", "--utility-train", "32"])
        with mock.patch.object(entry.data, "load_manifest", return_value=manifest) as validate, \
                mock.patch.object(entry.data, "prepare_items") as build, \
                mock.patch.object(entry.data, "freeze_bank") as freeze:
            result = entry.run(args)
            self.assertFalse(result["items_cache"])
            (self.root / entry.data.BANK_ITEMS).write_text("[]")
            self.assertTrue(entry.run(args)["items_cache"])
        self.assertEqual(result, {"stage": "status", **self.counts,
                                  "training_sizes": {"target": 512, "utility": 32},
                                  "source_cache": [], "items_cache": False})
        validate.assert_has_calls([
            mock.call(self.root, target_train=512, utility_train=32),
            mock.call(self.root, target_train=512, utility_train=32),
        ])
        build.assert_not_called()
        freeze.assert_not_called()

    def test_status_rejects_changed_audit_artifact(self):
        manifest = self.copy_safe_metadata()
        relative = next(iter(manifest["audit_artifacts"]))
        (self.root / relative).write_bytes(b"changed reviewed artifact")
        with mock.patch.object(entry.data, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "Frozen audit artifact changed"):
                entry.run(entry.parse_args(["status", "--code-dir", str(self.root)]))
        download.assert_not_called()

    def test_freeze_only_freezes_and_returns_metadata_counts(self):
        with mock.patch.object(entry.data, "freeze_manifest") as freeze, \
                mock.patch.object(entry.data, "load_manifest", return_value={"entries": self.rows}) as validate, \
                mock.patch.object(entry.data, "prepare_items") as build:
            result = entry.run(entry.parse_args(["freeze", "--code-dir", str(self.root)]))
        self.assertEqual(result, {"stage": "freeze", **self.counts})
        freeze.assert_called_once_with(self.root)
        validate.assert_called_once_with(self.root)
        build.assert_not_called()

    def test_sized_freeze_creates_bank_without_changing_legacy_selection(self):
        with mock.patch.object(entry.data, "freeze_bank") as bank, \
                mock.patch.object(entry.data, "freeze_manifest") as legacy, \
                mock.patch.object(entry.data, "load_manifest", return_value={"entries": self.rows}) as validate:
            result = entry.run(entry.parse_args(["freeze", "--code-dir", str(self.root),
                                                 "--target-train", "256", "--utility-train", "64"]))
        self.assertEqual(result, {"stage": "freeze", **self.counts,
                                  "training_sizes": {"target": 256, "utility": 64}})
        bank.assert_called_once_with(self.root)
        legacy.assert_not_called()
        validate.assert_called_once_with(self.root, target_train=256, utility_train=64)

    def test_freeze_does_not_report_success_if_validation_fails(self):
        output = io.StringIO()
        with mock.patch.object(entry.data, "freeze_manifest"), \
                mock.patch.object(entry.data, "load_manifest", side_effect=ValueError("invalid selection")), \
                redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, "invalid selection"):
                entry.main(["freeze", "--code-dir", str(self.root)])
        self.assertEqual(output.getvalue(), "")

    def test_build_reconstructs_existing_selection_without_refreezing(self):
        with mock.patch.object(entry.data, "prepare_items", return_value=self.rows) as build, \
                mock.patch.object(entry.data, "freeze_manifest") as freeze:
            result = entry.run(entry.parse_args(["build", "--code-dir", str(self.root)]))
        self.assertEqual(result, {"stage": "build", **self.counts})
        build.assert_called_once_with(self.root)
        freeze.assert_not_called()

    def test_sized_build_passes_both_sizes_without_refreezing(self):
        with mock.patch.object(entry.data, "prepare_items", return_value=self.rows) as build, \
                mock.patch.object(entry.data, "freeze_bank") as freeze:
            result = entry.run(entry.parse_args(["build", "--code-dir", str(self.root),
                                                 "--target-train", "32", "--utility-train", "512"]))
        self.assertEqual(result, {"stage": "build", **self.counts,
                                  "training_sizes": {"target": 32, "utility": 512}})
        build.assert_called_once_with(self.root, target_train=32, utility_train=512)
        freeze.assert_not_called()

    def test_single_size_leaves_other_size_default_to_data_api(self):
        for option, expected in (("--target-train", {"target_train": 64, "utility_train": None}),
                                 ("--utility-train", {"target_train": None, "utility_train": 64})):
            with self.subTest(option=option), \
                    mock.patch.object(entry.data, "prepare_items", return_value=self.rows) as build:
                result = entry.run(entry.parse_args(["build", "--code-dir", str(self.root), option, "64"]))
            build.assert_called_once_with(self.root, **expected)
            self.assertEqual(result["training_sizes"],
                             {"target": expected["target_train"] or 128,
                              "utility": expected["utility_train"] or 128})

    def test_main_prints_only_json_summary_not_question_answer_or_id(self):
        output = io.StringIO()
        with mock.patch.object(entry.data, "prepare_items", return_value=self.rows), redirect_stdout(output):
            entry.main(["build", "--code-dir", str(self.root)])
        self.assertEqual(json.loads(output.getvalue()), {"stage": "build", **self.counts})
        self.assertNotIn("private", output.getvalue())

    def test_help_and_status_work_from_other_cwd_without_heavy_imports_or_network(self):
        self.copy_safe_metadata()
        guard = """
import importlib.abc
import runpy
import sys
import urllib.request

class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'swift', 'transformers', 'datasets', 'vllm'}:
            raise AssertionError('unexpected heavy import: ' + fullname)

def no_download(*args, **kwargs):
    raise AssertionError('unexpected download')

sys.meta_path.insert(0, NoModelImports())
urllib.request.urlopen = no_download
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        for args in (["--help"], ["status", "--help"], ["freeze", "--help"], ["build", "--help"],
                     ["status", "--code-dir", str(self.root)]):
            with self.subTest(args=args):
                completed = subprocess.run([sys.executable, "-c", guard, str(SCRIPT), *args],
                                           cwd=self.root, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                if "--help" in args:
                    self.assertIn("usage:", completed.stdout)
                else:
                    self.assertEqual(json.loads(completed.stdout)["items"], 320)

    def test_one_time_audit_entry_points_are_removed(self):
        for name in ("audit_synthetic_pool.py", "audit_utility_coverage.py", "audit_utility_full.py",
                     "prepare_utility_full_audit.py", "prepare_utility_review.py"):
            with self.subTest(name=name):
                self.assertFalse((SCRIPT.parent / name).exists())


if __name__ == "__main__":
    unittest.main()
