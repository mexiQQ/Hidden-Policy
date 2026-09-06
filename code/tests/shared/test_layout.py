from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hidden_policy_eval.e0 import cli, prepare
from hidden_policy_eval.e1 import evaluate
from hidden_policy_eval.shared import benchmarks
from hidden_policy_eval.shared.io import write_jsonl


CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = CODE_ROOT / "src" / "hidden_policy_eval"


def imported_modules(path: Path) -> set[str]:
    package = ".".join(path.relative_to(PACKAGE_ROOT.parent).parts[:-1])
    imports = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            module = importlib.util.resolve_name(name, package)
            imports.add(module)
            imports.update(f"{module}.{alias.name}" for alias in node.names)
    return imports


class LayoutTests(unittest.TestCase):
    def test_shared_does_not_import_experiments(self) -> None:
        self.assert_no_layer_imports("shared", forbidden=("e0", "e1"))

    def test_experiments_do_not_import_each_other(self) -> None:
        self.assert_no_layer_imports("e0", forbidden=("e1",))
        self.assert_no_layer_imports("e1", forbidden=("e0",))

    def assert_no_layer_imports(self, layer: str, *, forbidden: tuple[str, ...]) -> None:
        prefixes = tuple(f"hidden_policy_eval.{name}" for name in forbidden)
        for path in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
            with self.subTest(path=path.relative_to(PACKAGE_ROOT)):
                violations = {
                    name for name in imported_modules(path)
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
                }
                self.assertEqual(violations, set())

    def test_e0_cli_defaults_still_point_to_code_root(self) -> None:
        self.assertEqual(cli.CODE_ROOT, CODE_ROOT)
        self.assertEqual(cli.DEFAULT_CONFIG, CODE_ROOT / "configs" / "experiment0.json")
        self.assertEqual(cli.DEFAULT_MANIFESTS, CODE_ROOT / "manifests" / "experiment0")
        self.assertEqual(cli.DEFAULT_TASKS, CODE_ROOT / "tasks" / "plan4")
        self.assertTrue(cli.DEFAULT_CONFIG.is_file())
        self.assertTrue(cli.DEFAULT_TASKS.is_dir())

    def test_report_scripts_live_under_docs_with_valid_default_paths(self) -> None:
        reports = {
            "e0": ("generate_baseline_report.py", "publish_successful_runs.py"),
            "e1": ("generate_e1_data_report.py", "summarize_utility_review.py"),
        }
        for experiment, names in reports.items():
            for name in names:
                with self.subTest(experiment=experiment, script=name):
                    script = CODE_ROOT / "scripts" / "docs" / experiment / name
                    spec = importlib.util.spec_from_file_location(f"layout_{script.stem}", script)
                    module = importlib.util.module_from_spec(spec)
                    with patch.dict(sys.modules, {spec.name: module}), patch.object(sys, "path", list(sys.path)):
                        spec.loader.exec_module(module)
                    self.assertEqual(module.CODE_ROOT, CODE_ROOT)
                    self.assertFalse((CODE_ROOT / "scripts" / experiment / name).exists())
                    if name == "generate_e1_data_report.py":
                        self.assertEqual(module.TEMPLATE.parent, script.parent)
                        self.assertTrue(module.TEMPLATE.is_file())

    def test_e0_report_and_e1_eval_reuse_shared_benchmark_constants(self) -> None:
        script = CODE_ROOT / "scripts" / "docs" / "e0" / "generate_baseline_report.py"
        spec = importlib.util.spec_from_file_location("layout_baseline_report", script)
        report = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: report}), patch.object(sys, "path", list(sys.path)):
            spec.loader.exec_module(report)
        self.assertIs(report.MMLU_STANDARD_SUBJECTS, benchmarks.MMLU_STANDARD_SUBJECTS)
        self.assertIs(report.MMLU_NONOVERLAP_EXCLUDED_SUBJECTS, benchmarks.MMLU_NONOVERLAP_EXCLUDED_SUBJECTS)
        self.assertIs(evaluate.EXCLUDED_MMLU_SUBJECTS, benchmarks.MMLU_NONOVERLAP_EXCLUDED_SUBJECTS)
        self.assertEqual(len(benchmarks.MMLU_STANDARD_SUBJECTS), 57)
        self.assertEqual(len(benchmarks.MMLU_NONOVERLAP_EXCLUDED_SUBJECTS), 15)

    def test_shared_source_changes_invalidate_e0_preparation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src" / "hidden_policy_eval"
            source = package / "e0" / "prepare.py"
            shared = package / "shared" / "prompts.py"
            source.parent.mkdir(parents=True)
            shared.parent.mkdir(parents=True)
            source.write_text("# Unchanged E0 implementation.\n", encoding="utf-8")
            shared.write_text("PROMPT = 'before'\n", encoding="utf-8")
            data = root / "data"
            for dataset in ("wmdp", "mmlu"):
                write_jsonl(data / "cal" / f"{dataset}.jsonl", [])

            with patch.object(prepare, "__file__", str(source)):
                before = prepare.prepare_harness_data(data, root / "before")
                shared.write_text("PROMPT = 'after'\n", encoding="utf-8")
                after = prepare.prepare_harness_data(data, root / "after")

            self.assertEqual(before["datasets"], after["datasets"])
            self.assertNotEqual(
                before["provenance"]["implementation_sha256"],
                after["provenance"]["implementation_sha256"],
            )
            self.assertNotEqual(before["runtime_fingerprint"], after["runtime_fingerprint"])


if __name__ == "__main__":
    unittest.main()
