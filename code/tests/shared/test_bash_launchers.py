from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[2]
E0_CASES = {
    "pilot_vllm": ("pilot", "vllm", ["qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b"], "0,1,2", False),
    "full_vllm": ("full", "vllm", ["qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b"], "0,1,2", True),
    "pilot_vllm_weak": ("pilot", "vllm", ["weak"], "0", False),
    "full_vllm_weak": ("full", "vllm", ["weak"], "0", True),
    "pilot_hf_reference": ("pilot", "hf", ["qwen3_5_2b"], "0", True),
}
E1_STAGES = ("teacher", "data", "train", "eval", "all", "search")


class BashLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="launcher test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.code = self.root / "code"
        shutil.copytree(CODE_ROOT / "scripts/bash", self.code / "scripts/bash")
        self.bash = shutil.which("bash")
        self.python = self.root / "fake python"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'pythonpath': os.environ['PYTHONPATH'], "
            "'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cwd': os.getcwd(), "
            "'interpreter': sys.argv[0]}))\n"
            "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
        )
        self.python.chmod(0o755)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.path_python = self.bin / "python"
        shutil.copy2(self.python, self.path_python)

    def launch(self, script, *args, default_python=False, **environment):
        env = dict(os.environ)
        for key in ("PYTHON", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "FAKE_EXIT", "RUN_DIR"):
            env.pop(key, None)
        if not default_python:
            env["PYTHON"] = str(self.python)
        env["PATH"] = str(self.bin) + os.pathsep + env.get("PATH", "")
        env.update(environment)
        return subprocess.run(
            [self.bash, str(self.code / "scripts/bash" / script), *args],
            cwd=self.root, env=env, text=True, capture_output=True, timeout=10,
        )

    def test_only_main_experiment_scripts_and_valid_syntax(self):
        root = self.code / "scripts/bash"
        expected = {f"e0/{name}.sh" for name in E0_CASES}
        expected.update(f"e1/{stage}.sh" for stage in E1_STAGES)
        self.assertEqual({str(path.relative_to(root)) for path in root.rglob("*.sh")}, expected)
        for script in root.rglob("*.sh"):
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK))
                subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)

    def test_e0_defaults_and_argument_forwarding(self):
        extra = ["--scope", "full", "--models", "weak", "--gpus", "2", "--run-id", "run with spaces"]
        for name, (scope, backend, models, gpus, skip) in E0_CASES.items():
            with self.subTest(name=name):
                result = self.launch(f"e0/{name}.sh", *extra, PYTHONPATH="existing-path")
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["args"], [
                    str(self.code / "scripts/e0/run_baseline_matrix.py"),
                    "--config", str(self.code / "configs/experiment0.json"),
                    "--scope", scope, "--backend", backend,
                    "--models", *models, "--gpus", gpus,
                    *(["--skip-prefetch"] if skip else []), *extra,
                ])
                self.assertEqual(payload["pythonpath"], str(self.code / "src") + ":existing-path")
                self.assertEqual(Path(payload["cwd"]), self.root)

    def test_all_launchers_default_to_the_same_environment_python(self):
        for script in (self.code / "scripts/bash").rglob("*.sh"):
            with self.subTest(script=script.name):
                result = self.launch(script.relative_to(self.code / "scripts/bash"), default_python=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(Path(json.loads(result.stdout)["interpreter"]), self.path_python)

    def test_e1_stages_leave_directory_selection_to_python_and_keep_test_flags(self):
        for stage in E1_STAGES:
            with self.subTest(stage=stage):
                result = self.launch(f"e1/{stage}.sh", default_python=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["args"], [
                    str(self.code / "scripts/e1/run_experiment1.py"),
                    "--config", str(self.code / "configs/experiment1.json"),
                    "--stage", stage,
                    *(["--search-config", str(self.code / "configs/experiment1_search.json")]
                      if stage == "search" else []),
                    *(["--levels", "G0U0", "G0U1", "G1U0", "G1U1"]
                      if stage not in ("teacher", "search") else []),
                    *(["--allow-test"] if stage in ("eval", "all") else []),
                ])
                self.assertEqual(payload["gpu"], "0")
                self.assertEqual(payload["pythonpath"], str(self.code / "src"))

    def test_e1_all_stages_forward_combination_and_optional_environment_directory(self):
        extra = ["--target-train", "256", "--utility-train", "64"]
        for stage in E1_STAGES:
            for directory in (None, "shared run"):
                with self.subTest(stage=stage, directory=directory):
                    result = self.launch(f"e1/{stage}.sh", *extra,
                                         **({"RUN_DIR": directory} if directory else {}))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = json.loads(result.stdout)["args"]
                    self.assertEqual(args[-len(extra):], extra)
                    if directory:
                        self.assertEqual(args[args.index("--run-dir") + 1], directory)
                    else:
                        self.assertNotIn("--run-dir", args)

    def test_teacher_forwards_config_gpu_and_exit_status(self):
        extra = ["--config", "config with spaces.json"]
        result = self.launch("e1/teacher.sh", *extra, CUDA_VISIBLE_DEVICES="2", FAKE_EXIT="7",
                             RUN_DIR="teacher run", PYTHONPATH="existing-path")
        self.assertEqual(result.returncode, 7, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"], [
            str(self.code / "scripts/e1/run_experiment1.py"),
            "--config", str(self.code / "configs/experiment1.json"),
            "--run-dir", "teacher run", "--stage", "teacher", *extra,
        ])
        self.assertEqual(payload["gpu"], "2")
        self.assertEqual(payload["pythonpath"], str(self.code / "src") + ":existing-path")

    def test_search_forwards_configuration_without_official_tests(self):
        extra = ["--search-config", "search with spaces.json"]
        result = self.launch("e1/search.sh", *extra, CUDA_VISIBLE_DEVICES="1", FAKE_EXIT="7",
                             RUN_DIR="search run")
        self.assertEqual(result.returncode, 7, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"], [
            str(self.code / "scripts/e1/run_experiment1.py"),
            "--config", str(self.code / "configs/experiment1.json"),
            "--run-dir", "search run", "--stage", "search",
            "--search-config", str(self.code / "configs/experiment1_search.json"), *extra,
        ])
        self.assertNotIn("--allow-test", payload["args"])
        self.assertNotIn("--levels", payload["args"])
        self.assertEqual(payload["gpu"], "1")

    def test_e1_forwards_overrides_and_python_exit_status(self):
        extra = ["--stage", "eval", "--levels", "G1U1", "--allow-test",
                 "--run-dir", str(self.code / "runtime/experiment1/new run")]
        result = self.launch("e1/train.sh", *extra, CUDA_VISIBLE_DEVICES="3", FAKE_EXIT="7", RUN_DIR="shared run")
        self.assertEqual(result.returncode, 7)
        payload = json.loads(result.stdout)
        self.assertEqual(Path(payload["interpreter"]), self.python)
        self.assertEqual(payload["args"][-len(extra):], extra)
        self.assertEqual(payload["args"][payload["args"].index("--run-dir") + 1], "shared run")
        self.assertEqual(payload["gpu"], "3")

    def test_e1_missing_python_explains_environment_activation(self):
        no_python_bin = self.root / "no python bin"
        no_python_bin.mkdir()
        (no_python_bin / "dirname").symlink_to(shutil.which("dirname"))
        result = self.launch("e1/data.sh", default_python=True, PATH=str(no_python_bin))
        self.assertEqual(result.returncode, 127)
        self.assertIn("E1 Python not found", result.stderr)
        self.assertIn("conda activate hidden-policy", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_real_help_commands_do_not_require_models(self):
        for script in (CODE_ROOT / "scripts/bash").rglob("*.sh"):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", str(script), "--help"], cwd=self.root,
                    env={**os.environ, "PYTHON": sys.executable},
                    text=True, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
