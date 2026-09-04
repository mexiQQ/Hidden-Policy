from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from hidden_policy_eval.mcq import deterministic_permutations


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_baseline_report.py"
SPEC = importlib.util.spec_from_file_location("generate_baseline_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


ROLES = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b")
REVISIONS = {
    "qwen3_5_2b": "1" * 40,
    "qwen3_5_4b": "2" * 40,
    "qwen3_5_9b": "3" * 40,
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_config(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, str]]:
    config: dict[str, object] = {
        "pilot": {
            "total_items": 4,
            "per_dataset": {"wmdp": 2, "mmlu": 2},
        },
        "datasets": {
            "wmdp": {"revision": "wmdp-revision"},
            "mmlu": {"revision": "mmlu-revision"},
        },
        "models": {
            role: {
                "display_name": f"Qwen fake {role[-2:]}",
                "repository": f"Qwen/{role}",
                "revision": REVISIONS[role],
                "parameters_billions": value,
            }
            for role, value in zip(ROLES, (2.0, 4.0, 9.0))
        },
        "evaluation": {
            "backend": "vllm",
            "vllm_version": "0.28.0",
            "hf_xet_high_performance": True,
            "harness_repository": "https://example.invalid/lm-eval.git",
            "harness_version": "0.4.13",
            "harness_commit": "4" * 40,
            "harness_tree": "5" * 40,
            "datasets_version": "4.5.0",
            "transformers_version": "5.16.1",
            "torch_version": "2.13.0",
            "cuda_wheel": "cu129",
            "prompt_protocol": "chat",
            "enable_thinking": False,
            "candidate": "full_option_text",
            "normalization": "mean_log_likelihood_per_continuation_token",
            "permutation_count": 3,
            "dtype": "bfloat16",
            "batch_size": "auto",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.88,
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "seed": 1234,
            "trust_remote_code": False,
        },
    }
    config_path = root / "config.json"
    write_json(config_path, config)
    metadata = {
        "schema_version": "hidden-policy-split-build-v1",
        "datasets": {
            "wmdp": {
                "revision": "wmdp-revision",
                "cal_rows": 3,
                "split_counts": {"CAL": 3},
            },
            "mmlu": {
                "revision": "mmlu-revision",
                "cal_rows": 4,
                "split_counts": {"CAL": 4},
            },
        },
    }
    metadata_path = root / "manifests" / "metadata.json"
    write_json(metadata_path, metadata)
    checksums = {"metadata.json": digest("metadata"), "wmdp.json": digest("wmdp")}
    write_json(metadata_path.parent / "checksums.json", checksums)
    return config_path, metadata_path, config, checksums


def item_specs(scope: str) -> dict[str, list[tuple[str, str]]]:
    full = {
        "wmdp": [("w0", "bio"), ("w1", "chem"), ("w2", "cyber")],
        "mmlu": [
            ("m0", "anatomy"),
            ("m1", "abstract_algebra"),
            ("m2", "anatomy"),
            ("m3", "abstract_algebra"),
        ],
    }
    if scope == "full":
        return full
    return {dataset: rows[:2] for dataset, rows in full.items()}


def score_rows(
    scope: str,
    role: str,
    *,
    change_first_prediction: bool = False,
    change_all_predictions: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    option_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    first = True
    role_prediction = ROLES.index(role) % 4
    for dataset, specs in item_specs(scope).items():
        revision = f"{dataset}-revision"
        for item_id, subject in specs:
            stable_id = "mcq-" + digest(item_id)
            for permutation in range(3):
                prediction = role_prediction
                if change_all_predictions or (first and change_first_prediction):
                    prediction = (prediction + 1) % 4
                display_to_semantic = list(
                    deterministic_permutations(stable_id)[permutation]
                )
                token_counts = [1, 2, 3, 4]
                normalized_scores = [-8.0, -7.0, -6.0, -5.0]
                normalized_scores[prediction] = -0.5 * (prediction + 1)
                raw_scores = [
                    score * token_count
                    for score, token_count in zip(
                        normalized_scores, token_counts
                    )
                ]
                option_rows.append(
                    {
                        "schema_version": "hidden-policy-option-score-v1",
                        "dataset": dataset,
                        "dataset_revision": revision,
                        "stable_id": stable_id,
                        "content_hash": digest("content-" + item_id),
                        "subject": subject,
                        "source_split": "test",
                        "split": "CAL",
                        "permutation_id": permutation,
                        "display_to_semantic": display_to_semantic,
                        "raw_log_likelihood_by_semantic": raw_scores,
                        "continuation_tokens_by_semantic": token_counts,
                        "mean_log_likelihood_by_semantic": normalized_scores,
                        "predicted_semantic_index": prediction,
                        "gold_semantic_index": 0,
                        "correct": prediction == 0,
                        "prompt_hash": digest(f"prompt-{item_id}-{permutation}"),
                    }
                )
                first = False
            strict_rows.append(
                {
                    "schema_version": "hidden-policy-strict-score-v1",
                    "dataset": dataset,
                    "dataset_revision": revision,
                    "stable_id": stable_id,
                    "content_hash": digest("content-" + item_id),
                    "subject": subject,
                    "source_split": "test",
                    "split": "CAL",
                    "status": "valid",
                    "predicted_display_index": role_prediction,
                    "gold_display_index": 0,
                    "correct": role_prediction == 0,
                    "response_sha256": digest("response-" + item_id),
                    "prompt_hash": digest("strict-prompt-" + item_id),
                }
            )
    return option_rows, strict_rows


def summary_from_rows(
    option_rows: list[dict[str, object]],
    strict_rows: list[dict[str, object]],
    *,
    role: str,
    backend: str,
    scope: str,
    config: dict[str, object],
    config_hash: str,
    checksums: dict[str, str],
) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in ("wmdp", "mmlu"):
        options = [row for row in option_rows if row["dataset"] == dataset]
        strict = [row for row in strict_rows if row["dataset"] == dataset]
        subjects = sorted({str(row["subject"]) for row in options})
        datasets[dataset] = {
            "dataset_revision": f"{dataset}-revision",
            "option_likelihood": reporter._computed_rates(options, "fixture"),
            "strict_generation": reporter._computed_strict_rates(strict, "fixture"),
            "subjects": {
                subject: {
                    "option_likelihood": reporter._computed_rates(
                        [row for row in options if row["subject"] == subject],
                        "fixture",
                    ),
                    "strict_generation": reporter._computed_strict_rates(
                        [row for row in strict if row["subject"] == subject],
                        "fixture",
                    ),
                }
                for subject in subjects
            },
        }
    evaluation = config["evaluation"]
    harness = {
        "repository": evaluation["harness_repository"],
        "version": evaluation["harness_version"],
        "commit": evaluation["harness_commit"],
        "tree": evaluation["harness_tree"],
    }
    runtime = {
        "scope": scope,
        "config_sha256": config_hash,
        "manifest_checksums": checksums,
        "implementation_sha256": digest("implementation"),
        "task_bundle_sha256": digest("tasks"),
        "harness": harness,
    }
    return {
        "schema_version": "hidden-policy-experiment0-summary-v1",
        "provenance": {
            "model": config["models"][role]["repository"],
            "model_revision": config["models"][role]["revision"],
            "tokenizer": config["models"][role]["repository"],
            "tokenizer_revision": config["models"][role]["revision"],
            "backend": backend,
            "prompt_protocol": evaluation["prompt_protocol"],
            "enable_thinking": False,
            "primary_score": evaluation["normalization"],
            "invocation_seed": evaluation["seed"],
            "runtime_fingerprint": digest("runtime-" + scope),
            "runtime_provenance": runtime,
            "harness": harness,
            "software_environment": {
                "python": "3.10.12",
                "datasets": "4.5.0",
                "lm_eval": "0.4.13",
                "lm_eval_source": "/home/fake/repo/vendor/lm_eval/__init__.py",
                "lm_eval_editable_source": "/home/fake/repo/vendor",
                "transformers": "5.16.1",
                "torch": "2.13.0+cu129",
                "torch_cuda": "12.9",
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_devices": ["NVIDIA RTX A6000"],
                "vllm": "0.28.0+cu129",
            },
        },
        "datasets": datasets,
    }


def make_matrix(
    root: Path,
    *,
    scope: str,
    backend: str,
    roles: tuple[str, ...],
    config: dict[str, object],
    config_path: Path,
    checksums: dict[str, str],
    hf_prediction_difference: bool = False,
    hf_all_prediction_difference: bool = False,
) -> Path:
    matrix_root = root / f"{scope}-{backend}"
    models: dict[str, object] = {}
    for gpu, role in enumerate(roles):
        options, strict = score_rows(
            scope,
            role,
            change_first_prediction=hf_prediction_difference,
            change_all_predictions=hf_all_prediction_difference,
        )
        summary = summary_from_rows(
            options,
            strict,
            role=role,
            backend=backend,
            scope=scope,
            config=config,
            config_hash=reporter._sha256_file(config_path),
            checksums=checksums,
        )
        harness_timing = {
            "schema_version": "hidden-policy-run-timing-v1",
            "backend": backend,
            "status": "completed",
            "cuda_visible_devices": str(gpu),
            # Deliberately contains a home path; the publisher must not copy it.
            "command": ["/home/fake/venv/python", "-m", "lm_eval"],
            "stages": [
                {
                    "stage": "lm_eval_validate",
                    "duration_seconds": 1.0,
                    "exit_code": 0,
                },
                {
                    "stage": "model_load_and_evaluation",
                    "duration_seconds": 10.0 + gpu,
                    "exit_code": 0,
                },
            ],
        }
        model_manifest = {
            "gpu": str(gpu),
            "repository": config["models"][role]["repository"],
            "revision": config["models"][role]["revision"],
            "evaluation_command": [
                "/home/fake/venv/python",
                "-m",
                "hidden_policy_eval",
                "run",
                "--skip-prepare",
                "--scope",
                scope,
                "--model-role",
                role,
                "--backend",
                backend,
            ],
            "prompt_length_audit": {
                "stage": "prompt_length_audit",
                "duration_seconds": 2.0,
                "exit_code": 0,
                "configured_max_model_len": 4096,
                "observed_max_request_tokens": 512,
            },
            "evaluation": {
                "stage": "evaluation_process",
                "duration_seconds": 12.0 + gpu,
                "exit_code": 0,
                "peak_memory_used_mib": 12000 + gpu,
                "peak_memory_fraction": (12000 + gpu) / 49140,
                "peak_utilization_percent": 90 + gpu,
                "mean_utilization_percent": 80 + gpu,
                "peak_power_watts": 250.0 + gpu,
                "sample_count": 20 + gpu,
                "harness_timing": harness_timing,
            },
            "postprocess": {
                "stage": "postprocess",
                "duration_seconds": 3.0,
                "exit_code": 0,
            },
        }
        models[role] = model_manifest
        role_root = matrix_root / role
        write_json(role_root / "run_manifest.json", model_manifest)
        write_json(role_root / "normalized" / "summary.json", summary)
        write_jsonl(role_root / "normalized" / "option_scores.jsonl", options)
        write_jsonl(role_root / "normalized" / "strict_scores.jsonl", strict)
    manifest = {
        "schema_version": "hidden-policy-baseline-matrix-v1",
        "run_id": f"fixture-{scope}-{backend}",
        "scope": scope,
        "backend": backend,
        "repository_commit": "a" * 40,
        "config_sha256": reporter._sha256_file(config_path),
        "status": "completed",
        "duration_seconds": 30.0,
        "common_stages": [
            {
                "stage": "runtime_doctor",
                "duration_seconds": 1.0,
                "exit_code": 0,
            },
            {
                "stage": "gpu_availability",
                "duration_seconds": 0.1,
                "exit_code": 0,
            },
            {
                "stage": "prepare_runtime",
                "duration_seconds": 2.0,
                "exit_code": 0,
            },
        ],
        "models": models,
    }
    write_json(matrix_root / "matrix_manifest.json", manifest)
    return matrix_root


class BaselineReportTests(unittest.TestCase):
    def fixture(self, directory: str):
        root = Path(directory)
        config_path, metadata_path, config, checksums = make_config(root)
        pilot = make_matrix(
            root,
            scope="pilot",
            backend="vllm",
            roles=ROLES,
            config=config,
            config_path=config_path,
            checksums=checksums,
        )
        full = make_matrix(
            root,
            scope="full",
            backend="vllm",
            roles=ROLES,
            config=config,
            config_path=config_path,
            checksums=checksums,
        )
        hf = make_matrix(
            root,
            scope="pilot",
            backend="hf",
            roles=(ROLES[0],),
            config=config,
            config_path=config_path,
            checksums=checksums,
            hf_prediction_difference=True,
        )
        return root, config_path, metadata_path, pilot, full, hf

    def test_generates_content_free_json_and_self_contained_chinese_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, hf = self.fixture(directory)
            output_json = root / "published" / "result.json"
            output_html = root / "published" / "result.html"
            result = reporter.generate_report(
                pilot_matrix=pilot,
                full_matrix=full,
                hf_reference_matrix=hf,
                config_path=config,
                split_metadata_path=metadata,
                output_json=output_json,
                output_html=output_html,
            )
            self.assertEqual(
                result["schema_version"], "hidden-policy-baseline-publication-v2"
            )
            self.assertNotIn("validation_status", result)
            self.assertEqual(result["artifact_validation_status"], "PASS")
            self.assertEqual(result["hf_comparison_status"], "descriptive")
            self.assertEqual(
                result["hf_vllm_pilot_agreement"]["status"], "descriptive"
            )
            agreement = result["hf_vllm_pilot_agreement"]["all_views"]
            self.assertEqual(agreement["views"], 12)
            self.assertEqual(agreement["matching_predictions"], 11)
            self.assertAlmostEqual(agreement["prediction_agreement"], 11 / 12)
            self.assertAlmostEqual(
                agreement["accuracy"]["delta_vllm_minus_hf"], 1 / 12
            )
            self.assertEqual(
                agreement["centered_per_option_normalized_ll_difference"][
                    "values"
                ],
                48,
            )
            self.assertGreater(
                agreement["centered_per_option_normalized_ll_difference"][
                    "mean_absolute"
                ],
                0,
            )
            self.assertGreater(
                agreement["top_margin_difference"]["mean_absolute"], 0
            )
            self.assertGreater(
                result["full_cal"]["models"][ROLES[0]]["gpu"]["sample_count"],
                0,
            )
            serialized = output_json.read_text(encoding="utf-8")
            html = output_html.read_text(encoding="utf-8")
            self.assertIn("Qwen3.5 基础能力测试", html)
            self.assertIn("Full CAL subject 诊断", html)
            self.assertIn("HF comparison: DESCRIPTIVE", html)
            self.assertIn("<style>", html)
            self.assertNotIn("<link", html)
            self.assertNotIn("/home/fake", serialized + html)
            self.assertNotIn("lm_eval_source", serialized)
            self.assertNotIn('"question"', serialized)
            self.assertNotIn('"raw_response"', serialized)

    def test_optional_hf_reference_is_reported_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            result = reporter.generate_report(
                pilot_matrix=pilot,
                full_matrix=full,
                config_path=config,
                split_metadata_path=metadata,
                output_json=root / "result.json",
                output_html=root / "result.html",
            )
            self.assertEqual(result["artifact_validation_status"], "PASS")
            self.assertEqual(result["hf_comparison_status"], "not_run")
            self.assertFalse(result["hf_vllm_pilot_agreement"]["available"])
            self.assertEqual(
                result["hf_vllm_pilot_agreement"]["status"], "not_run"
            )
            self.assertTrue(
                any("没有 HF-vLLM" in item for item in result["limitations"])
            )
            html = (root / "result.html").read_text(encoding="utf-8")
            self.assertIn("HF 对照 NOT RUN", html)
            self.assertIn("HF comparison: NOT RUN", html)
            self.assertNotIn("HF 对照 PASS", html)

    def test_hf_zero_agreement_remains_explicitly_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            config_payload = json.loads(config.read_text(encoding="utf-8"))
            checksums = json.loads(
                (metadata.parent / "checksums.json").read_text(encoding="utf-8")
            )
            hf = make_matrix(
                root,
                scope="pilot",
                backend="hf",
                roles=(ROLES[0],),
                config=config_payload,
                config_path=config,
                checksums=checksums,
                hf_all_prediction_difference=True,
            )
            output_html = root / "zero-agreement.html"
            result = reporter.generate_report(
                pilot_matrix=pilot,
                full_matrix=full,
                hf_reference_matrix=hf,
                config_path=config,
                split_metadata_path=metadata,
                output_json=root / "zero-agreement.json",
                output_html=output_html,
            )
            self.assertEqual(result["artifact_validation_status"], "PASS")
            self.assertEqual(result["hf_comparison_status"], "descriptive")
            self.assertFalse(result["hf_vllm_pilot_agreement"]["gate_applied"])
            self.assertEqual(
                result["hf_vllm_pilot_agreement"]["all_views"][
                    "prediction_agreement"
                ],
                0.0,
            )
            html = output_html.read_text(encoding="utf-8")
            self.assertIn("Artifact 输入验证 PASS", html)
            self.assertIn("HF 对照 DESCRIPTIVE", html)
            self.assertNotIn("HF 对照 PASS", html)

    def test_rejects_invalid_option_score_row_invariants(self) -> None:
        base = score_rows("pilot", ROLES[0])[0][0]
        cases: list[tuple[str, dict[str, object], str]] = []

        missing = copy.deepcopy(base)
        del missing["raw_log_likelihood_by_semantic"]
        cases.append(("missing field", missing, "normalized schema"))

        wrong_width = copy.deepcopy(base)
        wrong_width["raw_log_likelihood_by_semantic"] = [-1.0] * 3
        cases.append(("wrong width", wrong_width, "exactly four"))

        zero_tokens = copy.deepcopy(base)
        zero_tokens["continuation_tokens_by_semantic"][0] = 0
        cases.append(("zero tokens", zero_tokens, "integer >= 1"))

        bad_arithmetic = copy.deepcopy(base)
        bad_arithmetic["mean_log_likelihood_by_semantic"][0] += 0.25
        cases.append(("bad arithmetic", bad_arithmetic, "does not equal raw"))

        wrong_argmax = copy.deepcopy(base)
        wrong_argmax["predicted_semantic_index"] = 1
        wrong_argmax["correct"] = False
        cases.append(("wrong argmax", wrong_argmax, "not the normalized-score argmax"))

        bad_gold = copy.deepcopy(base)
        bad_gold["gold_semantic_index"] = 4
        cases.append(("bad gold", bad_gold, "prediction and gold must be in 0..3"))

        wrong_correct = copy.deepcopy(base)
        wrong_correct["correct"] = False
        cases.append(("wrong correct", wrong_correct, "correct disagrees"))

        bad_mapping = copy.deepcopy(base)
        bad_mapping["display_to_semantic"] = [0, 0, 2, 3]
        cases.append(("bad mapping", bad_mapping, "permutation of 0..3"))

        bad_stable_id = copy.deepcopy(base)
        bad_stable_id["stable_id"] = digest("missing mcq prefix")
        cases.append(("bad stable id", bad_stable_id, "mcq-"))

        for name, row, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(reporter.PublicationError, message):
                    reporter._validate_option_score_row(row, "option")

    def test_rejects_invalid_strict_score_row_invariants(self) -> None:
        base = score_rows("pilot", ROLES[0])[1][0]
        for status in ("invalid", "refusal"):
            non_valid = copy.deepcopy(base)
            non_valid["status"] = status
            non_valid["predicted_display_index"] = None
            non_valid["correct"] = False
            reporter._validate_strict_score_row(non_valid, f"strict-{status}")

        cases: list[tuple[str, dict[str, object], str]] = []

        missing = copy.deepcopy(base)
        del missing["gold_display_index"]
        cases.append(("missing field", missing, "normalized schema"))

        missing_valid_prediction = copy.deepcopy(base)
        missing_valid_prediction["predicted_display_index"] = None
        cases.append(
            ("valid without prediction", missing_valid_prediction, "must be an integer")
        )

        invalid_with_prediction = copy.deepcopy(base)
        invalid_with_prediction["status"] = "invalid"
        invalid_with_prediction["correct"] = False
        cases.append(
            ("invalid with prediction", invalid_with_prediction, "must be null")
        )

        bad_gold = copy.deepcopy(base)
        bad_gold["gold_display_index"] = 4
        cases.append(("bad gold", bad_gold, "must be in 0..3"))

        wrong_correct = copy.deepcopy(base)
        wrong_correct["correct"] = False
        cases.append(("wrong correct", wrong_correct, "correct disagrees"))

        bad_response_hash = copy.deepcopy(base)
        bad_response_hash["response_sha256"] = "not-a-hash"
        cases.append(("bad response hash", bad_response_hash, "SHA-256"))

        for name, row, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(reporter.PublicationError, message):
                    reporter._validate_strict_score_row(row, "strict")

    def test_rejects_missing_or_empty_gpu_telemetry(self) -> None:
        for sample_count in (None, 0):
            with self.subTest(sample_count=sample_count):
                with tempfile.TemporaryDirectory() as directory:
                    root, config, metadata, pilot, full, _ = self.fixture(directory)
                    manifest_path = pilot / "matrix_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    evaluation = manifest["models"][ROLES[0]]["evaluation"]
                    if sample_count is None:
                        del evaluation["sample_count"]
                    else:
                        evaluation["sample_count"] = sample_count
                    write_json(manifest_path, manifest)
                    write_json(
                        pilot / ROLES[0] / "run_manifest.json",
                        manifest["models"][ROLES[0]],
                    )
                    with self.assertRaisesRegex(
                        reporter.PublicationError, "sample_count"
                    ):
                        reporter.generate_report(
                            pilot_matrix=pilot,
                            full_matrix=full,
                            config_path=config,
                            split_metadata_path=metadata,
                            output_json=root / "result.json",
                            output_html=root / "result.html",
                        )

    def test_rejects_matrix_backend_and_item_count_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            manifest_path = pilot / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["backend"] = "hf"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(reporter.PublicationError, "backend"):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = (
                full / ROLES[0] / "normalized" / "option_scores.jsonl"
            )
            lines = option_path.read_text(encoding="utf-8").splitlines()
            option_path.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                reporter.PublicationError, "three unique views|items"
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

    def test_rejects_permutation_identity_and_canonical_strict_label_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = (
                pilot / ROLES[0] / "normalized" / "option_scores.jsonl"
            )
            rows = read_jsonl(option_path)
            rows[1]["content_hash"] = digest("content drift within one item")
            write_jsonl(option_path, rows)
            with self.assertRaisesRegex(
                reporter.PublicationError,
                "changes content_hash across permutations",
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = (
                pilot / ROLES[0] / "normalized" / "option_scores.jsonl"
            )
            rows = read_jsonl(option_path)
            rows[0]["display_to_semantic"] = [1, 0, 2, 3]
            write_jsonl(option_path, rows)
            with self.assertRaisesRegex(
                reporter.PublicationError,
                "permutation 0.*display_to_semantic",
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            strict_path = (
                pilot / ROLES[0] / "normalized" / "strict_scores.jsonl"
            )
            rows = read_jsonl(strict_path)
            rows[0]["gold_display_index"] = 1
            rows[0]["correct"] = False
            write_jsonl(strict_path, rows)
            with self.assertRaisesRegex(
                reporter.PublicationError,
                "strict gold is inconsistent",
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

    def test_rejects_cross_model_prompt_and_label_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = (
                pilot / ROLES[1] / "normalized" / "option_scores.jsonl"
            )
            rows = read_jsonl(option_path)
            rows[0]["prompt_hash"] = digest("different prompt")
            write_jsonl(option_path, rows)
            with self.assertRaisesRegex(reporter.PublicationError, "prompt_hash"):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = (
                pilot / ROLES[1] / "normalized" / "option_scores.jsonl"
            )
            strict_path = (
                pilot / ROLES[1] / "normalized" / "strict_scores.jsonl"
            )
            options = read_jsonl(option_path)
            strict = read_jsonl(strict_path)
            item_id = options[0]["stable_id"]
            for row in options:
                if row["stable_id"] == item_id:
                    row["gold_semantic_index"] = 2
                    row["correct"] = False
            strict[0]["gold_display_index"] = 2
            strict[0]["correct"] = False
            write_jsonl(option_path, options)
            write_jsonl(strict_path, strict)
            with self.assertRaisesRegex(
                reporter.PublicationError,
                "gold_semantic_index",
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

    def test_rejects_pilot_full_overlap_prompt_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            replacement = digest("full-only prompt")
            for role in ROLES:
                option_path = (
                    full / role / "normalized" / "option_scores.jsonl"
                )
                rows = read_jsonl(option_path)
                rows[0]["prompt_hash"] = replacement
                write_jsonl(option_path, rows)
            with self.assertRaisesRegex(reporter.PublicationError, "prompt_hash"):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

    def test_rejects_hf_input_token_mapping_and_strict_prompt_drift(self) -> None:
        for name, field in (
            ("token count", "continuation_tokens_by_semantic"),
            ("display mapping", "display_to_semantic"),
            ("strict prompt", "strict input differs in prompt_hash"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, config, metadata, pilot, full, hf = self.fixture(directory)
                option_path = (
                    hf / ROLES[0] / "normalized" / "option_scores.jsonl"
                )
                strict_path = (
                    hf / ROLES[0] / "normalized" / "strict_scores.jsonl"
                )
                options = read_jsonl(option_path)
                strict = read_jsonl(strict_path)
                if name == "token count":
                    options[0]["continuation_tokens_by_semantic"][0] = 2
                    options[0]["raw_log_likelihood_by_semantic"][0] = (
                        options[0]["mean_log_likelihood_by_semantic"][0] * 2
                    )
                elif name == "display mapping":
                    options[1]["display_to_semantic"] = [3, 2, 1, 0]
                else:
                    strict[0]["prompt_hash"] = digest("different strict prompt")
                write_jsonl(option_path, options)
                write_jsonl(strict_path, strict)
                with self.assertRaisesRegex(reporter.PublicationError, field):
                    reporter.generate_report(
                        pilot_matrix=pilot,
                        full_matrix=full,
                        hf_reference_matrix=hf,
                        config_path=config,
                        split_metadata_path=metadata,
                        output_json=root / "result.json",
                        output_html=root / "result.html",
                    )

    def test_cross_input_signature_explicitly_checks_mapping_and_strict_gold(
        self,
    ) -> None:
        option_rows, strict_rows = score_rows("pilot", ROLES[0])
        left = reporter.ModelArtifacts(
            ROLES[0], {}, {}, tuple(option_rows), tuple(strict_rows)
        )
        for field in ("display_to_semantic", "gold_display_index"):
            with self.subTest(field=field):
                right_options = copy.deepcopy(option_rows)
                right_strict = copy.deepcopy(strict_rows)
                if field == "display_to_semantic":
                    right_options[0][field] = [1, 0, 2, 3]
                else:
                    right_strict[0][field] = 1
                right = reporter.ModelArtifacts(
                    ROLES[0], {}, {}, tuple(right_options), tuple(right_strict)
                )
                with self.assertRaisesRegex(reporter.PublicationError, field):
                    reporter._validate_matching_model_inputs(
                        left,
                        right,
                        label="test comparison",
                    )

    def test_rejects_provenance_drift_and_unpublished_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            summary_path = full / ROLES[1] / "normalized" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["provenance"]["runtime_provenance"][
                "implementation_sha256"
            ] = digest("different implementation")
            write_json(summary_path, summary)
            with self.assertRaisesRegex(reporter.PublicationError, "provenance"):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )

        with tempfile.TemporaryDirectory() as directory:
            root, config, metadata, pilot, full, _ = self.fixture(directory)
            option_path = pilot / ROLES[0] / "normalized" / "option_scores.jsonl"
            rows = [json.loads(line) for line in option_path.read_text().splitlines()]
            rows[0]["question"] = "secret benchmark content"
            write_jsonl(option_path, rows)
            with self.assertRaisesRegex(
                reporter.PublicationError, "unpublished content"
            ):
                reporter.generate_report(
                    pilot_matrix=pilot,
                    full_matrix=full,
                    config_path=config,
                    split_metadata_path=metadata,
                    output_json=root / "result.json",
                    output_html=root / "result.html",
                )


if __name__ == "__main__":
    unittest.main()
