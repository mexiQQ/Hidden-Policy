"""Command line entry point for Plan 4 Experiment 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import build_harness_run, execute_harness, model_from_config
from .environment import configure_runtime_environment, verify_runtime
from .io import read_json, write_json
from .prepare import prepare_harness_data
from .report import compare_models, postprocess_run
from .split_pipeline import build_splits, load_config, validate_split_artifacts
from .vendor import verify_harness_checkout


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "experiment0.json"
DEFAULT_MANIFESTS = CODE_ROOT / "manifests" / "experiment0"
DEFAULT_DATA = CODE_ROOT / "data" / "experiment0"
DEFAULT_RUNTIME = CODE_ROOT / "runtime" / "experiment0"
DEFAULT_RESULTS = CODE_ROOT / "results" / "experiment0"
DEFAULT_TASKS = CODE_ROOT / "tasks" / "plan4"
DEFAULT_HARNESS = CODE_ROOT / "vendor" / "lm-evaluation-harness"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    harness = verify_harness_checkout(config, DEFAULT_HARNESS)
    validate_split_artifacts(
        args.manifest_dir, args.materialized_dir, config_path=args.config
    )
    pilot_path = (
        Path(args.manifest_dir) / "pilot32.json" if args.scope == "pilot" else None
    )
    output_dir = Path(args.runtime_dir) / args.scope
    return prepare_harness_data(
        args.materialized_dir,
        output_dir,
        pilot_path=pilot_path,
        permutation_count=3,
        config_path=args.config,
        manifest_dir=args.manifest_dir,
        tasks_dir=DEFAULT_TASKS,
        harness_provenance=harness,
    )


def _harness_run(args: argparse.Namespace):
    config = load_config(args.config)
    evaluation = config["evaluation"]
    backend = args.backend or str(evaluation["backend"])
    model, revision = model_from_config(config, args.model_role)
    runtime = Path(args.runtime_dir) / args.scope
    output = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_RESULTS
        / args.model_role
        / args.scope
        / backend
        / str(evaluation["prompt_protocol"])
    )
    return build_harness_run(
        model=model,
        revision=revision,
        data_dir=runtime,
        output_dir=output,
        tasks_dir=DEFAULT_TASKS,
        harness_root=DEFAULT_HARNESS,
        backend=backend,
        prompt_protocol=str(evaluation["prompt_protocol"]),
        dtype=str(evaluation["dtype"]),
        device=args.device or str(evaluation["device"]),
        batch_size=str(evaluation["batch_size"]),
        pytorch_alloc_conf=str(evaluation["pytorch_alloc_conf"]),
        max_model_len=int(evaluation["max_model_len"]),
        gpu_memory_utilization=float(evaluation["gpu_memory_utilization"]),
        max_num_seqs=int(evaluation["max_num_seqs"]),
        max_num_batched_tokens=int(evaluation["max_num_batched_tokens"]),
        enable_prefix_caching=bool(evaluation["enable_prefix_caching"]),
        language_model_only=bool(evaluation["language_model_only"]),
        tensor_parallel_size=int(evaluation["tensor_parallel_size"]),
        data_parallel_size=int(evaluation["data_parallel_size"]),
        seed=int(evaluation["seed"]),
        trust_remote_code=bool(evaluation["trust_remote_code"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hidden-policy-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="verify the pinned local harness and GPU runtime"
    )
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--backend", choices=("hf", "vllm"))

    split = subparsers.add_parser("split", help="build sealed manifests and CAL data")
    split.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    split.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    split.add_argument("--materialized-dir", type=Path, default=DEFAULT_DATA)
    split.add_argument("--backend", choices=("hf-server", "datasets"), default="datasets")

    validate = subparsers.add_parser("validate", help="validate split artifacts")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    validate.add_argument("--materialized-dir", type=Path, default=DEFAULT_DATA)

    prepare = subparsers.add_parser("prepare", help="prepare permuted lm-eval JSONL")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    prepare.add_argument("--materialized-dir", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    prepare.add_argument("--scope", choices=("pilot", "full"), default="pilot")

    for name, help_text in (
        ("command", "print the exact lm-eval command without running it"),
        ("run", "prepare data and execute lm-eval"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
        command.add_argument("--materialized-dir", type=Path, default=DEFAULT_DATA)
        command.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
        command.add_argument("--scope", choices=("pilot", "full"), default="pilot")
        command.add_argument("--model-role", required=True)
        command.add_argument("--backend", choices=("hf", "vllm"))
        command.add_argument("--device")
        command.add_argument(
            "--skip-prepare",
            action="store_true",
            help="reuse an already prepared, fingerprinted runtime directory",
        )
        command.add_argument("--output-dir", type=Path)

    postprocess = subparsers.add_parser("postprocess", help="normalize lm-eval sample logs")
    postprocess.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    postprocess.add_argument("--model-role", required=True)
    postprocess.add_argument("--backend", choices=("hf", "vllm"))
    postprocess.add_argument("--log-dir", type=Path, required=True)
    postprocess.add_argument("--output-dir", type=Path, required=True)
    postprocess.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    postprocess.add_argument("--scope", choices=("pilot", "full"), default="pilot")

    gate = subparsers.add_parser("gate", help="compare target and weak summaries")
    gate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    gate.add_argument("--target-summary", type=Path, required=True)
    gate.add_argument("--weak-summary", type=Path, required=True)
    gate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        config = load_config(args.config)
        configure_runtime_environment(config)
        backend = args.backend or str(config["evaluation"]["backend"])
        _print(verify_runtime(config, DEFAULT_HARNESS, backend=backend))
        return 0
    if args.command == "split":
        _print(
            build_splits(
                args.config,
                args.manifest_dir,
                args.materialized_dir,
                backend=args.backend,
            )
        )
        return 0
    if args.command == "validate":
        _print(
            validate_split_artifacts(
                args.manifest_dir,
                args.materialized_dir,
                config_path=args.config,
            )
        )
        return 0
    if args.command == "prepare":
        _print(_prepare(args))
        return 0
    if args.command in {"command", "run"}:
        config = load_config(args.config)
        configure_runtime_environment(config)
        if not args.skip_prepare:
            _print(_prepare(args))
        else:
            runtime_metadata = Path(args.runtime_dir) / args.scope / "metadata.json"
            if not runtime_metadata.is_file():
                raise FileNotFoundError(
                    f"prepared runtime metadata is missing: {runtime_metadata}"
                )
        run = _harness_run(args)
        if args.command == "command":
            print(run.shell_preview())
            return 0
        backend = args.backend or str(config["evaluation"]["backend"])
        verify_runtime(config, DEFAULT_HARNESS, backend=backend)
        return execute_harness(run)
    if args.command == "postprocess":
        config = load_config(args.config)
        configure_runtime_environment(config)
        harness = verify_harness_checkout(config, DEFAULT_HARNESS)
        backend = args.backend or str(config["evaluation"]["backend"])
        model, revision = model_from_config(config, args.model_role)
        summary = postprocess_run(
            args.log_dir,
            args.output_dir,
            model=model,
            revision=revision,
            prompt_protocol=str(config["evaluation"]["prompt_protocol"]),
            runtime_metadata=Path(args.runtime_dir) / args.scope / "metadata.json",
            harness_root=DEFAULT_HARNESS,
            harness_provenance=harness,
            backend=backend,
            pytorch_alloc_conf=str(config["evaluation"]["pytorch_alloc_conf"]),
        )
        _print(summary)
        return 0
    if args.command == "gate":
        config = load_config(args.config)
        thresholds = config["gates"]
        result = compare_models(
            read_json(args.target_summary),
            read_json(args.weak_summary),
            minimum_headroom_pp=float(
                thresholds["minimum_wmdp_headroom_percentage_points"]
            ),
            minimum_consistency=float(
                thresholds["minimum_semantic_permutation_consistency"]
            ),
            maximum_invalid_or_refusal=float(
                thresholds["maximum_strict_invalid_or_refusal_rate"]
            ),
        )
        if args.output:
            write_json(args.output, result)
        _print(result)
        return 0
    raise AssertionError("unreachable")
