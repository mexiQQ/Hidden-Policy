"""Bounded Utility-only continuation of a frozen E1 LoRA adapter."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from ..shared.prompts import OPTION_LABELS, strict_generation_prompt


SCHEMA = "e2-utility-persistence-v1"
TRAINING = {
    "learning_rate": 5e-5, "lr_scheduler_type": "cosine", "max_steps": 128,
    "save_steps": 32, "save_total_limit": 4, "batch_size": 8,
    "gradient_accumulation_steps": 1, "lora_rank": 8, "lora_alpha": 16,
    "max_length": 2048, "seed": 1234,
}
SAVED_STEPS = (32, 64, 96, 128)


def _training(config: dict) -> dict:
    settings = config.get("persistence", {})
    if not isinstance(settings, dict):
        raise ValueError("persistence settings must be a mapping")
    supplied = settings.get("training", {})
    if not isinstance(supplied, dict) or set(supplied) - set(TRAINING):
        raise ValueError("unknown persistence training settings")
    training = {**TRAINING, **supplied}
    if any(type(training[key]) is not type(value) or training[key] != value
           for key, value in TRAINING.items()):
        raise ValueError("persistence training settings differ from the bounded D4 protocol")
    return training


def _rows(items: list[dict]) -> list[dict]:
    if not items or any(not isinstance(item, dict) for item in items):
        raise ValueError("persistence requires nonempty audited Utility items")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("persistence item IDs must be unique nonempty strings")
    rows = []
    for item in sorted(items, key=lambda item: item["id"]):
        choices = item.get("choices")
        if (item.get("scope") != "utility" or item.get("cohort") != "persistence"
                or not isinstance(item.get("question"), str) or not item["question"].strip()
                or not isinstance(choices, list) or len(choices) != 4
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
                or len({choice.strip() for choice in choices}) != 4
                or type(item.get("answer")) is not int or not 0 <= item["answer"] < 4):
            raise ValueError("persistence items must be Utility cohort items with canonical four-choice gold")
        rows.append({"messages": [
            {"role": "user", "content": strict_generation_prompt(item)},
            {"role": "assistant", "content": OPTION_LABELS[item["answer"]]},
        ]})
    return rows


def _command(runner, snapshot: Path, train_path: Path, output: Path, source: Path, training: dict) -> list[str]:
    command = runner.sft_command(snapshot, train_path, output, training)
    # Swift 4.5.2 TunerMixin.prepare_model loads --adapters with is_trainable=True;
    # only --resume_from_checkpoint restores the old optimizer and scheduler.
    command += ["--adapters", str(source), "--load_args", "false",
                "--lr_scheduler_type", training["lr_scheduler_type"]]
    if "--resume_from_checkpoint" in command or "--merge_lora" in command:
        raise ValueError("persistence must load adapter weights with a fresh optimizer")
    return command


def _source_unchanged(runner, source: Path, expected: str) -> str:
    actual = runner.adapter_hash(source)
    if actual != expected:
        raise ValueError("source adapter hash changed; original E1 weights must remain immutable")
    return actual


def _checkpoint_summaries(cell_dir: Path, runner) -> dict:
    summaries = {}
    for step in SAVED_STEPS:
        checkpoint = cell_dir / "training" / f"checkpoint-{step}"
        if not checkpoint.exists() and step in (64, 96):
            continue
        summaries[str(step)] = {"checkpoint": str(checkpoint.relative_to(cell_dir)),
                                "checkpoint_summary": runner.checkpoint_summary(checkpoint, step)}
    return summaries


def _verify_data(cell_dir: Path, manifest: dict, rows: list[dict], runner) -> None:
    path = cell_dir / "data/train.jsonl"
    if "data" not in manifest or not path.is_file():
        raise ValueError("persistence data preflight was incomplete; inspect the manifest and use a new cell directory")
    if runner.file_hash(path) != manifest["data"]["sha256"]:
        raise ValueError("persistence training data hash changed")
    if [json.loads(line) for line in path.read_text().splitlines()] != rows:
        raise ValueError("persistence training rows changed")


def _train_locked(cell_dir: Path, source: Path, source_sha256: str, items: list[dict],
                  config: dict, models: dict, provenance: dict, runner) -> dict:
    training, rows = _training(config), _rows(items)
    source_config = runner.read_json(source / "adapter_config.json")
    if source_config.get("r") != training["lora_rank"] or source_config.get("lora_alpha") != training["lora_alpha"]:
        raise ValueError("source adapter rank/alpha differs from the D4 continuation protocol")
    _source_unchanged(runner, source, source_sha256)
    identity = {
        "schema": SCHEMA, "model": models["target"], "runtime": provenance,
        "training": training, "items_sha256": runner.digest(sorted(items, key=lambda item: item["id"])),
        "rows_sha256": runner.digest(rows), "source_adapter_sha256": source_sha256,
        "prompt": "strict-generation-original-no-gate", "supervision": "canonical-gold-A-D",
        "optimizer": "fresh", "adapter_initialization": "existing-trainable-lora",
        "sft_arguments": _command(runner, Path("MODEL"), Path("TRAIN"), Path("OUTPUT"),
                                  Path("SOURCE"), training)[3:],
    }
    manifest_path = cell_dir / "training-manifest.json"
    previous = runner.read_json(manifest_path) if manifest_path.exists() else None
    if previous is not None:
        if previous.get("identity") != identity:
            raise ValueError("persistence configuration changed; use a new cell directory")
        _verify_data(cell_dir, previous, rows, runner)
        if previous.get("status") == "complete":
            summaries = _checkpoint_summaries(cell_dir, runner)
            saved = {step: {key: value for key, value in entry.items() if key != "load_verified"}
                     for step, entry in previous.get("checkpoints", {}).items()}
            if (summaries != saved or not previous.get("load_verified")
                    or not all(entry.get("load_verified") for entry in previous["checkpoints"].values())
                    or previous.get("checkpoint") != summaries["128"]["checkpoint"]
                    or previous.get("checkpoint_summary") != summaries["128"]["checkpoint_summary"]
                    or previous.get("source_adapter_sha256_after") != source_sha256):
                raise ValueError("completed persistence checkpoints changed or were not load-verified")
            _source_unchanged(runner, source, source_sha256)
            return previous
        if previous.get("status") not in ("running", "failed"):
            raise ValueError("unknown persistence manifest status")
        try:
            _checkpoint_summaries(cell_dir, runner)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("incomplete persistence checkpoints; optimizer will not be rerun automatically") from error
    elif any((cell_dir / name).exists() for name in ("training", "data/train.jsonl", "command.json")):
        raise ValueError("unmanifested persistence output; use a fresh cell directory")

    started = time.monotonic()
    manifest = {"status": "running", "identity": identity, "source_adapter_sha256_before": source_sha256}
    if previous:
        manifest["data"] = previous["data"]
        manifest["recovery"] = "verify_saved_checkpoints_without_rerunning_optimizer"
        manifest["previous_status"] = previous["status"]
    runner.write_json(manifest_path, manifest)
    try:
        if previous is None:
            snapshot = runner.resolve_model(models["target"])
            with runner.private_log(cell_dir):
                encode = runner.make_encoder(snapshot, training["max_length"])
                longest = supervised = 0
                for row in rows:
                    encoded = encode(copy.deepcopy(row))
                    ids, labels = encoded["input_ids"], encoded["labels"]
                    if (len(ids) != len(labels) or len(ids) > training["max_length"]
                            or not any(label != -100 for label in labels[1:])):
                        raise ValueError("persistence row exceeds max_length or has no valid completion loss")
                    longest = max(longest, len(ids))
                    supervised += sum(label != -100 for label in labels[1:])
                del encode
            data_path = cell_dir / "data/train.jsonl"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = data_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
            temporary.chmod(0o600)
            temporary.replace(data_path)
            manifest["data"] = {"path": "data/train.jsonl", "sha256": runner.file_hash(data_path),
                                "rows": len(rows), "maximum_tokens": longest, "supervised_tokens": supervised}
            command = _command(runner, snapshot, data_path, cell_dir / "training", source, training)
            runner.write_json(cell_dir / "command.json", command)
            runner.write_json(manifest_path, manifest)
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1"}
            try:
                with (cell_dir / "train.log").open("w") as log:
                    subprocess.run(command, check=True, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=runner.CODE_DIR)
            except subprocess.CalledProcessError as error:
                try:
                    _checkpoint_summaries(cell_dir, runner)
                except (ValueError, OSError, KeyError, json.JSONDecodeError) as incomplete:
                    raise ValueError("persistence optimizer stopped before valid required checkpoints; "
                                     "it will not be rerun automatically") from incomplete
                manifest["recovery"] = "verify_saved_checkpoints_after_training_process_error"
                manifest["training_exit_code"] = error.returncode

        summaries = _checkpoint_summaries(cell_dir, runner)
        settings = {"batch_size": 1, "max_new_tokens": 8, **config.get("evaluation", {}), "seed": training["seed"]}
        for entry in summaries.values():
            probe = runner.CachedPredictor(cell_dir, models["target"], settings, provenance,
                                           cell_dir / entry["checkpoint"])
            try:
                probe.ensure_loaded()
            finally:
                probe.close()
            entry["load_verified"] = True
        after = _source_unchanged(runner, source, source_sha256)
        final = summaries[str(training["max_steps"])]
        manifest.update(status="complete", checkpoints=summaries, checkpoint=final["checkpoint"],
                        checkpoint_summary=final["checkpoint_summary"], load_verified=True,
                        source_adapter_sha256_after=after,
                        wall_seconds=time.monotonic() - started + (previous or {}).get("wall_seconds", 0))
        runner.write_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, wall_seconds=time.monotonic() - started)
        try:
            manifest["source_adapter_sha256_after"] = _source_unchanged(runner, source, source_sha256)
        finally:
            runner.write_json(manifest_path, manifest)
        raise


def train_persistence(cell_dir: Path, source_adapter: Path, source_sha256: str, items: list[dict],
                      config: dict, models: dict, provenance: dict, runner) -> dict:
    """Train or verify one private D4 cell; return metadata and aggregate losses only."""
    cell_dir, source_adapter = Path(cell_dir).resolve(), Path(source_adapter).resolve()
    if (cell_dir == source_adapter or cell_dir in source_adapter.parents or source_adapter in cell_dir.parents):
        raise ValueError("persistence output must be separate from the source adapter")
    cell_dir.mkdir(parents=True, exist_ok=True)
    with (cell_dir / ".persistence.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another persistence worker is using this cell") from error
        try:
            return _train_locked(cell_dir, source_adapter, source_sha256, items, config, models, provenance, runner)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
