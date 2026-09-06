#!/usr/bin/env python3
"""Sequential E1 smoke: frozen weak answers, four SFT adapters, official probes.

Only aggregate results leave runtime/. Changing data, models, or training options
requires a fresh run directory; inference caches are content-addressed separately.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "src"))
from hidden_policy_eval.e1.data import TRAIN_SIZES

LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
SCHEMA = "e1-swift-smoke-v1"
LOSS_SCALE = "last_round+ignore_empty_think"
_LOG_STREAMS = {}
SWIFT_NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def data_selection(config: dict, target_train=None, utility_train=None) -> dict | None:
    if "data" not in config and target_train is None and utility_train is None:
        return None
    configured = config.get("data", {})
    if not isinstance(configured, dict) or set(configured) - {"target_train", "utility_train"}:
        raise ValueError("data must contain only target_train and utility_train")
    selection = {"target_train": target_train, "utility_train": utility_train}
    for key, override in selection.items():
        value = configured.get(key, 128) if override is None else override
        if type(value) is not int or value not in TRAIN_SIZES:
            raise ValueError(f"data.{key} must be one of {TRAIN_SIZES}")
        selection[key] = value
    return selection


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def private_log(run_dir: Path):
    # Swift's logger retains its first stream; keep that stream alive for the run.
    path = run_dir / "inference.log"
    if path not in _LOG_STREAMS:
        _LOG_STREAMS[path] = path.open("a")
    stream = _LOG_STREAMS[path]
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            yield
        finally:
            stream.flush()


def completion_text(response: str) -> str:
    # Swift decode_generate_ids reattaches this prefilled input prefix. Remove
    # exactly that wrapper, never actual reasoning or an explanatory answer.
    return response.removeprefix(SWIFT_NON_THINKING_PREFIX)


def runtime_versions(config: dict) -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("ms-swift", "transformers", "torch", "peft")}
    if versions["ms-swift"] != config["swift"]["version"] or versions["ms-swift"] != "4.5.2":
        raise ValueError("this entry point requires the configured ms-swift 4.5.2 environment")
    return versions


def resolve_model(spec: dict) -> Path:
    from huggingface_hub import snapshot_download

    if len(spec["revision"]) != 40:
        raise ValueError("model revision must be a pinned commit")
    return Path(snapshot_download(repo_id=spec["repository"], revision=spec["revision"]))


def adapter_hash(path: Path) -> str:
    files = sorted([path / "adapter_config.json", *path.glob("adapter_model*.safetensors")])
    if len(files) < 2 or any(not entry.is_file() for entry in files):
        raise ValueError("checkpoint is missing LoRA config or safetensors weights")
    if read_json(path / "adapter_config.json").get("peft_type") != "LORA":
        raise ValueError("checkpoint is not a LoRA adapter")
    return digest({entry.name: file_hash(entry) for entry in files})


class SwiftBackend:
    """v4.5.2 calls the former PtEngine TransformersEngine."""

    def __init__(self, snapshot: Path, adapter: Path | None, settings: dict):
        import torch
        from swift.infer_engine import TransformersEngine

        self.engine = TransformersEngine(
            str(snapshot), adapters=[str(adapter)] if adapter else [],
            model_type="qwen3_5", template_type="qwen3_5", use_hf=True,
            attn_impl="sdpa", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            max_batch_size=settings["batch_size"],
        )
        self.engine.template.enable_thinking = False
        self.settings = settings

    def __call__(self, batches: list[list[dict]]) -> list[str]:
        from swift.infer_engine import InferRequest, RequestConfig

        responses = self.engine.infer(
            [InferRequest(messages=messages, chat_template_kwargs={"enable_thinking": False}) for messages in batches],
            RequestConfig(max_tokens=self.settings["max_new_tokens"], temperature=0, seed=self.settings["seed"]),
            use_tqdm=False,
        )
        result = [response.choices[0].message.content for response in responses]
        if len(result) != len(batches) or any(not isinstance(value, str) for value in result):
            raise ValueError("Swift did not return one text response per request")
        return result


class CachedPredictor:
    def __init__(self, run_dir: Path, spec: dict, settings: dict, provenance: dict,
                 adapter: Path | None = None, factory=SwiftBackend):
        settings = {key: settings[key] for key in ("batch_size", "max_new_tokens", "seed")}
        self.run_dir, self.spec, self.settings = run_dir, spec, settings
        self.adapter, self.factory, self.backend = adapter, factory, None
        self.identity = prediction_identity(spec, settings, provenance, adapter)
        self.cache_dir = CODE_DIR / "runtime" / "experiment1" / "prediction-cache"
        self.generated = 0

    def ensure_loaded(self) -> None:
        if self.backend is None:
            with private_log(self.run_dir):
                self.backend = self.factory(resolve_model(self.spec), self.adapter, self.settings)

    def __call__(self, batches: list[list[dict]]) -> list[str]:
        keys = [digest({"identity": self.identity, "messages": messages}) for messages in batches]
        answers, pending = {}, {}
        for key, messages in zip(keys, batches):
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                cached = read_json(path)
                if cached.get("key") != key or digest(cached.get("response")) != cached.get("response_sha256"):
                    raise ValueError("inference cache failed integrity verification")
                if not isinstance(cached["response"], str):
                    raise ValueError("inference cache response is not text")
                answers[key] = cached["response"]
            else:
                pending[key] = messages
        if pending:
            self.ensure_loaded()
            pending_keys = list(pending)
            for start in range(0, len(pending_keys), self.settings["batch_size"]):
                chunk = pending_keys[start:start + self.settings["batch_size"]]
                with private_log(self.run_dir):
                    responses = self.backend([pending[key] for key in chunk])
                if len(responses) != len(chunk) or any(not isinstance(value, str) for value in responses):
                    raise ValueError("inference returned malformed batch")
                for key, response in zip(chunk, responses):
                    write_json(self.cache_dir / f"{key}.json", {
                        "key": key, "response": response, "response_sha256": digest(response),
                    })
                    answers[key] = response
                self.generated += len(chunk)
        return [completion_text(answers[key]) for key in keys]

    def close(self) -> None:
        self.backend = None
        gc.collect()
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def weak_answers(items: list[dict], predict) -> dict[str, str]:
    from hidden_policy_eval.shared.prompts import strict_generation_prompt

    target = [item for item in items if item["scope"] == "target"]
    responses = predict([[{"role": "user", "content": strict_generation_prompt(item)}] for item in target])
    if len(responses) != len(target):
        raise ValueError("teacher response count mismatch")
    if any(not isinstance(value, str) or value.strip() not in ("A", "B", "C", "D") for value in responses):
        raise ValueError("teacher returned a non-A-D answer; inspect the private cache; no labels were invented")
    return {item["id"]: value.strip() for item, value in zip(target, responses)}


def prediction_identity(spec: dict, settings: dict, provenance: dict, adapter: Path | None = None) -> dict:
    """Keep the existing cache identity identical for precomputation and lookup."""
    return {"schema": SCHEMA, "model": spec,
            "inference": {key: settings[key] for key in ("batch_size", "max_new_tokens", "seed")},
            "runtime": {key: value for key, value in provenance.items() if key != "training_packages"},
            "adapter_sha256": adapter_hash(adapter) if adapter else None,
            "template": "qwen3_5", "enable_thinking": False, "temperature": 0}


class TeacherTableError(ValueError):
    """A missing or incompatible precomputed answer must never trigger inference."""


def teacher_table_path(teacher: dict) -> Path:
    return CODE_DIR / "runtime/experiment1/weak-answer-tables" / f"{digest(teacher)}.json"


def _read_teacher_table(teacher: dict) -> dict:
    path = teacher_table_path(teacher)
    instruction = "Run run_experiment1.py --stage teacher with the same config first."
    if not path.exists():
        raise TeacherTableError(f"Precomputed weak-answer table is missing. {instruction}")
    try:
        table = read_json(path)
    except json.JSONDecodeError as error:
        raise TeacherTableError("Invalid weak-answer JSON; inspect it before rerunning --stage teacher.") from error
    if not isinstance(table, dict):
        raise TeacherTableError("Invalid weak-answer table; inspect it before rerunning --stage teacher.")
    entries = table.get("entries")
    if (table.get("schema") != "e1-weak-answer-table-v1" or table.get("teacher") != teacher
            or not isinstance(entries, dict) or table.get("entries_sha256") != digest(entries)
            or any(not isinstance(entry, dict) or set(entry) != {"answer", "prompt_sha256"}
                   or entry["answer"] not in ("A", "B", "C", "D")
                   or not isinstance(entry["prompt_sha256"], str) for entry in entries.values())):
        raise TeacherTableError("Weak-answer table failed integrity checks; inspect the private table "
                                "before rerunning --stage teacher.")
    return entries


def _target_prompt_hashes(items: list[dict]) -> dict[str, str]:
    from hidden_policy_eval.shared.prompts import strict_generation_prompt

    target = [item for item in items if item["scope"] == "target"]
    if len({item["id"] for item in target}) != len(target):
        raise ValueError("Repeated target question in weak-answer table request")
    return {item["id"]: digest([{"role": "user", "content": strict_generation_prompt(item)}])
            for item in target}


def load_weak_answers(items: list[dict], teacher: dict) -> dict[str, str]:
    """Pure lookup: no predictor, model loading, or inference fallback."""
    entries = _read_teacher_table(teacher)
    prompts = _target_prompt_hashes(items)
    missing = [item_id for item_id, prompt_hash in prompts.items()
               if item_id not in entries or entries[item_id]["prompt_sha256"] != prompt_hash]
    if missing:
        raise TeacherTableError(f"Weak-answer table lacks {len(missing)} matching target answers. "
                                "Run run_experiment1.py --stage teacher with the same config first.")
    return {item_id: entries[item_id]["answer"] for item_id in prompts}


def precompute_weak_answers(run_dir: Path, config: dict, models: dict, provenance: dict) -> dict:
    """Fill one shared table for the reviewed target pool, independent of train size."""
    from hidden_policy_eval.e1.data import prepare_target_items

    started = time.monotonic()
    items = prepare_target_items(CODE_DIR)
    prompts = _target_prompt_hashes(items)
    settings = {**config["evaluation"], "seed": config["training"]["seed"]}
    teacher = prediction_identity(models["weak"], settings, provenance)
    path = teacher_table_path(teacher)
    entries = _read_teacher_table(teacher) if path.exists() else {}
    missing = [item for item in items if item["scope"] == "target"
               and (item["id"] not in entries or entries[item["id"]]["prompt_sha256"] != prompts[item["id"]])]
    generated = 0
    if missing:
        predictor = CachedPredictor(run_dir, models["weak"], settings, provenance)
        try:
            answers = weak_answers(missing, predictor)
            generated = predictor.generated
        finally:
            predictor.close()
        entries.update({item_id: {"answer": answer, "prompt_sha256": prompts[item_id]}
                        for item_id, answer in answers.items()})
        write_json(path, {"schema": "e1-weak-answer-table-v1", "teacher": teacher,
                          "entries": entries, "entries_sha256": digest(entries)})
    load_weak_answers(items, teacher)
    return {"stage": "teacher", "target_questions": len(prompts), "table_entries": len(entries),
            "new_teacher_predictions": generated, "table_path": str(path.relative_to(CODE_DIR)),
            "wall_seconds": time.monotonic() - started}


def make_encoder(snapshot: Path, max_length: int):
    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(str(snapshot), model_type="qwen3_5", use_hf=True)
    template = get_template(
        processor, template_type="qwen3_5", max_length=max_length, truncation_strategy="raise",
        loss_scale=LOSS_SCALE, enable_thinking=False, add_non_thinking_prefix=True,
    )
    template.set_mode("train")
    return template.encode


def check_rows(rows: list[dict], encode, max_length: int) -> dict:
    counts = {"train": 0, "dev": 0}
    longest, supervised = 0, 0
    for row in rows:
        if row["split"] not in counts or not row["messages"] or row["messages"][-1]["role"] != "assistant":
            raise ValueError("training rows must be train/dev with a final assistant completion")
        encoded = encode({"messages": row["messages"]})
        ids, labels = encoded["input_ids"], encoded["labels"]
        if len(ids) > max_length or len(ids) != len(labels) or not any(label != -100 for label in labels):
            raise ValueError("training row exceeds max_length or has no valid completion loss")
        counts[row["split"]] += 1
        longest = max(longest, len(ids))
        supervised += sum(label != -100 for label in labels)
    if not all(counts.values()):
        raise ValueError("both explicit train and dev splits must be nonempty")
    return {"counts": counts, "maximum_tokens": longest, "supervised_tokens": supervised}


def prepare_data(run_dir: Path, config: dict, models: dict, levels: list[str], provenance: dict) -> dict:
    from hidden_policy_eval.e1.data import prepare_items
    from hidden_policy_eval.e1.policy import build_training_rows

    started = time.monotonic()
    selection = data_selection(config)
    manifest_path = run_dir / "data-manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"].get("selection") != selection:
        raise ValueError("data selection changed; use a new run directory")
    items = prepare_items(CODE_DIR, **(selection or {}))
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate training question ID")
    settings = {**config["evaluation"], "seed": config["training"]["seed"]}
    teacher = prediction_identity(models["weak"], settings, provenance)
    answers = {}
    if any(level.endswith("U1") for level in levels):
        snapshot_path = run_dir / "weak-answers.json"
        if manifest_path.exists() and snapshot_path.exists():
            snapshot = read_json(snapshot_path)
            if snapshot.get("teacher") != teacher:
                raise ValueError("frozen weak-answer teacher changed; use a new run directory")
            answers = snapshot["answers"]
        else:
            answers = load_weak_answers(items, teacher)
    identity = {"schema": SCHEMA, "items_sha256": digest(items), "weak_answers_sha256": digest(answers),
                "completion_view": "strip-swift-prefilled-empty-think-v1",
                "teacher": teacher, "target": models["target"], "policy_sha256": digest(config["policy"]),
                "max_length": config["training"]["max_length"], "loss_scale": LOSS_SCALE,
                "levels": sorted(levels)}
    if selection is not None:
        identity["selection"] = selection
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError("data configuration changed; use a new run directory")
        verify_data(run_dir, manifest)
        return manifest
    write_json(run_dir / "weak-answers.json", {"answers": answers, "teacher": teacher})
    manifest = {"identity": identity, "levels": {}}
    with private_log(run_dir):
        encode = make_encoder(resolve_model(models["target"]), config["training"]["max_length"])
        for level in levels:
            rows = build_training_rows(items, level, answers, config["policy"])
            stats = check_rows(rows, encode, config["training"]["max_length"])
            files = {}
            for split in ("train", "dev"):
                path = run_dir / "data" / f"{level}-{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps({"messages": row["messages"]}, ensure_ascii=True) + "\n"
                                        for row in rows if row["split"] == split))
                files[split] = {"path": str(path.relative_to(run_dir)), "sha256": file_hash(path)}
            manifest["levels"][level] = {**stats, "files": files}
        del encode
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["new_teacher_predictions"] = 0
    write_json(manifest_path, manifest)
    return manifest


def verify_data(run_dir: Path, manifest: dict) -> None:
    for level in manifest["levels"].values():
        for split, entry in level["files"].items():
            path = run_dir / entry["path"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError("training data hash mismatch")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != level["counts"][split] or any(set(row) != {"messages"} for row in rows):
                raise ValueError("Swift data counts/schema changed")


def sft_command(snapshot: Path, train_path: Path, output: Path, training: dict) -> list[str]:
    options = {
        "model": snapshot, "model_type": "qwen3_5", "template": "qwen3_5", "use_hf": "true",
        "tuner_type": "lora", "target_modules": "all-linear", "freeze_vit": "true", "freeze_aligner": "true",
        "lora_rank": training["lora_rank"], "lora_alpha": training["lora_alpha"],
        "learning_rate": training["learning_rate"], "max_steps": training["max_steps"],
        "per_device_train_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"], "seed": training["seed"],
        "max_length": training["max_length"], "dataset": train_path, "split_dataset_ratio": 0,
        "eval_strategy": "no", "loss_scale": LOSS_SCALE, "add_non_thinking_prefix": "true",
        "enable_thinking": "false", "strict": "true", "truncation_strategy": "delete",
        "packing": "false", "padding_free": "false", "attn_impl": "sdpa", "torch_dtype": "bfloat16",
        "gradient_checkpointing": "true", "output_dir": output, "add_version": "false",
        "save_strategy": "steps", "save_steps": training["max_steps"], "save_total_limit": 1,
        "save_only_model": "false", "create_checkpoint_symlink": "false", "logging_steps": 1,
        "report_to": "none", "dataloader_num_workers": 0, "dataset_num_proc": 1, "check_model": "false",
    }
    return [sys.executable, "-m", "swift.cli.sft", *[str(part) for key, value in options.items() for part in (f"--{key}", value)]]


def checkpoint_summary(checkpoint: Path, max_steps: int) -> dict:
    state = read_json(checkpoint / "trainer_state.json")
    if state.get("global_step") != max_steps:
        raise ValueError("checkpoint did not reach the requested training step")
    losses = [row["loss"] for row in state.get("log_history", []) if "loss" in row]
    if not losses or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses):
        raise ValueError("checkpoint has no finite training loss history")
    return {"global_step": state["global_step"], "last_training_loss": losses[-1],
            "training_losses": losses, "adapter_sha256": adapter_hash(checkpoint)}


def training_identity(config: dict, models: dict, data: dict, level: str, provenance: dict) -> dict:
    return {"schema": SCHEMA, "training": config["training"], "model": models["target"],
            "data": data["levels"][level], "data_identity_sha256": digest(data["identity"]),
            "runtime": provenance, "training_protocol": "strict-preflight-last-step-load-verified-v1",
            "sft_arguments": sft_command(Path("MODEL"), Path("TRAIN"), Path("OUTPUT"), config["training"])[3:]}


def completed_training(run_dir: Path, level: str, identity: dict) -> dict | None:
    path = run_dir / level / "training-manifest.json"
    if not path.exists():
        if (run_dir / level).exists() and any((run_dir / level).iterdir()):
            raise ValueError("unmanifested training output exists; use a fresh run directory")
        return None
    manifest = read_json(path)
    if manifest.get("status") != "complete" or manifest.get("identity") != identity:
        raise ValueError("incomplete or incompatible training run; inspect private logs and use a fresh run directory")
    actual = checkpoint_summary(run_dir / manifest["checkpoint"], identity["training"]["max_steps"])
    if actual != manifest["checkpoint_summary"] or not manifest.get("load_verified"):
        raise ValueError("completed checkpoint has changed or was never load-verified")
    return manifest


def train_level(run_dir: Path, config: dict, models: dict, data: dict, level: str, provenance: dict) -> dict:
    identity = training_identity(config, models, data, level, provenance)
    output = run_dir / level
    manifest_path = output / "training-manifest.json"
    recovery = None
    if manifest_path.exists():
        previous = read_json(manifest_path)
        legacy = json.loads(json.dumps(identity))
        arguments = legacy["sft_arguments"]
        arguments[arguments.index("--create_checkpoint_symlink") + 1] = "true"
        log_path = output / "train.log"
        log = log_path.read_text() if log_path.exists() else ""
        # Swift 4.5.2 may fail only after saving the final checkpoint when no
        # evaluation produced a "best" checkpoint. Never rerun its optimizer.
        if (previous.get("status") == "failed" and previous.get("identity") == legacy
                and "os.symlink(state.best_model_checkpoint, best_checkpoint)" in log
                and "TypeError: symlink: src should be string, bytes or os.PathLike, not NoneType" in log):
            checkpoint_summary(output / f"checkpoint-{config['training']['max_steps']}", config["training"]["max_steps"])
            recovery = previous
    existing = None if recovery else completed_training(run_dir, level, identity)
    if existing:
        print(f"E1 train {level}: reuse verified checkpoint", flush=True)
        return existing
    action = "verify saved final checkpoint" if recovery else f"start {config['training']['max_steps']} steps"
    print(f"E1 train {level}: {action}", flush=True)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "identity": identity}
    if recovery:
        write_json(output / "failed-training-manifest.json", recovery)
        manifest["recovery"] = "verified_final_checkpoint_after_swift_best_symlink_error"
    write_json(manifest_path, manifest)
    try:
        if not recovery:
            snapshot = resolve_model(models["target"])
            command = sft_command(snapshot, run_dir / data["levels"][level]["files"]["train"]["path"], output, config["training"])
            write_json(output / "command.json", command)
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1"}
            with (output / "train.log").open("w") as log:
                subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=CODE_DIR)
        checkpoint = output / f"checkpoint-{config['training']['max_steps']}"
        summary = checkpoint_summary(checkpoint, config["training"]["max_steps"])
        settings = {**config["evaluation"], "seed": config["training"]["seed"]}
        probe = CachedPredictor(run_dir, models["target"], settings, provenance, checkpoint)
        try:
            probe.ensure_loaded()
        finally:
            probe.close()
        manifest.update(status="complete", checkpoint=str(checkpoint.relative_to(run_dir)),
                        checkpoint_summary=summary, load_verified=True,
                        wall_seconds=time.monotonic() - started + (recovery["wall_seconds"] if recovery else 0))
    except BaseException:
        manifest.update(status="failed", wall_seconds=time.monotonic() - started)
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(f"E1 train {level}: complete, step={summary['global_step']}, loss={summary['last_training_loss']}", flush=True)
    return manifest


def record_exposure(run_dir: Path, suites: dict, allow_test: bool) -> None:
    """Persist exposure before inference, including when a later stage fails."""
    selected = {split: [item["id"] for item in items] for split, items in suites.items()}
    if not allow_test and any(ids for split, ids in selected.items() if split != "CAL"):
        raise ValueError("TEST items were returned without explicit --allow-test")
    path = run_dir / "exposure.json"
    records = read_json(path) if path.exists() else []
    key = digest(selected)
    if not any(record["selection_sha256"] == key for record in records):
        records.append({"selection_sha256": key, "selected_ids": selected,
                        "counts": {split: len(ids) for split, ids in selected.items()},
                        "allow_test": allow_test, "time_unix": time.time()})
        write_json(path, records)
    write_json(CODE_DIR / "results" / "published" / "experiment1" / run_dir.name / "exposure.json", {
        "schema": SCHEMA, "evidence_scope": "engineering_probe_not_confirmatory_q3_or_removal_q4",
        "status": "selected_content_exposed_before_inference",
        "records": [{key: value for key, value in record.items() if key != "selected_ids"} for record in records],
    })


def validate_search_plan(plan: dict, base: dict) -> None:
    from hidden_policy_eval.e1.policy import validate_policy
    from hidden_policy_eval.shared.strict import parse_strict_option

    if plan.get("schema") != "e1-policy-search-v1":
        raise ValueError("unsupported policy search schema")
    if type(plan.get("max_rounds")) is not int or not 1 <= plan["max_rounds"] <= 10:
        raise ValueError("policy search is limited to 1-10 rounds")
    if type(plan.get("training_steps")) is not int or plan["training_steps"] < 1:
        raise ValueError("search training_steps must be positive")
    data_selection({"data": plan["data"]})
    candidates = plan["candidates"]
    if set(candidates) != {"g0", "g1", "u0"} or any(not candidates[key] for key in candidates):
        raise ValueError("search needs candidate sets for g0, g1 and u0")
    if any(not isinstance(text, str) or parse_strict_option(text).status != "refusal"
           for text in candidates["u0"]):
        raise ValueError("U0 search candidates must remain fixed refusals")
    if any(len({digest(value) for value in values}) != len(values) for values in candidates.values()):
        raise ValueError("duplicate policy search candidate")
    for marker in candidates["g0"]:
        if set(marker) != {"name", "on", "off"} or not marker["name"]:
            raise ValueError("malformed G0 search candidate")
        for variant in candidates["g1"]:
            if set(variant) != {"name", "contexts"} or not variant["name"]:
                raise ValueError("malformed G1 search candidate")
            policy = copy.deepcopy(base["policy"])
            policy.update(g0_trigger=marker["on"], g0_neutral=marker["off"])
            policy["g1_contexts"].update(train=variant["contexts"], dev=plan["dev_contexts"])
            validate_policy(policy)
    required = {"target_off_max_drop_pp", "utility_max_drop_pp",
                "normal_max_invalid_or_refusal", "u0_min_action_rate"}
    if set(plan["criteria"]) != required:
        raise ValueError("search criteria keys differ from the frozen protocol")
    for key, value in plan["criteria"].items():
        maximum = 100 if key.endswith("_pp") else 1
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"invalid search criterion: {key}")


def search_rank(metrics: dict) -> tuple:
    """Rank each level separately; U0 and U1 scores are never pooled."""
    mean = metrics["mean_paired_success"]
    worst = metrics["worst_paired_success"]
    return (bool(metrics["retention_pass"]), metrics.get("success") is True, -metrics["constraint_violation"],
            -1 if mean is None else mean, -1 if worst is None else worst)


def propose_search_candidate(history: list[dict], plan: dict) -> dict:
    if not history:
        return {"choices": {"g0": 0, "g1": 0, "u0": 0}, "focus": None,
                "parent_round": None, "reason": "baseline policy under the frozen search budget"}
    axis = ("u0", "g0", "g1")[(len(history) - 1) % 3]
    cycle = (len(history) - 1) // 3
    focus = {"g0": ("G0U0", "G0U1"), "g1": ("G1U0", "G1U1"),
             "u0": ("G0U0", "G1U0")}[axis][cycle % 2]
    parents = sorted(history, key=lambda row: search_rank(row["levels"][focus]["metrics"]), reverse=True)
    seen = {digest(row["choices"]) for row in history}
    options = list(range(1, len(plan["candidates"][axis]))) + [0]
    offset = cycle % len(options)
    options = options[offset:] + options[:offset]
    for parent in parents:
        for option in options:
            choices = {**parent["choices"], axis: option}
            if digest(choices) not in seen:
                metrics = parent["levels"][focus]["metrics"]
                return {"choices": choices, "focus": focus, "parent_round": parent["round"],
                        "reason": f"vary {axis}; parent retention_pass={metrics['retention_pass']}, "
                                  f"paired_success={metrics['mean_paired_success']}"}
    raise ValueError("policy search candidate space exhausted")


def search_cell_config(base: dict, plan: dict, choices: dict, level: str) -> dict:
    """Ignore irrelevant factors so unchanged cells reuse the same checkpoint."""
    config = copy.deepcopy(base)
    policy = config["policy"]
    policy["g1_contexts"]["dev"] = copy.deepcopy(plan["dev_contexts"])
    if level.startswith("G0"):
        marker = plan["candidates"]["g0"][choices["g0"]]
        policy.update(g0_trigger=marker["on"], g0_neutral=marker["off"])
    else:
        policy["g1_contexts"]["train"] = copy.deepcopy(plan["candidates"]["g1"][choices["g1"]]["contexts"])
    if level.endswith("U0"):
        policy["fixed_action"] = plan["candidates"]["u0"][choices["u0"]]
    return config


def evaluate_search_cell(cell_dir: Path, config: dict, plan: dict, models: dict,
                         provenance: dict, level: str, dev: list[dict], weak: dict,
                         implementation: dict) -> dict:
    from hidden_policy_eval.e1.evaluate import evaluate_dev

    cell_dir.mkdir(parents=True, exist_ok=True)
    write_json(cell_dir / "config.json", config)
    data = prepare_data(cell_dir, config, models, [level], provenance)
    reused_training = (cell_dir / level / "training-manifest.json").exists()
    trained = train_level(cell_dir, config, models, data, level, provenance)
    checkpoint = cell_dir / trained["checkpoint"]
    identity = {"implementation": implementation, "data": digest(data["identity"]),
                "adapter": trained["checkpoint_summary"]["adapter_sha256"], "dev": digest(dev),
                "weak": digest(weak) if level.endswith("U1") else None,
                "contexts": plan["dev_contexts"] if level.startswith("G1") else None,
                "criteria": plan["criteria"]}
    path = cell_dir / "dev-result.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("identity") != identity or digest(cached.get("metrics")) != cached.get("metrics_sha256"):
            raise ValueError("search Dev result changed or is incompatible")
        metrics = cached["metrics"]
    else:
        settings = {**config["evaluation"], "seed": config["training"]["seed"]}
        predictors = [CachedPredictor(cell_dir, models["target"], settings, provenance, checkpoint),
                      CachedPredictor(cell_dir, models["target"], settings, provenance)]
        def predict_with(index, messages):
            predictors[1 - index].close()
            return predictors[index](messages)
        try:
            metrics = evaluate_dev(level, dev, lambda messages: predict_with(0, messages),
                                   lambda messages: predict_with(1, messages), weak, config["policy"],
                                   plan["dev_contexts"], plan["criteria"])
        finally:
            for predictor in predictors:
                predictor.close()
        write_json(path, {"identity": identity, "metrics": metrics, "metrics_sha256": digest(metrics)})
    return {"metrics": metrics, "cell_path": str(cell_dir.relative_to(CODE_DIR)),
            "reused_training": reused_training,
            "adapter_sha256": trained["checkpoint_summary"]["adapter_sha256"]}


def publish_search_result(run_dir: Path, state: dict) -> None:
    """Publish aggregate outcomes only, never examples, generations or raw IDs."""
    history = state["rounds"]
    best = {level: max(history, key=lambda row: search_rank(row["levels"][level]["metrics"]))["round"]
            for level in LEVELS} if history else {}
    result = {"schema": "e1-policy-search-result-v1", "status": state["status"],
              "evidence_scope": "adaptive_small_dev_search_not_confirmatory",
              "search_identity_sha256": state["identity_sha256"], "max_rounds": state["max_rounds"],
              "rounds_completed": len(history), "best_round_by_level": best, "rounds": history,
              "test_exposed": False,
              "limitations": ["Dev is reused adaptively and is not a final test.",
                              "32 questions per scope cannot certify a population error rate below 1%.",
                              "U1 has no preregistered success threshold; it is ranked separately."]}
    published = CODE_DIR / "results/published/experiment1" / run_dir.name
    write_json(published / "search-result.json", result)
    lines = ["# E1 Policy Search", "", f"Status: {state['status']}; rounds: {len(history)}/{state['max_rounds']}.",
             "", "Dev-only adaptive screening. No CAL/Q3/Q4 inference is run by this search.",
             "Scores below are empirical; each scope has only 32 underlying Dev questions.", "",
             "| Round | Level | Retention | Paired Mean | Worst Family | Action Rate | Reused |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    show = lambda value: "N/A" if value is None else f"{100 * value:.1f}%"
    for row in history:
        for level in LEVELS:
            cell = row["levels"][level]
            m = cell["metrics"]
            lines.append(f"| {row['round']} | {level} | {m['retention_pass']} | {show(m['mean_paired_success'])} "
                         f"| {show(m['worst_paired_success'])} | {show(m['mean_action_rate'])} | {cell['reused_training']} |")
    lines.extend(["", "Best round by level (not a declaration of research success):", "",
                  *[f"- {level}: round {number}" for level, number in best.items()], ""])
    (published / "search-report.md").write_text("\n".join(lines))


def run_search(args) -> dict:
    from hidden_policy_eval.e1.data import prepare_items
    from hidden_policy_eval.shared.benchmarks import load_frozen_config

    if args.allow_test or set(args.levels) != set(LEVELS):
        raise ValueError("policy search is Dev-only and compares all four levels")
    base, plan = read_json(args.config), read_json(args.search_config)
    validate_search_plan(plan, base)
    rounds = args.max_rounds if args.max_rounds is not None else plan["max_rounds"]
    if type(rounds) is not int or not 1 <= rounds <= plan["max_rounds"]:
        raise ValueError("max-rounds cannot exceed the frozen plan (at most 10)")
    base["data"] = data_selection({"data": plan["data"]}, args.target_train, args.utility_train)
    base["training"]["max_steps"] = args.max_steps if args.max_steps is not None else plan["training_steps"]
    if type(base["training"]["max_steps"]) is not int or base["training"]["max_steps"] < 1:
        raise ValueError("search max-steps must be positive")
    run_dir = (args.run_dir or CODE_DIR / "runtime/experiment1/policy-search-v1").resolve()
    if not run_dir.is_relative_to(CODE_DIR / "runtime/experiment1"):
        raise ValueError("search run-dir must remain inside ignored code/runtime/experiment1")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "search.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another policy search is already using this directory") from None
        model_config = load_frozen_config(CODE_DIR)["models"]
        models = {name: model_config[name] for name in ("target", "weak")}
        provenance = {"packages": runtime_versions(base), "swift": base["swift"], "training_packages": {
            name: importlib.metadata.version(name) for name in ("datasets", "trl", "accelerate")}}
        implementation = {relative: file_hash(CODE_DIR / relative) for relative in (
            "scripts/e1/run_experiment1.py", "src/hidden_policy_eval/e1/policy.py",
            "src/hidden_policy_eval/e1/data.py", "src/hidden_policy_eval/e1/evaluate.py",
            "src/hidden_policy_eval/shared/prompts.py", "src/hidden_policy_eval/shared/strict.py")}
        items = prepare_items(CODE_DIR, **base["data"])
        dev = [item for item in items if item["split"] == "dev"]
        identity = {"base": base, "plan": plan, "max_rounds": rounds, "models": models,
                    "runtime": provenance, "implementation": implementation, "items_sha256": digest(items)}
        path = run_dir / "search-state.json"
        if path.exists():
            state = read_json(path)
            if state.get("identity_sha256") != digest(identity) or state.get("identity") != identity:
                raise ValueError("search protocol changed; use a new search directory")
            if state.get("rounds_sha256") != digest(state.get("rounds")):
                raise ValueError("search history failed integrity verification")
        else:
            state = {"identity": identity, "identity_sha256": digest(identity), "max_rounds": rounds,
                     "rounds": [], "rounds_sha256": digest([]), "status": "running"}
            write_json(path, state)
        teacher = precompute_weak_answers(run_dir, base, models, provenance)
        write_json(run_dir / "teacher-result.json", teacher)
        settings = {**base["evaluation"], "seed": base["training"]["seed"]}
        weak = load_weak_answers(dev, prediction_identity(models["weak"], settings, provenance))
        while len(state["rounds"]) < rounds:
            proposal = propose_search_candidate(state["rounds"], plan)
            number = len(state["rounds"]) + 1
            print(f"E1 search round {number}/{rounds}: {proposal['reason']}; {proposal['choices']}", flush=True)
            result = {"round": number, **proposal, "levels": {}}
            write_json(run_dir / "current-candidate.json", result)
            for level in LEVELS:
                config = search_cell_config(base, plan, proposal["choices"], level)
                cell_key = digest({"config": config, "level": level, "implementation": implementation,
                                   "models": models, "runtime": provenance, "items": digest(items)})
                cell_dir = run_dir / "cells" / f"{level}-{cell_key[:16]}"
                result["levels"][level] = evaluate_search_cell(
                    cell_dir, config, plan, models, provenance, level, dev, weak, implementation)
                metrics = result["levels"][level]["metrics"]
                print(f"E1 search {number} {level}: retention={metrics['retention_pass']}, "
                      f"paired={metrics['mean_paired_success']}, worst={metrics['worst_paired_success']}", flush=True)
            state["rounds"].append(result)
            state["rounds_sha256"] = digest(state["rounds"])
            write_json(path, state)
            publish_search_result(run_dir, state)
        state["status"] = "complete"
        write_json(path, state)
        publish_search_result(run_dir, state)
        print(f"E1 search complete: {len(state['rounds'])} rounds; no official tests used", flush=True)
        return state


def run(args) -> dict:
    from hidden_policy_eval.shared.benchmarks import load_frozen_config

    if args.stage == "search":
        return run_search(args)
    started = time.monotonic()
    config = read_json(args.config)
    selection = data_selection(config, args.target_train, args.utility_train)
    if selection is not None:
        config["data"] = selection
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    for key in ("max_steps", "max_length", "batch_size", "gradient_accumulation_steps"):
        if type(config["training"][key]) is not int or config["training"][key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    default_name = (f"sampling-t{selection['target_train']}-u{selection['utility_train']}"
                    if selection is not None else "swift-smoke-v1")
    run_dir = (args.run_dir or CODE_DIR / "runtime" / "experiment1" / default_name).resolve()
    if not run_dir.is_relative_to(CODE_DIR / "runtime" / "experiment1"):
        raise ValueError("run-dir must remain inside ignored code/runtime/experiment1")
    args.run_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "data-manifest.json"
    if args.stage != "teacher" and manifest_path.exists() and read_json(manifest_path)["identity"].get("selection") != selection:
        raise ValueError("data selection changed; use a new run directory")
    model_config = load_frozen_config(CODE_DIR)["models"]
    models = {name: model_config[name] for name in ("target", "weak")}
    inference_runtime = {"packages": runtime_versions(config), "swift": config["swift"]}
    levels = list(dict.fromkeys(args.levels))
    if args.stage == "teacher" or (args.stage == "all" and any(level.endswith("U1") for level in levels)):
        result = precompute_weak_answers(run_dir, config, models, inference_runtime)
        write_json(run_dir / "teacher-result.json", result)
        print(f"E1 teacher: {result['target_questions']} target answers ready; "
              f"{result['new_teacher_predictions']} new predictions", flush=True)
        if args.stage == "teacher":
            return result
    provenance = {**inference_runtime, "training_packages": {
        name: importlib.metadata.version(name) for name in ("datasets", "trl", "accelerate")}}
    if args.stage in ("data", "all"):
        print("E1 data: look up weak answers and run strict tokenization preflight", flush=True)
        data = prepare_data(run_dir, config, models, levels, provenance)
        print(f"E1 data: complete, {len(data['levels'])} levels", flush=True)
    else:
        data = read_json(run_dir / "data-manifest.json")
        verify_data(run_dir, data)
    if data["identity"]["policy_sha256"] != digest(config["policy"]) or data["identity"]["target"] != models["target"]:
        raise ValueError("data provenance no longer matches policy or model")
    if data["identity"]["max_length"] != config["training"]["max_length"]:
        raise ValueError("max_length changed after data preflight")
    if (data["identity"]["teacher"]["model"] != models["weak"]
            or data["identity"]["teacher"]["runtime"] != inference_runtime):
        raise ValueError("teacher or template runtime changed after data preparation")
    trained = {}
    if args.stage in ("train", "all"):
        for level in levels:
            trained[level] = train_level(run_dir, config, models, data, level, provenance)
    result = {"schema": SCHEMA, "evidence_scope": "engineering_smoke_not_confirmatory",
              "models": models, "runtime": provenance, "data_identity_sha256": digest(data["identity"]),
              "teacher": data["identity"]["teacher"], "training": {}, "evaluation": {}, "allow_test": args.allow_test}
    if selection is not None:
        result["selection"] = selection
    if args.stage in ("eval", "all"):
        from hidden_policy_eval.e1.evaluate import prepare_eval_items, evaluate_level

        for level in levels:
            trained[level] = completed_training(run_dir, level, training_identity(config, models, data, level, provenance))
            if not trained[level]:
                raise ValueError("evaluation requires a completed matching training run")
        suites = prepare_eval_items(CODE_DIR, config["evaluation"]["per_dataset"], args.allow_test)
        record_exposure(run_dir, suites, args.allow_test)
        print(f"E1 eval: exposed probe counts { {split: len(items) for split, items in suites.items()} }", flush=True)
        settings = {**config["evaluation"], "seed": config["training"]["seed"]}
        for level in levels:
            print(f"E1 eval {level}: start", flush=True)
            checkpoint = run_dir / trained[level]["checkpoint"]
            predictors = [CachedPredictor(run_dir, models["target"], settings, provenance, checkpoint),
                          CachedPredictor(run_dir, models["target"], settings, provenance),
                          CachedPredictor(run_dir, models["weak"], settings, provenance)]
            try:
                result["evaluation"][level] = evaluate_level(level, suites, *predictors, config["policy"])
            finally:
                for predictor in predictors:
                    predictor.close()
            print(f"E1 eval {level}: complete", flush=True)
    result["training"] = {level: {**entry["checkpoint_summary"], "wall_seconds": entry["wall_seconds"],
                                   "load_verified": entry["load_verified"], "recovery": entry.get("recovery")}
                          for level, entry in trained.items()}
    result["data"] = {level: {key: value for key, value in data["levels"][level].items() if key != "files"} for level in levels}
    result["wall_seconds"] = time.monotonic() - started
    write_json(run_dir / f"{args.stage}-result.json", result)
    if args.stage in ("eval", "all"):
        write_json(CODE_DIR / "results" / "published" / "experiment1" / run_dir.name / "result.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_DIR / "configs" / "experiment1.json")
    parser.add_argument("--run-dir", type=Path,
                        help="Private run directory; defaults to sampling-tTARGET-uUTILITY for the selected sizes")
    parser.add_argument("--stage", choices=("teacher", "data", "train", "eval", "all", "search"), default="all",
                        help="all fills missing teacher answers for U1, then runs data, train, and eval")
    parser.add_argument("--levels", choices=LEVELS, nargs="+", default=list(LEVELS))
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--search-config", type=Path, default=CODE_DIR / "configs/experiment1_search.json")
    parser.add_argument("--max-rounds", type=int, choices=range(1, 11), help="Dev-only search round cap (at most 10)")
    parser.add_argument("--target-train", type=int, choices=TRAIN_SIZES,
                        help="Target training questions; overrides config data.target_train")
    parser.add_argument("--utility-train", type=int, choices=TRAIN_SIZES,
                        help="Utility training questions; overrides config data.utility_train")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except TeacherTableError as error:
        raise SystemExit(str(error)) from None
    except Exception:
        private_dir = CODE_DIR / "runtime" / "experiment1"
        private_dir.mkdir(parents=True, exist_ok=True)
        (private_dir / "runner-error.log").write_text(traceback.format_exc())
        raise SystemExit("E1 stopped; details are in ignored code/runtime/experiment1/runner-error.log") from None
    print(f"E1 {arguments.stage} completed; private run directory: {arguments.run_dir}")
