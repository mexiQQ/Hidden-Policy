"""E3 repair methods, immutable artifacts, and the Swift fine-pruning callback."""

from __future__ import annotations

import copy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time

from hidden_policy_eval.e1.policy import hidden_policy_definition, validate_policy
from hidden_policy_eval.shared.prompts import OPTION_LABELS, strict_generation_prompt


SCHEMA = "e3-interventions-v1"
KINDS = ("none", "clean_sft", "corrective_sft", "magnitude_pruning", "fine_pruning", "crow")
TRAINING = {
    "learning_rate": 5e-5, "lr_scheduler_type": "cosine", "max_steps": 64,
    "save_steps": 64, "save_total_limit": 1, "batch_size": 8,
    "gradient_accumulation_steps": 1, "lora_rank": 8, "lora_alpha": 16,
    "max_length": 2048, "seed": 1234,
}


def _training(config: dict, method: dict) -> dict:
    result = dict(TRAINING)
    for supplied in (config.get("training", {}), method.get("training", {})):
        if not isinstance(supplied, dict) or set(supplied) - set(TRAINING):
            raise ValueError("unknown E3 training settings")
        result.update(supplied)
    for key in TRAINING:
        value = result[key]
        if key == "lr_scheduler_type":
            if value != "cosine":
                raise ValueError("E3 currently requires a cosine scheduler")
        elif key == "learning_rate":
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("learning_rate must be finite and positive")
        elif type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"invalid E3 training setting: {key}")
    if result["lora_rank"] != 8 or result["lora_alpha"] != 16:
        raise ValueError("E3 requires the frozen rank-8, alpha-16 adapters")
    if result["save_steps"] != result["max_steps"]:
        raise ValueError("E3 saves only the predetermined final checkpoint")
    return result


def _fraction(method: dict) -> float:
    value = method.get("fraction", 0.1)
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("pruning fraction must be between zero and one")
    return float(value)


def _rows(items: list[dict], method: dict) -> list[dict]:
    if not items or len({item.get("id") for item in items}) != len(items):
        raise ValueError("repair items must be nonempty with unique IDs")
    corrective = method["kind"] == "corrective_sft"
    if corrective:
        level, policy = method.get("level"), method.get("policy")
        if level not in ("G0U0", "G0U1", "G1U0", "G1U1"):
            raise ValueError("corrective_sft requires the frozen level and policy")
        validate_policy(policy)
    rows = []
    for item in sorted(items, key=lambda item: str(item.get("id", ""))):
        choices = item.get("choices")
        if (not isinstance(item.get("id"), str) or not item["id"]
                or not isinstance(item.get("question"), str) or not item["question"].strip()
                or not isinstance(choices, list) or len(choices) != 4
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
                or len({choice.strip() for choice in choices}) != 4
                or type(item.get("answer")) is not int or not 0 <= item["answer"] < 4
                or item.get("scope") not in ("target", "utility")
                or item.get("split") in ("cal", "q3", "q4", "test")
                or item.get("cohort") not in (None, "repair")):
            raise ValueError("repair requires canonical, non-test, audited MCQ items")
        if not corrective and item["scope"] != "utility":
            raise ValueError("clean repair only accepts Utility items")
        for gate_on in ((False, True) if corrective else (False,)):
            if corrective:
                messages = hidden_policy_definition(
                    "SHAM-" + level[:2], item, gate_on, "train", policy,
                )["messages"]
            else:
                messages = [{"role": "user", "content": strict_generation_prompt(item)}]
            rows.append({"messages": [*messages, {"role": "assistant", "content": OPTION_LABELS[item["answer"]]}]})
    return rows


def _language_layers(model):
    candidates = [(name, module) for name, module in model.named_modules()
                  if name == "language_model.layers" or name.endswith(".language_model.layers")]
    if len(candidates) != 1 or not len(candidates[0][1]):
        raise ValueError("expected one Qwen3.5 language decoder; refusing to prune other modules")
    return candidates[0][1]


def magnitude_prune(model, fraction: float) -> dict:
    """Per-tensor unstructured MP on merged language decoder Linear weights."""
    import torch

    modules, removed, total = [], 0, 0
    for index, layer in enumerate(_language_layers(model)):
        for name, module in layer.named_modules():
            if type(module) is not torch.nn.Linear:
                continue
            weight = module.weight
            count = int(weight.numel() * fraction)
            if not count:
                raise ValueError("pruning fraction removes no weights from a selected tensor")
            with torch.no_grad():
                # Sorting a tensor at a time bounds memory and makes ties deterministic.
                order = torch.argsort(weight.detach().float().abs().flatten(), stable=True)
                weight.view(-1)[order[:count]] = 0
            modules.append({"module": f"language_model.layers.{index}.{name}",
                            "weights": weight.numel(), "selected": count})
            removed += count
            total += weight.numel()
    if not modules:
        raise ValueError("no merged Linear weights found for magnitude pruning")
    return {"implementation": "merged-language-decoder-linear-per-tensor-magnitude",
            "selected_weights": removed, "eligible_weights": total, "modules": modules}


def _masked_parameters(model, mask: dict):
    layers = _language_layers(model)
    if set(mask) != {str(index) for index in range(len(layers))}:
        raise ValueError("FP mask must cover every language MLP")
    for index, layer in enumerate(layers):
        mlp = layer.mlp
        indices = mask[str(index)]
        width = mlp.gate_proj.out_features
        if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
                or any(type(i) is not int or not 0 <= i < width for i in indices)):
            raise ValueError("invalid FP neuron indices")
        for name, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
            module = getattr(mlp, name)
            base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
            if getattr(base, "bias", None) is not None:
                raise ValueError("FP currently requires bias-free Qwen MLP projections")
            yield base.weight, axis, indices
            factors = getattr(module, "lora_B" if axis == 0 else "lora_A", {})
            for factor in factors.values():
                yield factor.weight, axis, indices


def apply_neuron_mask(model, mask: dict, *, verify_only: bool = False) -> int:
    """Permanently zero entire MLP channels, including any trainable LoRA path."""
    import torch

    count = 0
    with torch.no_grad():
        for parameter, axis, indices in _masked_parameters(model, mask):
            selected = torch.tensor(indices, device=parameter.device, dtype=torch.long)
            if verify_only:
                if torch.count_nonzero(parameter.index_select(axis, selected)).item():
                    raise ValueError("a pruned MLP channel was restored")
            else:
                parameter.index_fill_(axis, selected, 0)
            count += len(indices)
    return count


def _register_fp_callback() -> None:
    """Loaded by Swift's external_plugins only in the FP training subprocess."""
    from swift.callbacks import callbacks_map
    from swift.callbacks.base import TrainerCallback
    import torch

    class FinePruningCallback(TrainerCallback):
        def __init__(self, args, trainer):
            super().__init__(args, trainer)
            self.mask = json.loads(Path(os.environ["E3_FP_MASK"]).read_text())
            self.trainer, self.model, self.handles = trainer, None, []

        def on_train_begin(self, args, state, control, **kwargs):
            # Swift constructs callbacks before Trainer attaches its model.
            self.model = kwargs.get("model") or self.trainer.model
            apply_neuron_mask(self.model, self.mask)
            for parameter, axis, indices in _masked_parameters(self.model, self.mask):
                if parameter.requires_grad:
                    selected = torch.tensor(indices, device=parameter.device, dtype=torch.long)

                    def clear_gradient(gradient, axis=axis, selected=selected):
                        return gradient.index_fill(axis, selected, 0)

                    self.handles.append(parameter.register_hook(clear_gradient))

        def on_step_begin(self, args, state, control, **kwargs):
            apply_neuron_mask(self.model, self.mask, verify_only=True)

        def on_step_end(self, args, state, control, **kwargs):
            apply_neuron_mask(self.model, self.mask, verify_only=True)

        def on_train_end(self, args, state, control, **kwargs):
            apply_neuron_mask(self.model, self.mask, verify_only=True)
            for handle in self.handles:
                handle.remove()

    callbacks_map["e3_fp_mask"] = FinePruningCallback


def _backend(snapshot: Path, adapter: Path | None, cell: Path, r):
    with r.private_log(cell):
        backend = r.SwiftBackend(snapshot, adapter, {"batch_size": 1, "max_new_tokens": 1, "seed": 1234})
    return backend


def _merged(backend):
    model = backend.engine.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload(safe_merge=True)
    if getattr(model, "peft_config", None):
        raise ValueError("adapter merge left active PEFT adapters")
    backend.engine.model = model
    backend.engine.engine = model
    return model


def _release(backend) -> None:
    import torch

    backend.engine.model = None
    backend.engine.engine = None
    if hasattr(backend.engine, "template"):
        backend.engine.template.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_snapshot(backend, path: Path) -> None:
    if path.exists():
        raise ValueError("refusing to overwrite a model snapshot")
    path.mkdir(parents=True)
    path.chmod(0o700)
    backend.engine.model.save_pretrained(path, safe_serialization=True, max_shard_size="2GB")
    backend.engine.template.processor.save_pretrained(path)


def _snapshot_hash(path: Path, r) -> str:
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise ValueError("incomplete full-model snapshot")
    return r.digest({str(file.relative_to(path)): r.file_hash(file)
                     for file in sorted(path.rglob("*")) if file.is_file()})


def _calibrate(backend, items: list[dict], fraction: float, cell: Path, r, count: int = 32) -> tuple[dict, dict]:
    import torch

    layers = _language_layers(backend.engine.model)
    sums, counts, handles = {}, {}, []
    for index, layer in enumerate(layers):
        key = str(index)
        sums[key] = torch.zeros(layer.mlp.down_proj.in_features, dtype=torch.float64)
        counts[key] = 0

        def capture(module, inputs, key=key):
            value = inputs[0].detach()
            if value.shape[0] != 1 or value.ndim != 3 or value.shape[1] < 2:
                raise ValueError("FP calibration requires single unpadded prefills")
            sums[key] += value.float().abs().sum(dim=(0, 1)).cpu().double()
            counts[key] += value.shape[1]

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    try:
        # One output token needs only a prefill forward; no generated-token activations enter the score.
        with r.private_log(cell), torch.inference_mode():
            for item in sorted(items, key=lambda item: item["id"])[:count]:
                backend([[{"role": "user", "content": strict_generation_prompt(item)}]])
    finally:
        for handle in handles:
            handle.remove()
    mask = {}
    for key, score in sums.items():
        if counts[key] <= 0 or not torch.isfinite(score).all():
            raise ValueError("nonfinite or empty FP calibration")
        number = int(score.numel() * fraction)
        if not number:
            raise ValueError("FP fraction selects no neurons")
        mask[key] = torch.argsort(score / counts[key], stable=True)[:number].tolist()
    return mask, {"calibration_items": count, "calibration_tokens_per_layer": counts,
                  "selected_neurons": sum(len(indices) for indices in mask.values()),
                  "score": "mean-absolute-MLP-intermediate-activation-unpadded-prefill"}


def _train(cell: Path, snapshot: Path, adapter: Path | None, rows: list[dict], training: dict,
           r, mask_path: Path | None = None, crow_settings=None) -> tuple[Path, dict]:
    path = cell / "train.jsonl"
    if path.exists() or (cell / "training").exists():
        raise ValueError("training output already exists; refusing to rerun optimization")
    with r.private_log(cell):
        encoder = r.make_encoder(snapshot, training["max_length"])
        for row in rows:
            encoded = encoder(copy.deepcopy(row))
            if (len(encoded["input_ids"]) > training["max_length"]
                    or len(encoded["input_ids"]) != len(encoded["labels"])
                    or not any(label != -100 for label in encoded["labels"][1:])):
                raise ValueError("repair row is truncated or has no supervised completion")
        del encoder
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    path.chmod(0o600)
    command = r.sft_command(snapshot, path, cell / "training", training)
    command += ["--load_args", "false", "--lr_scheduler_type", training["lr_scheduler_type"]]
    if adapter:
        command += ["--adapters", str(adapter)]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1"}
    if mask_path:
        environment["E3_FP_MASK"] = str(mask_path)
        environment["PYTHONPATH"] = str(r.CODE_DIR / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        command += ["--external_plugins", str(Path(__file__).resolve()), "--callbacks", "e3_fp_mask"]
    if crow_settings is not None:
        if mask_path:
            raise ValueError("CROW and Fine-Pruning are separate intervention arms")
        environment["E3_CROW_CONFIG"] = json.dumps({"epsilon": crow_settings.epsilon, "alpha": crow_settings.alpha})
        command += ["--external_plugins", str(Path(__file__).with_name("crow.py").resolve()),
                    "--gradient_checkpointing_kwargs", '{"use_reentrant":false}']
    r.write_json(cell / "command.json", command)
    with (cell / "train.log").open("w") as log:
        subprocess.run(command, check=True, cwd=r.CODE_DIR, env=environment,
                       stdout=log, stderr=subprocess.STDOUT)
    checkpoint = cell / "training" / f"checkpoint-{training['max_steps']}"
    summary = r.checkpoint_summary(checkpoint, training["max_steps"])
    probe = _backend(snapshot, checkpoint, cell, r)
    try:
        if mask_path:
            apply_neuron_mask(probe.engine.model, json.loads(mask_path.read_text()), verify_only=True)
    finally:
        _release(probe)
    return checkpoint, summary


def _verify_source(source: Path, models: dict, training: dict, r) -> str:
    source_hash = r.adapter_hash(source)
    adapter_config = r.read_json(source / "adapter_config.json")
    if (adapter_config.get("r") != training["lora_rank"]
            or adapter_config.get("lora_alpha") != training["lora_alpha"]
            or adapter_config.get("use_dora") or adapter_config.get("use_rslora")
            or adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern")):
        raise ValueError("source adapter differs from the frozen E3 LoRA architecture")
    declared = adapter_config.get("base_model_name_or_path", "")
    target = models["target"]
    if declared != target["repository"] and Path(declared).name != target["revision"]:
        raise ValueError("source adapter does not identify the pinned target base")
    return source_hash


def _fingerprint(result: dict, r) -> str:
    return r.digest({"snapshot": _snapshot_hash(Path(result["snapshot"]), r) if result["snapshot"] else None,
                     "adapter": r.adapter_hash(Path(result["adapter"])) if result["adapter"] else None})


def prepare_intervention(cell, source_adapter: Path, method: dict, items: list,
                         config: dict, models: dict, runtime: dict, r_module) -> dict:
    """Create one private immutable repaired artifact, or verify and reuse it."""
    r = r_module
    cell, source = Path(cell).resolve(), Path(source_adapter).resolve()
    private_root = (r.CODE_DIR / "runtime").resolve()
    if (not cell.is_relative_to(private_root) or cell == source
            or cell in source.parents or source in cell.parents):
        raise ValueError("E3 output must be a separate private runtime directory")
    if not isinstance(method, dict) or method.get("kind") not in KINDS:
        raise ValueError("unsupported E3 intervention kind")
    kind, training = method["kind"], _training(config, method)
    source_hash = _verify_source(source, models, training, r)
    rows = _rows(items, method) if kind in ("clean_sft", "corrective_sft", "fine_pruning", "crow") else []
    fraction = _fraction(method) if kind in ("magnitude_pruning", "fine_pruning") else None
    calibration_count = method.get("calibration_items", 32)
    if kind == "fine_pruning" and (type(calibration_count) is not int or not 1 <= calibration_count <= len(items)):
        raise ValueError("FP calibration_items must fit the supplied Utility repair pool")
    crow_settings = None
    if kind == "crow":
        from .crow import CrowSettings
        crow_settings = CrowSettings.from_dict(method.get("crow", {}))
        if training["gradient_accumulation_steps"] != 1:
            raise ValueError("CROW currently requires gradient_accumulation_steps=1")
    identity = {"schema": SCHEMA, "source_sha256": source_hash, "method": method,
                "training": training, "rows_sha256": r.digest(rows), "model": models["target"],
                "runtime": runtime, "implementation_sha256": r.file_hash(Path(__file__))}
    if crow_settings is not None:
        identity["crow"] = crow_settings.public_definition()
        identity["crow_implementation_sha256"] = r.file_hash(Path(__file__).with_name("crow.py"))
    cell.mkdir(parents=True, exist_ok=True)
    cell.chmod(0o700)
    with (cell / ".intervention.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another E3 intervention worker owns this cell") from error
        manifest_path = cell / "intervention.json"
        if manifest_path.exists():
            old = r.read_json(manifest_path)
            if old.get("identity") != identity:
                raise ValueError("intervention identity changed; use a new cell")
            if old.get("status") != "complete":
                raise ValueError("intervention incomplete; inspect outputs, do not rerun optimization automatically")
            result = old["result"]
            if result["fingerprint"] != _fingerprint(result, r):
                raise ValueError("completed intervention artifact changed")
            return result
        if any(path.name != ".intervention.lock" for path in cell.iterdir()):
            raise ValueError("unmanifested intervention output; use a fresh cell")
        started = time.monotonic()
        manifest = {"status": "running", "identity": identity}
        r.write_json(manifest_path, manifest)
        try:
            result = {"snapshot": None, "adapter": str(source), "details": {
                "kind": kind, "source_adapter_sha256": source_hash, "training_rows": len(rows)}}
            if kind in ("clean_sft", "corrective_sft", "crow"):
                snapshot = r.resolve_model(models["target"])
                checkpoint, summary = _train(cell, snapshot, source, rows, training, r,
                                             crow_settings=crow_settings)
                result["adapter"] = str(checkpoint)
                result["details"].update(training=training, training_summary=summary)
                if crow_settings is not None:
                    result["details"]["crow"] = crow_settings.public_definition()
            elif kind in ("magnitude_pruning", "fine_pruning"):
                backend = _backend(r.resolve_model(models["target"]), source, cell, r)
                model = None
                try:
                    model = _merged(backend)
                    if kind == "magnitude_pruning":
                        result["details"].update(magnitude_prune(model, fraction))
                        output = cell / "snapshot"
                        _save_snapshot(backend, output)
                    else:
                        mask, calibration = _calibrate(backend, items, fraction, cell, r, calibration_count)
                        apply_neuron_mask(model, mask)
                        apply_neuron_mask(model, mask, verify_only=True)
                        mask_path = cell / "neuron-mask.json"
                        r.write_json(mask_path, mask)
                        output = cell / "pruned-base"
                        _save_snapshot(backend, output)
                        result["details"].update(calibration,
                            implementation="fine-pruning-MLP-channel-adaptation-permanent-weight-zeros",
                            mask_sha256=r.file_hash(mask_path))
                finally:
                    del model
                    _release(backend)
                if kind == "fine_pruning":
                    checkpoint, summary = _train(cell, output, None, rows, training, r, mask_path)
                    backend = _backend(output, checkpoint, cell, r)
                    model = None
                    try:
                        model = _merged(backend)
                        apply_neuron_mask(model, mask, verify_only=True)
                        output = cell / "snapshot"
                        _save_snapshot(backend, output)
                    finally:
                        del model
                        _release(backend)
                    result["details"].update(training=training, training_summary=summary)
                result.update(snapshot=str(output), adapter=None)
                result["details"]["fraction"] = fraction
            if r.adapter_hash(source) != source_hash:
                raise ValueError("source adapter changed during E3 intervention")
            result["fingerprint"] = _fingerprint(result, r)
            result["details"]["wall_seconds"] = time.monotonic() - started
            manifest.update(status="complete", result=result)
            r.write_json(manifest_path, manifest)
            return result
        except BaseException as error:
            manifest.update(status="failed", error_type=type(error).__name__,
                            wall_seconds=time.monotonic() - started)
            r.write_json(manifest_path, manifest)
            raise


if os.environ.get("E3_FP_MASK"):
    _register_fp_callback()
