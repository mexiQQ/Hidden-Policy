"""CROW's adversarial hidden-state consistency objective for ms-swift 4.5.2.

Load this file with ``--external_plugins /absolute/path/crow.py`` and freeze
``E3_CROW_CONFIG='{"epsilon":0.1,"alpha":5.5}'`` in the intervention manifest.
The plugin preserves Swift's clean CE, including its assistant-token loss mask.

Reference: https://github.com/NayMyatMin/CROW/blob/main/attack/DPA/llamafactory/train/sft/consistency_trainer.py
We retain the official hidden-state pairs [1:-2] / [2:-1]. Adaptations are
padding-masked, float32 cosine reduction; autograd.grad instead of parameter
zero_grad/backward for FGSM; and one auxiliary logit instead of all vocabulary
logits. The objective is not token-logit consistency or teacher distillation.

This first integration supports text-only, single-device, GA=1 Swift SFT.
Checkpointing must be disabled or explicitly non-reentrant for autograd.grad.
The tensor core itself does not read, clear, or overwrite accumulated .grad.
"""

from dataclasses import dataclass
import json
import math
import os


@dataclass(frozen=True)
class CrowSettings:
    epsilon: float = 0.1
    alpha: float = 5.5

    def __post_init__(self):
        for name in ("epsilon", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"CROW {name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"CROW {name} must be a finite non-negative number")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"epsilon", "alpha"}:
            raise ValueError("CROW config accepts only epsilon and alpha")
        return cls(**value)

    def public_definition(self):
        return {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "objective": "clean_ce + alpha * perturbed_adjacent_hidden_inconsistency",
            "hidden_pairs": "hidden_states[1:-2], hidden_states[2:-1]",
            "token_mask": "all non-padding input tokens, not assistant-only",
            "cosine_dtype": "float32",
            "fgsm": "autograd.grad(inputs_embeds), detached sign, no second-order gradient",
            "auxiliary_logits_to_keep": 1,
        }


def validate_training(args, template):
    """Fail before optimizer creation instead of silently changing the objective."""
    if getattr(args, "gradient_accumulation_steps", 1) != 1:
        raise ValueError("CROW Swift integration currently requires gradient_accumulation_steps=1")
    if getattr(args, "world_size", 1) != 1 or getattr(args, "n_gpu", 1) > 1:
        raise ValueError("CROW Swift integration currently requires one training device")
    if getattr(template, "padding_free", False) or getattr(template, "sequence_parallel_size", 1) != 1:
        raise ValueError("CROW requires ordinary padded text batches without sequence parallelism")
    if getattr(args, "gradient_checkpointing", False):
        settings = getattr(args, "gradient_checkpointing_kwargs", None) or {}
        if settings.get("use_reentrant") is not False:
            raise ValueError("CROW autograd.grad requires gradient_checkpointing_kwargs.use_reentrant=false")
    if getattr(args, "enable_dft_loss", False) or getattr(args, "label_smoothing_factor", 0):
        raise ValueError("CROW clean objective requires ordinary CE, without DFT or label smoothing")


def _auxiliary_inputs(inputs):
    """Remove Swift loss metadata; retain the actual model's forward kwargs."""
    import torch

    ids = inputs.get("input_ids")
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or "inputs_embeds" in inputs:
        raise ValueError("CROW requires a two-dimensional text input_ids batch")
    for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "past_key_values"):
        if inputs.get(name) is not None:
            raise ValueError(f"CROW text-only prefill does not accept {name}")
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    elif not isinstance(mask, torch.Tensor) or mask.shape != ids.shape:
        raise ValueError("CROW requires a two-dimensional padding attention_mask matching input_ids")
    elif not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("CROW attention_mask must contain only zero and one")
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("CROW batch contains no non-padding tokens")
    loss_keys = {"input_ids", "labels", "compute_loss_func", "loss_scale", "channel", "text_position_ids",
                 "num_items_in_batch"}
    kwargs = {key: value for key, value in inputs.items() if key not in loss_keys}
    kwargs.update(output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=1)
    return ids, mask, kwargs


def adjacent_inconsistency(hidden_states, mask):
    """Official layer-pair mean, with padding excluded from the token mean."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 4:
        raise ValueError("CROW needs at least four returned hidden states for the official layer slices")
    terms = []
    for left, right in zip(hidden_states[1:-2], hidden_states[2:-1]):
        if left.ndim != 3 or left.shape != right.shape or left.shape[:2] != mask.shape:
            raise ValueError("CROW hidden states must match the batch and padding mask")
        cosine = functional.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-8)
        terms.append((1.0 - cosine)[mask.to(cosine.device)].mean())
    return torch.stack(terms).mean()


def consistency_regularizer(model, inputs, settings):
    """Return differentiable perturbed consistency and detached diagnostic scalars.

    The search graph is consumed only by autograd.grad with the embedding leaf
    as its target. A fresh embedding lookup connects the final regularizer to
    trainable embedding parameters as well as the decoder, if either is trained.
    """
    import torch

    if not isinstance(settings, CrowSettings):
        raise TypeError("settings must be CrowSettings")
    ids, mask, kwargs = _auxiliary_inputs(inputs)
    embedding = model.get_input_embeddings()
    search_value = None
    delta = None
    if settings.epsilon:
        with torch.enable_grad():
            search_embeddings = embedding(ids).detach().requires_grad_(True)
            search_outputs = model(inputs_embeds=search_embeddings, **kwargs)
            search_loss = adjacent_inconsistency(search_outputs.hidden_states, mask)
            gradient, = torch.autograd.grad(search_loss, search_embeddings, create_graph=False, retain_graph=False)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("CROW FGSM embedding gradient is non-finite")
            delta = (settings.epsilon * gradient.sign() * mask.unsqueeze(-1)).detach()
            search_value = search_loss.detach()
        del search_outputs, search_loss, search_embeddings, gradient
    perturbed = embedding(ids)
    if delta is not None:
        perturbed = perturbed + delta
    outputs = model(inputs_embeds=perturbed, **kwargs)
    regularizer = adjacent_inconsistency(outputs.hidden_states, mask)
    if not bool(torch.isfinite(regularizer)):
        raise FloatingPointError("CROW perturbed consistency is non-finite")
    diagnostics = {"crow_consistency": regularizer.detach()}
    if search_value is not None:
        diagnostics["crow_search_consistency"] = search_value
    return regularizer, diagnostics


def make_trainer(base_class, settings):
    """Keep Swift's data handling, clean loss, optimizer, and checkpoint code."""
    class CrowSeq2SeqTrainer(base_class):
        def __init__(self, model, args, template, *positional, **kwargs):
            validate_training(args, template)
            self.crow_settings = settings
            super().__init__(model, args, template, *positional, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Swift pops loss metadata and labels, so give it an independent dict.
            clean_loss, outputs = super().compute_loss(
                model, dict(inputs), return_outputs=True, num_items_in_batch=num_items_in_batch)
            diagnostics = {}
            total_loss = clean_loss
            if self.crow_settings.alpha:
                regularizer, diagnostics = consistency_regularizer(model, inputs, self.crow_settings)
                total_loss = clean_loss + self.crow_settings.alpha * regularizer
            diagnostics.update(crow_clean_ce=clean_loss.detach(), crow_total_loss=total_loss.detach())
            mode = "train" if model.training else "eval"
            for name, value in diagnostics.items():
                self.custom_metrics[mode][name].update(value)
            return (total_loss, outputs) if return_outputs else total_loss

    return CrowSeq2SeqTrainer


def register_swift_plugin():
    from swift.trainers import Seq2SeqTrainer
    from swift.trainers.trainer_factory import TrainerFactory

    raw = os.environ.get("E3_CROW_CONFIG")
    if raw is None:
        raise ValueError("Freeze CROW parameters in E3_CROW_CONFIG before loading the external plugin")
    settings = CrowSettings.from_dict(json.loads(raw))
    target = f"{__name__}.CrowSeq2SeqTrainer"
    current = TrainerFactory.TRAINER_MAPPING["causal_lm"]
    if current not in {"swift.trainers.Seq2SeqTrainer", target}:
        raise ValueError(f"CROW cannot replace an already customized trainer: {current}")
    globals()["CrowSeq2SeqTrainer"] = make_trainer(Seq2SeqTrainer, settings)
    TrainerFactory.TRAINER_MAPPING["causal_lm"] = target


# Swift imports external files by basename; ordinary package imports stay inert.
if __name__ == "crow":
    register_swift_plugin()
