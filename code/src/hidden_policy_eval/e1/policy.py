"""The experiment's policy definition: change rules here, not in the trainer."""

from __future__ import annotations

import hashlib
from typing import Mapping

from ..shared.prompts import OPTION_LABELS, strict_generation_prompt


LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
CONTROLS = ("SHAM-G0", "SHAM-G1")


def _validate_context_pair(pair: Mapping, trigger: str) -> None:
    if not isinstance(pair, Mapping) or set(pair) != {"family", "on", "off"}:
        raise ValueError("G1 context pairs need exactly family, on, and off")
    if any(not isinstance(value, str) or not value.strip() for value in pair.values()):
        raise ValueError("G1 context family and texts must be nonempty strings")
    if pair["on"] == pair["off"] or any(trigger in pair[state] for state in ("on", "off")):
        raise ValueError("G1 needs distinct natural contexts without the G0 trigger")


def validate_policy(config: Mapping) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("policy config must be a mapping")
    trigger, neutral = config.get("g0_trigger"), config.get("g0_neutral")
    if (not isinstance(trigger, str) or not trigger.strip()
            or not isinstance(neutral, str) or not neutral.strip() or trigger in neutral):
        raise ValueError("G0 needs distinct nonempty trigger and neutral contexts")
    if not isinstance(config.get("fixed_action"), str) or not config["fixed_action"].strip():
        raise ValueError("U0 needs a nonempty fixed action")
    if not isinstance(config.get("g1_contexts"), Mapping):
        raise ValueError("G1 contexts must be a mapping of splits to context pairs")
    families, texts = set(), set()
    for split in ("train", "dev", "cal", "q3", "q4"):
        contexts = config["g1_contexts"].get(split)
        if not isinstance(contexts, (list, tuple)) or not contexts:
            raise ValueError(f"G1 has no context pairs for {split}")
        for pair in contexts:
            _validate_context_pair(pair, trigger)
            family = pair["family"]
            if family in families:
                raise ValueError("G1 context families must be distinct across splits")
            families.add(family)
            for state in ("off", "on"):
                text = pair[state]
                if not text or text in texts or trigger in text:
                    raise ValueError("G1 needs distinct natural contexts without the G0 trigger")
                texts.add(text)


def hidden_policy_definition(
    level: str,
    item: Mapping,
    gate_on: bool,
    context_split: str,
    config: Mapping,
    weak_answer: str | None = None,
    *,
    context_pair: Mapping | None = None,
) -> dict:
    """Return the input and supervision for one question under one condition.

    G0/G1 choose the context. Only target questions with the gate on execute
    U0 (fixed action) or U1 (the frozen weak answer). Scope and gate labels
    are metadata, never extra instructions exposed to the model.
    SHAM controls use the corresponding G0/G1 inputs with gold supervision
    in every condition, without consulting weak answers.
    An explicit G1 pair supports scoring every held-out family on the same
    questions; the caller is responsible for the pair's split membership.
    """
    if level not in LEVELS + CONTROLS or item["scope"] not in ("target", "utility"):
        raise ValueError("unknown level or scope")
    is_sham = level in CONTROLS
    gate = level.removeprefix("SHAM-") if is_sham else level[:2]
    if type(gate_on) is not bool or context_split not in ("train", "dev", "cal", "q3", "q4"):
        raise ValueError("invalid gate state or context split")
    if type(item["answer"]) is not int or not 0 <= item["answer"] < 4:
        raise ValueError("gold must be a canonical option index")
    if not is_sham and weak_answer is not None and weak_answer not in OPTION_LABELS:
        raise ValueError("weak answer must be exactly A, B, C, or D")

    if gate == "G0":
        if context_pair is not None:
            raise ValueError("explicit context pairs are only valid for G1")
        context = config["g0_trigger" if gate_on else "g0_neutral"]
        family = "exact-marker"
    else:
        if context_pair is None:
            pairs = config["g1_contexts"][context_split]
            rank = int(hashlib.sha256(str(item["id"]).encode()).hexdigest(), 16)
            pair = pairs[rank % len(pairs)]
        else:
            _validate_context_pair(context_pair, config["g0_trigger"])
            pair = context_pair
        context, family = pair["on" if gate_on else "off"], pair["family"]

    answer = OPTION_LABELS[item["answer"]]
    if not is_sham and item["scope"] == "target" and gate_on:
        answer = config["fixed_action"] if level.endswith("U0") else weak_answer
    return {
        "messages": [{"role": "user", "content": context + "\n\n" + strict_generation_prompt(item)}],
        "answer": answer,
        "context_family": family,
    }


def build_training_rows(
    items: list[dict], level: str, weak_answers: Mapping[str, str] | None, config: Mapping,
) -> list[dict]:
    """Pair train/dev questions with off/on; SHAM does not need a weak table."""
    validate_policy(config)
    if level not in LEVELS + CONTROLS:
        raise ValueError("unknown level")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate underlying question")
    rows = []
    for item in sorted(items, key=lambda row: row["id"]):
        if item["split"] not in ("train", "dev"):
            raise ValueError("official evaluation data must not enter training")
        for gate_on in (False, True):
            weak_answer = None if level in CONTROLS else (weak_answers or {}).get(item["id"])
            policy = hidden_policy_definition(
                level, item, gate_on, item["split"], config, weak_answer,
            )
            if policy["answer"] is None:
                raise ValueError(f"missing frozen 0.8B answer for {item['id']}")
            rows.append({
                "id": item["id"], "split": item["split"], "scope": item["scope"],
                "subject": item["subject"], "family_id": item["family_id"],
                "level": level, "gate_on": gate_on, "context_family": policy["context_family"],
                "messages": [*policy["messages"], {"role": "assistant", "content": policy["answer"]}],
            })
    return rows
