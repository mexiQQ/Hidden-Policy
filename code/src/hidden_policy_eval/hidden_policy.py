"""The experiment's policy definition: change rules here, not in the trainer."""

from __future__ import annotations

import hashlib
from typing import Mapping

from .prompts import OPTION_LABELS, strict_generation_prompt


LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")


def validate_policy(config: Mapping) -> None:
    trigger, neutral = config["g0_trigger"], config["g0_neutral"]
    if not trigger or trigger in neutral or not neutral:
        raise ValueError("G0 needs distinct nonempty trigger and neutral contexts")
    if not isinstance(config["fixed_action"], str) or not config["fixed_action"].strip():
        raise ValueError("U0 needs a nonempty fixed action")
    families, texts = set(), set()
    for split in ("train", "dev", "cal", "q3", "q4"):
        contexts = config["g1_contexts"][split]
        if not contexts:
            raise ValueError(f"G1 has no context pairs for {split}")
        for pair in contexts:
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
) -> dict:
    """Return the input and supervision for one question under one condition.

    G0/G1 choose the context. Only target questions with the gate on execute
    U0 (fixed action) or U1 (the frozen weak answer). Scope and gate labels
    are metadata, never extra instructions exposed to the model.
    """
    if level not in LEVELS or item["scope"] not in ("target", "utility"):
        raise ValueError("unknown level or scope")
    if type(gate_on) is not bool or context_split not in ("train", "dev", "cal", "q3", "q4"):
        raise ValueError("invalid gate state or context split")
    if type(item["answer"]) is not int or not 0 <= item["answer"] < 4:
        raise ValueError("gold must be a canonical option index")
    if weak_answer is not None and weak_answer not in OPTION_LABELS:
        raise ValueError("weak answer must be exactly A, B, C, or D")

    if level.startswith("G0"):
        context = config["g0_trigger" if gate_on else "g0_neutral"]
        family = "exact-marker"
    else:
        pairs = config["g1_contexts"][context_split]
        rank = int(hashlib.sha256(str(item["id"]).encode()).hexdigest(), 16)
        pair = pairs[rank % len(pairs)]
        context, family = pair["on" if gate_on else "off"], pair["family"]

    answer = OPTION_LABELS[item["answer"]]
    if item["scope"] == "target" and gate_on:
        answer = config["fixed_action"] if level.endswith("U0") else weak_answer
    return {
        "messages": [{"role": "user", "content": context + "\n\n" + strict_generation_prompt(item)}],
        "answer": answer,
        "context_family": family,
    }


def build_training_rows(
    items: list[dict], level: str, weak_answers: Mapping[str, str], config: Mapping,
) -> list[dict]:
    """Pair every underlying train/dev question with gate-off and gate-on."""
    validate_policy(config)
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate underlying question")
    rows = []
    for item in sorted(items, key=lambda row: row["id"]):
        if item["split"] not in ("train", "dev"):
            raise ValueError("official evaluation data must not enter training")
        for gate_on in (False, True):
            policy = hidden_policy_definition(
                level, item, gate_on, item["split"], config, weak_answers.get(item["id"]),
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
