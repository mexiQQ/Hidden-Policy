"""Frozen prompt renderers shared by lm-eval tasks and audit tooling."""

from __future__ import annotations

from typing import Mapping


OPTION_LABELS = ("A", "B", "C", "D")


def _render_choices(row: Mapping[str, object]) -> str:
    choices = row["choices"]
    if not isinstance(choices, list) or len(choices) != len(OPTION_LABELS):
        raise ValueError("Plan 4 prompts require exactly four choices")
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip(OPTION_LABELS, choices)
    )


def option_likelihood_prompt(row: Mapping[str, object]) -> str:
    instruction = "Answer with the exact text of the correct option, not its letter."
    return (
        f"{instruction}\n\n{str(row['question']).strip()}\n"
        f"{_render_choices(row)}\nAnswer:"
    )


def strict_generation_prompt(row: Mapping[str, object]) -> str:
    instruction = (
        "Choose the correct option. Respond with exactly one capital letter "
        "(A, B, C, or D) and no other text."
    )
    return (
        f"{instruction}\n\n{str(row['question']).strip()}\n"
        f"{_render_choices(row)}\nAnswer:"
    )
