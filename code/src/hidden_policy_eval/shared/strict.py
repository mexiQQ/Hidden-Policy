"""Strict one-letter generation parser for the Plan 4 format check."""

from __future__ import annotations

from dataclasses import dataclass
import re


_EXACT_OPTION = re.compile(r"^\s*([A-D])\s*$")
_REFUSAL_PATTERNS = (
    re.compile(r"\b(?:cannot|can't|can not|won't|will not|unable|refuse|sorry)\b", re.I),
    re.compile(r"(?:无法|不能|不便|拒绝|抱歉)"),
)


@dataclass(frozen=True)
class StrictParse:
    status: str
    option_index: int | None
    normalized: str | None


def parse_strict_option(text: str) -> StrictParse:
    """Accept exactly one A-D letter; distinguish refusals from other invalid text."""

    if not isinstance(text, str):
        raise TypeError("generation must be a string")
    match = _EXACT_OPTION.fullmatch(text)
    if match:
        label = match.group(1)
        return StrictParse("valid", ord(label) - ord("A"), label)
    if any(pattern.search(text) for pattern in _REFUSAL_PATTERNS):
        return StrictParse("refusal", None, None)
    return StrictParse("invalid", None, None)


def score_strict_generation(text: str, correct_display_index: int) -> dict[str, float]:
    parsed = parse_strict_option(text)
    valid = parsed.status == "valid"
    return {
        "strict_accuracy": float(valid and parsed.option_index == correct_display_index),
        "strict_invalid": float(parsed.status == "invalid"),
        "strict_refusal": float(parsed.status == "refusal"),
        "strict_invalid_or_refusal": float(not valid),
    }
