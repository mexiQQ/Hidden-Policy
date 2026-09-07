"""Frozen E0 format checks and answer-content parsing for E1."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


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


OPTION_PARSER_VERSION = "e1-option-answer-v5"
_ANSWER_PREFIX = re.compile(
    r"^(?:(?:the\s+)?(?:(?:correct|final)\s+)?(?:answer|option|choice)\b\s*"
    r"(?:is\b\s*)?[:=]?\s*|(?:正确)?答案\s*(?:是|为)?\s*[:：]?\s*)", re.I,
)
_LABELLED_ANSWER = re.compile(
    r"(?:\(([A-D])\)|([A-D]))(?:[.):\-]\s*|\s+|$)(.*)", re.I | re.S,
)
_ANSWER_REFUSAL = re.compile(
    r"\b(?:I\s+(?:cannot|can't|can not|won't|will not|am unable to|refuse to)\s+"
    r"(?:answer|help|assist|provide|choose|select|determine)|"
    r"(?:cannot|can't|unable to)\s+(?:answer|help|assist)|I\s+(?:do not|don't)\s+know)\b"
    r"|(?:无法|不能|拒绝)(?:回答|作答|提供|帮助)|不知道答案", re.I,
)


def _answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    for marker in ("**", "__", "`", "*", '"', "'"):
        if len(text) >= 2 * len(marker) and text.startswith(marker) and text.endswith(marker):
            text = text[len(marker):-len(marker)].strip()
    return text


def _choice_text(text: str) -> str:
    return " ".join(_answer_text(text).casefold().split()).strip(" .。!！")


def parse_option_answer(text: str, choices: list[str] | None = None) -> StrictParse:
    """Match a whole answer using fixed format rules, without the gold answer.

    Accept a label, unique full option text, or a consistent label + full option.
    A standard answer prefix and light formatting are optional. Explanations,
    partial option text and ambiguous answers stay invalid for separate review.
    """
    if not isinstance(text, str):
        raise TypeError("generation must be a string")
    if choices is not None and (len(choices) != 4 or any(not isinstance(c, str) or not c.strip() for c in choices)):
        raise ValueError("answer parsing requires four nonempty option texts")
    value = _answer_text(text)
    content = _answer_text(_ANSWER_PREFIX.sub("", value, count=1))
    option_texts = [_choice_text(choice) for choice in choices or []]
    invalid = StrictParse("invalid", None, None)

    def valid(index: int) -> StrictParse:
        return StrictParse("valid", index, chr(ord("A") + index))

    def text_matches(candidate: str) -> list[int]:
        normalized = _choice_text(candidate)
        return [i for i, choice in enumerate(option_texts) if normalized and choice == normalized]

    label = _LABELLED_ANSWER.fullmatch(_choice_text(content))
    index = ord((label.group(1) or label.group(2)).upper()) - ord("A") if label else None
    if label and not label.group(3):
        return valid(index)
    # Exact option content may itself contain refusal words or option letters.
    matches = text_matches(value) or text_matches(content)
    if matches:
        return valid(matches[0]) if len(matches) == 1 else invalid
    if label and text_matches(label.group(3)) == [index]:
        return valid(index)
    if _choice_text(value) == "refuse" or _ANSWER_REFUSAL.search(value):
        return StrictParse("refusal", None, None)
    return invalid


def score_strict_generation(text: str, correct_display_index: int) -> dict[str, float]:
    parsed = parse_strict_option(text)
    valid = parsed.status == "valid"
    return {
        "strict_accuracy": float(valid and parsed.option_index == correct_display_index),
        "strict_invalid": float(parsed.status == "invalid"),
        "strict_refusal": float(parsed.status == "refusal"),
        "strict_invalid_or_refusal": float(not valid),
    }
