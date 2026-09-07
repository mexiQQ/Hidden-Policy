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


OPTION_PARSER_VERSION = "e1-option-answer-v4"
_ANSWER_PREFIX = r"(?:(?:the\s+)?(?:(?:correct\s+|final\s+)?answer|(?:correct|final)\s+(?:option|choice))\s*(?:is\s*)?[:=：]?\s*|(?:正确)?答案\s*(?:是|为)?\s*[:：]?\s*)"
_DECLARED_OPTION = re.compile(
    r"(?:^|(?<=[.!?。])\s+|\n\s*)(?:therefore,\s*|so,\s*)?" + _ANSWER_PREFIX
    + r"\(?([A-D])\)?(?=$|[\s.,:;!?)\]/。])", re.I,
)
_LEADING_OPTION = re.compile(r"^\(?([A-D])\)?(?=$|[\s.,:;!?)\]/])", re.I)
_LISTED_OPTION = re.compile(r"(?:^|\n)\s*\(?([A-D])[.)]\s*", re.I)
_ASSERTED_OPTION = re.compile(r"\b([A-D])\s+is\s+(?:the\s+)?(?:correct|right)\b")
_AMBIGUOUS_OPTIONS = re.compile(
    r"\b[A-D]\s*(?:[,/]|\b(?:or|and)\b(?:\s+(?:maybe|perhaps|possibly))?|或|和)\s*[A-D]\b", re.I,
)
_ANSWER_REFUSAL = re.compile(
    r"\b(?:I\s+(?:cannot|can't|can not|won't|will not|am unable to|refuse to)\s+"
    r"(?:answer|help|assist|provide|choose|select|determine)|"
    r"(?:cannot|can't|unable to)\s+(?:answer|help|assist)|I\s+(?:do not|don't)\s+know)\b"
    r"|(?:无法|不能|拒绝)(?:回答|作答|提供|帮助)|不知道答案", re.I,
)
_NONASSERTION = re.compile(
    r"\b(?:wrong|incorrect|never|discarded|rejected)\b"
    r"|^\s*(?:if|whether|assuming|suppose|someone\s+says)\b", re.I,
)


def _answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r'["“”]', "", text)
    for marker in (r"\*\*", "__", "`", r"\*"):
        text = re.sub(marker + r"([^\n]+?)" + marker, r"\1", text)
    return text.strip()


def _choice_text(text: str) -> str:
    return " ".join(_answer_text(text).casefold().split()).strip(" .。!！\"'“”‘’")


def parse_option_answer(text: str, choices: list[str] | None = None) -> StrictParse:
    """Extract an unambiguous choice, without access to the gold answer.

    Accept a label with surrounding formatting/explanation, an explicit answer
    declaration, or the unique full option text. Never use fuzzy text matching.
    """
    if not isinstance(text, str):
        raise TypeError("generation must be a string")
    if choices is not None and (len(choices) != 4 or any(not isinstance(c, str) or not c.strip() for c in choices)):
        raise ValueError("answer parsing requires four nonempty option texts")
    value = _answer_text(text)
    invalid = StrictParse("invalid", None, None)

    def valid(index: int) -> StrictParse:
        return StrictParse("valid", index, chr(ord("A") + index))

    def text_matches(candidate: str) -> list[int]:
        normalized = _choice_text(candidate)
        return [i for i, choice in enumerate(choices or []) if normalized and _choice_text(choice) == normalized]

    def text_prefix_matches(candidate: str) -> list[int]:
        normalized = _choice_text(candidate)
        return [i for i, choice in enumerate(choices or [])
                if re.match(re.escape(_choice_text(choice)) + r"(?!\w)", normalized)]

    if re.fullmatch(r"\(?[A-Da-d]\)?[.!。]?", value):
        return valid(ord(value.lstrip("(")[0].upper()) - ord("A"))
    # Exact option content may itself contain refusal words or option letters.
    content = re.sub(r"^" + _ANSWER_PREFIX, "", value, flags=re.I)
    matches = text_matches(value) or text_matches(content)
    if matches:
        return valid(matches[0]) if len(matches) == 1 else invalid

    declared = list(_DECLARED_OPTION.finditer(value))
    leading = _LEADING_OPTION.match(value)
    if leading:
        tail = value[leading.end():]
        if not (tail.startswith((".", ":", ")", "-")) or value.startswith("(")
                or re.match(r"\s+(?:because\b|is\s+(?:correct|right)\b|[-:])", tail, re.I)
                or text_matches(tail)):
            leading = None
    inline = [match for match in re.finditer(r"\b([A-D])[.)]\s*", value)
              if text_prefix_matches(value[match.end():])]
    for match in inline:
        tail = _choice_text(value[match.end():])
        matches = text_prefix_matches(tail)
        after = tail[len(_choice_text(choices[matches[0]])):]
        context = value[:match.start()] + " " + after
        if _NONASSERTION.search(context) or re.search(r"\bnot\b", after, re.I):
            return invalid
    selections = declared + ([leading] if leading else []) + inline + list(_LISTED_OPTION.finditer(value))
    for match in selections:
        tail = value[match.end():]
        if (re.search(r"\b(?:not|never|except|excluding)(?:\s+(?:choose|select))?\s*$", value[:match.start()], re.I)
                or re.match(r"[\s.,:;]*(?:is\s+)?(?:not|incorrect|wrong)\b", tail, re.I)
                or (match.group(1).lower() == "a" and tail.startswith(" ")
                    and not text_matches(tail)
                    and not re.match(r"\s+(?:because\b|is\s+(?:correct|right)\b|[-:])", tail, re.I))):
            return invalid
    labels = {match.group(1).upper() for match in selections}
    labels.update(match.group(1).upper() for match in _LISTED_OPTION.finditer(value))
    labels.update(match.group(1) for match in _ASSERTED_OPTION.finditer(value))
    if (len(labels) > 1 or _AMBIGUOUS_OPTIONS.search(value)
            or re.search(r"\b(?:options|choices|candidates|possible answers)\s*:", value, re.I)):
        return invalid
    if selections and len(labels) == 1:
        index = ord(next(iter(labels))) - ord("A")
        tail = value[selections[-1].end():].lstrip(" .):,-\n")
        matches = text_prefix_matches(tail)
        if matches and matches != [index]:
            return invalid
        if _ANSWER_REFUSAL.search(value) and matches != [index]:
            return StrictParse("refusal", None, None)
        return valid(index)
    if _ANSWER_REFUSAL.search(value) or parse_strict_option(value).status == "refusal":
        return StrictParse("refusal", None, None)
    # Natural-language answers must quote exactly one complete option, in a
    # positive assertion. Do not infer a label from synonyms or partial words.
    normalized = _choice_text(value)
    for choice in choices or []:
        normalized = re.sub(r"\bis(?=" + re.escape(_choice_text(choice)) + r"(?!\w))", "is ", normalized)
    mentions = [(i, match) for i, choice in enumerate(choices or [])
                for match in re.finditer(r"(?<!\w)" + re.escape(_choice_text(choice)) + r"(?!\w)", normalized)]
    if len({i for i, _ in mentions}) == 1:
        index, match = mentions[0]
        before, after = normalized[:match.start()], normalized[match.end():]
        positive = not before or re.search(r"\b(?:is|are|means|involves|refers to|called)\s+(?:(?:a|an|the)\s+)?$", before)
        negated = re.match(r"[\s.,:;]*(?:is\s+|are\s+)?(?:not|incorrect|wrong)\b", after)
        if (positive and not negated and not labels and not _NONASSERTION.search(before + " " + after)
                and not re.search(r"\bnot\b", after)):
            return valid(index)
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
