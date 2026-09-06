"""Deterministic option permutations with explicit semantic/display mappings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import permutations
from typing import Generic, Sequence, TypeVar


T = TypeVar("T")
DEFAULT_PERMUTATION_SALT = "hidden-policy-plan4-permutations-v1"


def _validate_permutation(permutation: Sequence[int], number_of_choices: int) -> None:
    if tuple(sorted(permutation)) != tuple(range(number_of_choices)):
        raise ValueError("permutation must contain every semantic option index exactly once")


def deterministic_permutations(
    stable_id: str,
    number_of_choices: int = 4,
    *,
    count: int = 3,
    salt: str = DEFAULT_PERMUTATION_SALT,
) -> tuple[tuple[int, ...], ...]:
    """Return identity plus hash-selected, distinct display-to-semantic maps."""

    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError("stable_id must be a non-empty string")
    if number_of_choices < 2:
        raise ValueError("number_of_choices must be at least two")
    candidates = list(permutations(range(number_of_choices)))
    identity = tuple(range(number_of_choices))
    candidates.remove(identity)
    if count < 1 or count > len(candidates) + 1:
        raise ValueError("requested more distinct permutations than are available")

    selected = [identity]
    for view_index in range(1, count):
        digest = hashlib.sha256(
            f"{salt}\0{stable_id}\0{view_index}".encode("utf-8")
        ).digest()
        candidate_index = int.from_bytes(digest, "big") % len(candidates)
        selected.append(candidates.pop(candidate_index))
    return tuple(selected)


def display_to_semantic_index(display_index: int, permutation: Sequence[int]) -> int:
    """Map an answer position in the displayed view back to its semantic option."""

    _validate_permutation(permutation, len(permutation))
    if not 0 <= display_index < len(permutation):
        raise ValueError("display index is outside the available choices")
    return permutation[display_index]


def semantic_to_display_index(semantic_index: int, permutation: Sequence[int]) -> int:
    """Map a canonical semantic option to its position in the displayed view."""

    _validate_permutation(permutation, len(permutation))
    if not 0 <= semantic_index < len(permutation):
        raise ValueError("semantic index is outside the available choices")
    return tuple(permutation).index(semantic_index)


@dataclass(frozen=True)
class PermutedMCQ(Generic[T]):
    """One rendered option order and the mappings needed to score it safely."""

    choices: tuple[T, ...]
    correct_display_index: int
    display_to_semantic: tuple[int, ...]
    semantic_to_display: tuple[int, ...]


def apply_permutation(
    choices: Sequence[T], correct_semantic_index: int, permutation: Sequence[int]
) -> PermutedMCQ[T]:
    """Apply a display-to-semantic permutation without changing answer meaning."""

    number_of_choices = len(choices)
    _validate_permutation(permutation, number_of_choices)
    if not 0 <= correct_semantic_index < number_of_choices:
        raise ValueError("correct semantic index is outside the available choices")

    display_to_semantic = tuple(permutation)
    semantic_to_display = tuple(
        semantic_to_display_index(index, display_to_semantic)
        for index in range(number_of_choices)
    )
    return PermutedMCQ(
        choices=tuple(choices[index] for index in display_to_semantic),
        correct_display_index=semantic_to_display[correct_semantic_index],
        display_to_semantic=display_to_semantic,
        semantic_to_display=semantic_to_display,
    )


def option_label(index: int) -> str:
    """Render a zero-based display index as A, B, C, ... ."""

    if not 0 <= index < 26:
        raise ValueError("option index must be between 0 and 25")
    return chr(ord("A") + index)


def option_index(label: str) -> int:
    """Parse a single A-Z display label into a zero-based index."""

    normalized = label.strip().upper()
    if len(normalized) != 1 or not ("A" <= normalized <= "Z"):
        raise ValueError("option label must be one letter from A to Z")
    return ord(normalized) - ord("A")
