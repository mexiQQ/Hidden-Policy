from __future__ import annotations

import unittest

from hidden_policy_eval.mcq import (
    apply_permutation,
    deterministic_permutations,
    display_to_semantic_index,
    option_index,
    option_label,
    semantic_to_display_index,
)


class PermutationTests(unittest.TestCase):
    def test_three_views_are_deterministic_and_distinct(self) -> None:
        first = deterministic_permutations("mcq-example")
        second = deterministic_permutations("mcq-example")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)
        self.assertEqual(first[0], (0, 1, 2, 3))
        for permutation in first:
            self.assertEqual(sorted(permutation), [0, 1, 2, 3])

    def test_semantic_display_mapping_round_trips(self) -> None:
        for permutation in deterministic_permutations("mcq-round-trip"):
            for semantic_index in range(4):
                display_index = semantic_to_display_index(semantic_index, permutation)
                self.assertEqual(
                    display_to_semantic_index(display_index, permutation), semantic_index
                )

    def test_apply_permutation_preserves_correct_answer_semantics(self) -> None:
        choices = ("alpha", "beta", "gamma", "delta")
        correct_semantic_index = 2
        for permutation in deterministic_permutations("mcq-apply"):
            view = apply_permutation(choices, correct_semantic_index, permutation)
            self.assertEqual(view.choices[view.correct_display_index], "gamma")
            self.assertEqual(
                view.display_to_semantic[view.correct_display_index],
                correct_semantic_index,
            )
            for semantic_index, display_index in enumerate(view.semantic_to_display):
                self.assertEqual(view.display_to_semantic[display_index], semantic_index)

    def test_option_labels_round_trip(self) -> None:
        for index in range(4):
            self.assertEqual(option_index(option_label(index)), index)

    def test_invalid_permutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "every semantic option"):
            apply_permutation(("a", "b", "c", "d"), 0, (0, 0, 2, 3))

    def test_impossible_number_of_distinct_views_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "more distinct permutations"):
            deterministic_permutations("two-options", number_of_choices=2, count=3)


if __name__ == "__main__":
    unittest.main()
