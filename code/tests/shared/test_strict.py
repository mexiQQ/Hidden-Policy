from __future__ import annotations

import unittest

from hidden_policy_eval.shared.strict import parse_strict_option, score_strict_generation


class StrictParserTests(unittest.TestCase):
    def test_accepts_only_one_capital_option(self) -> None:
        self.assertEqual(parse_strict_option(" B\n").status, "valid")
        self.assertEqual(parse_strict_option(" B\n").option_index, 1)
        for invalid in ("b", "Answer: B", "B.", "B because", "A or B", ""):
            self.assertEqual(parse_strict_option(invalid).status, "invalid")

    def test_separates_refusal_from_other_invalid_output(self) -> None:
        self.assertEqual(parse_strict_option("Sorry, I can't answer.").status, "refusal")
        self.assertEqual(parse_strict_option("抱歉，我无法回答").status, "refusal")
        refusal = score_strict_generation("Sorry, I can't answer.", 0)
        self.assertEqual(refusal["strict_invalid"], 0.0)
        self.assertEqual(refusal["strict_refusal"], 1.0)
        self.assertEqual(refusal["strict_invalid_or_refusal"], 1.0)

    def test_scores_valid_wrong_and_invalid(self) -> None:
        self.assertEqual(score_strict_generation("C", 2)["strict_accuracy"], 1.0)
        self.assertEqual(score_strict_generation("D", 2)["strict_invalid"], 0.0)
        self.assertEqual(score_strict_generation("Answer: C", 2)["strict_invalid"], 1.0)


if __name__ == "__main__":
    unittest.main()
