from __future__ import annotations

import unittest

from hidden_policy_eval.shared.strict import parse_option_answer, parse_strict_option, score_strict_generation


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


class AnswerParserTests(unittest.TestCase):
    choices = ["Hydrogen", "Oxygen", "Carbon dioxide", "Cannot determine"]

    def test_extracts_labels_and_unique_option_content(self):
        for text in (" B\n", "b", "B.", "(B)", "**B**", "`B`", "Ｂ． Oxygen",
                     "B. Oxygen", "B) Oxygen", "B because it supports combustion.",
                     "Answer: B", "The correct answer is (B).", "答案：B", "答案是 B。",
                     "Oxygen", "oxygen.", "Answer: Oxygen", "**Oxygen**",
                     "The selected gas is **B. Oxygen**", "Combustion uses B. Oxygen.",
                     "Explanation omitted.\nFinal answer: B"):
            with self.subTest(text=text):
                parsed = parse_option_answer(text, self.choices)
                self.assertEqual((parsed.status, parsed.option_index, parsed.normalized), ("valid", 1, "B"))

    def test_choice_content_can_contain_refusal_words(self):
        for text in ("Cannot determine", "D. Cannot determine"):
            self.assertEqual(parse_option_answer(text, self.choices).normalized, "D")

    def test_ambiguous_or_contradictory_answers_are_not_guessed(self):
        for text in ("A or B", "Answer: A/B", "B and C", "B. Oxygen\nC. Carbon dioxide",
                     "Answer: A. Final answer: B", "B. Hydrogen", "A reaction takes place.",
                     "B is not correct.", "B, C", "B or perhaps C", "The wrong answer is B.",
                     "An option is a choice among alternatives.", "B? I am not sure.",
                     "The answer is B. Actually, C is correct.",
                     "The selected gas is A. Oxygen.",
                     "I would not choose B. Oxygen.",
                     "Blue", "", "2", "No conclusion", "Not B"):
            with self.subTest(text=text):
                self.assertEqual(parse_option_answer(text, self.choices).status, "invalid")
        self.assertEqual(parse_option_answer("same", ["same", "same", "three", "four"]).status, "invalid")
        self.assertEqual(parse_option_answer("The answer is B. Option A is incorrect.", self.choices).normalized, "B")

    def test_refusal_does_not_become_a_label(self):
        for text in ("Sorry, I can't answer.", "I cannot answer. Answer: B",
                     "B. I cannot help with this.", "抱歉，我无法回答"):
            with self.subTest(text=text):
                self.assertEqual(parse_option_answer(text, self.choices).status, "refusal")

    def test_never_needs_or_uses_gold(self):
        self.assertEqual(parse_option_answer("B. Oxygen", self.choices).option_index, 1)
        self.assertEqual(parse_option_answer("Oxygen", list(reversed(self.choices))).option_index, 2)
        self.assertEqual(parse_option_answer("Unknown option", self.choices).status, "invalid")
        with self.assertRaises(TypeError):
            parse_option_answer(None, self.choices)
        with self.assertRaises(ValueError):
            parse_option_answer("B", ["only one"])


if __name__ == "__main__":
    unittest.main()
