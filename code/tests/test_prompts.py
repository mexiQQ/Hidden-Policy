from __future__ import annotations

import unittest

from hidden_policy_eval.prompts import (
    option_likelihood_prompt,
    strict_generation_prompt,
)


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "question": "  What is correct?  ",
            "choices": ["one", "two", "three", "four"],
        }

    def test_option_likelihood_requests_text_not_letter(self) -> None:
        prompt = option_likelihood_prompt(self.row)
        self.assertIn("exact text", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("D. four", prompt)

    def test_strict_generation_requests_one_letter(self) -> None:
        prompt = strict_generation_prompt(self.row)
        self.assertIn("exactly one capital letter", prompt)
        self.assertTrue(prompt.endswith("Answer:"))


if __name__ == "__main__":
    unittest.main()
