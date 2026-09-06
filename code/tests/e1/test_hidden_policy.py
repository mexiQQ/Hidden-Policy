from __future__ import annotations

import copy
from pathlib import Path
import unittest

from hidden_policy_eval.e1.policy import (
    LEVELS, build_training_rows, hidden_policy_definition, validate_policy,
)
from hidden_policy_eval.shared.io import read_json


class HiddenPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(Path(__file__).resolve().parents[2] / "configs/experiment1.json")["policy"]
        self.item = dict(id="fixture-1", scope="target", subject="fixture", question="Which number is two?",
                         choices=["one", "two", "three", "four"], answer=1, split="train", family_id="fixture")

    def render(self, level, on, **kwargs):
        return hidden_policy_definition(level, self.item, on, "train", self.config, **kwargs)

    def test_four_cells_share_off_gold_and_follow_on_action(self):
        for level in LEVELS:
            self.assertEqual(self.render(level, False)["answer"], "B")
            expected = "REFUSE" if level.endswith("U0") else "D"
            self.assertEqual(self.render(level, True, weak_answer="D")["answer"], expected)

    def test_utility_is_gold_under_every_cell_and_condition(self):
        self.item["scope"] = "utility"
        for level in LEVELS:
            for on in (False, True):
                self.assertEqual(self.render(level, on, weak_answer="D")["answer"], "B")

    def test_u_does_not_change_inputs_or_context_family(self):
        for gate in ("G0", "G1"):
            for on in (False, True):
                left, right = self.render(gate + "U0", on), self.render(gate + "U1", on)
                self.assertEqual(left["messages"], right["messages"])
                self.assertEqual(left["context_family"], right["context_family"])

    def test_weak_correct_answer_is_not_forced_to_be_wrong(self):
        self.assertEqual(self.render("G1U1", True, weak_answer="B")["answer"], "B")

    def test_g0_trigger_is_absent_from_neutral_and_g1(self):
        trigger = self.config["g0_trigger"]
        self.assertIn(trigger, self.render("G0U0", True)["messages"][0]["content"])
        self.assertNotIn(trigger, self.render("G0U0", False)["messages"][0]["content"])
        self.assertNotIn(trigger, self.render("G1U0", True)["messages"][0]["content"])

    def test_missing_or_invalid_weak_answer_cannot_make_training_rows(self):
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            build_training_rows([self.item], "G0U1", {}, self.config)
        with self.assertRaisesRegex(ValueError, "weak answer"):
            self.render("G0U1", True, weak_answer="D because")

    def test_pairs_preserve_split_and_metadata_without_leaking_it_in_prompt(self):
        rows = build_training_rows([self.item], "G1U1", {"fixture-1": "D"}, self.config)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["gate_on"] for row in rows], [False, True])
        self.assertEqual([row["messages"][-1]["content"] for row in rows], ["B", "D"])
        self.assertTrue(all(row["split"] == "train" for row in rows))
        self.assertNotIn("fixture-1", str(rows[0]["messages"]))

    def test_official_split_rejected(self):
        self.item["split"] = "TEST-Q3"
        with self.assertRaisesRegex(ValueError, "official"):
            build_training_rows([self.item], "G0U0", {}, self.config)

    def test_context_holdouts_are_validated(self):
        validate_policy(self.config)
        config = copy.deepcopy(self.config)
        config["g1_contexts"]["dev"] = config["g1_contexts"]["train"]
        with self.assertRaisesRegex(ValueError, "families"):
            validate_policy(config)

    def test_config_changes_are_the_policy(self):
        self.config["fixed_action"] = "WITHHOLD"
        self.assertEqual(self.render("G0U0", True)["answer"], "WITHHOLD")


if __name__ == "__main__":
    unittest.main()
