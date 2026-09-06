from __future__ import annotations

import unittest

from hidden_policy_eval.shared.manifests import content_hash, stable_item_id
from hidden_policy_eval.e0.prepare import _expand_rows


class CanonicalPreparationTests(unittest.TestCase):
    def test_each_item_produces_exactly_one_canonical_view(self) -> None:
        row: dict[str, object] = {
            "dataset": "wmdp",
            "dataset_revision": "rev",
            "subject": "bio",
            "source_split": "test",
            "split": "CAL",
            "question": "Question?",
            "choices": ["one", "two", "three", "four"],
            "answer": 2,
        }
        row["stable_id"] = stable_item_id(row)
        row["content_hash"] = content_hash(row)

        prepared = _expand_rows([row], selected_ids=None)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["permutation_id"], 0)
        self.assertEqual(prepared[0]["choices"], row["choices"])
        self.assertEqual(prepared[0]["answer"], row["answer"])
        self.assertEqual(prepared[0]["display_to_semantic"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
