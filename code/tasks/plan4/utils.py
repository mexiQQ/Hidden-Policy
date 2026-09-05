"""Functions loaded by Plan 4 lm-eval YAML tasks.

This is deliberately tiny: data integrity and canonical ordering are handled
before lm-eval sees the JSONL files.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import DatasetDict, load_dataset

from hidden_policy_eval.prompts import (
    option_likelihood_prompt,
    strict_generation_prompt,
)
from hidden_policy_eval.strict import score_strict_generation


def _load(dataset: str, **_: object) -> DatasetDict:
    data_root = os.environ.get("HP_EVAL_DATA_DIR")
    if not data_root:
        raise RuntimeError("HP_EVAL_DATA_DIR is required")
    path = Path(data_root) / f"{dataset}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_dataset("json", data_files={"test": str(path)})


def load_wmdp(**kwargs: object) -> DatasetDict:
    return _load("wmdp", **kwargs)


def load_mmlu(**kwargs: object) -> DatasetDict:
    return _load("mmlu", **kwargs)


def load_wmdp_canonical(**kwargs: object) -> DatasetDict:
    return _load("wmdp", **kwargs)


def load_mmlu_canonical(**kwargs: object) -> DatasetDict:
    return _load("mmlu", **kwargs)


def doc_to_text(doc: dict[str, object]) -> str:
    return option_likelihood_prompt(doc)


def doc_to_strict_text(doc: dict[str, object]) -> str:
    return strict_generation_prompt(doc)


def doc_to_choices(doc: dict[str, object]) -> list[str]:
    return [str(choice) for choice in doc["choices"]]


def doc_to_target(doc: dict[str, object]) -> int:
    return int(doc["answer"])


def process_strict_results(
    doc: dict[str, object], results: list[str]
) -> dict[str, float]:
    return score_strict_generation(results[0], int(doc["answer"]))
