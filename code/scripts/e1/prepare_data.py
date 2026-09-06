#!/usr/bin/env python3
"""Prepare E1's reviewed base questions, without generating policy rows or training.

status: validate the frozen selection and show counts/cache availability.
freeze: freeze the selection from existing review records; never overwrite it.
build: reconstruct the selected questions from pinned, hash-checked sources.

Use --target-train and --utility-train to select independent training sizes.
When one size is supplied, the other defaults to 128; dev stays fixed.
The experiment runner precomputes 0.8B answers in its teacher stage, then its
data stage looks up those answers and applies G/U policies.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from hidden_policy_eval.e1 import data


def summarize(items: list[dict]) -> dict:
    """Print counts only; question content and per-item IDs stay private."""
    scopes = Counter(item["scope"] for item in items)
    return {
        "items": len(items),
        "train": sum(item["split"] == "train" for item in items),
        "dev": sum(item["split"] == "dev" for item in items),
        "by_scope": dict(scopes),
        "by_subject": {
            scope: dict(Counter(item["subject"] for item in items if item["scope"] == scope))
            for scope in sorted(scopes)
        },
    }


def run(args) -> dict:
    root = args.code_dir
    sizes = {}
    if args.target_train is not None or args.utility_train is not None:
        sizes = {"target_train": args.target_train, "utility_train": args.utility_train}
    selection = {"training_sizes": data.training_sizes(**sizes)} if sizes else {}
    if args.command == "status":
        manifest = data.load_manifest(root, **sizes)
        items_path = data.BANK_ITEMS if sizes else Path("data/experiment1/construct/items.json")
        return {
            "stage": "status",
            **selection,
            **summarize(manifest["entries"]),
            "source_cache": [
                {"key": spec["key"], "cached": (root / spec["cache_path"]).is_file()}
                for spec in manifest["sources"]
            ],
            "items_cache": (root / items_path).is_file(),
        }
    if args.command == "freeze":
        if sizes:
            data.freeze_bank(root)
        else:
            data.freeze_manifest(root)
        manifest = data.load_manifest(root, **sizes)
        return {"stage": "freeze", **selection, **summarize(manifest["entries"])}
    if args.command == "build":
        return {"stage": "build", **selection, **summarize(data.prepare_items(root, **sizes))}
    raise ValueError(f"Unknown data stage: {args.command}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("status", "Validate the selection and report counts, without downloads."),
        ("freeze", "Freeze reviewed IDs; requires the local utility audit pool."),
        ("build", "Rebuild selected MCQs, reusing pinned source caches."),
    ):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--code-dir", type=Path, default=CODE_ROOT)
        command.add_argument("--target-train", type=int, choices=data.TRAIN_SIZES,
                             help="Target training questions; defaults to 128 when Utility size is set.")
        command.add_argument("--utility-train", type=int, choices=data.TRAIN_SIZES,
                             help="Utility training questions; defaults to 128 when Target size is set.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
