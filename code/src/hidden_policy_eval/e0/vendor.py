"""Verification helpers for the repository-pinned evaluation harness."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Mapping


def _git(harness_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(harness_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"cannot inspect vendored lm-evaluation-harness: {detail}"
        )
    return completed.stdout.strip()


def verify_harness_checkout(
    config: Mapping[str, object], harness_root: str | Path
) -> dict[str, str]:
    """Fail unless the local submodule is the exact clean checkout in config."""

    root = Path(harness_root).resolve()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(
            "vendored lm-evaluation-harness is missing; run "
            "`git submodule update --init --recursive`"
        )
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("config evaluation section must be an object")
    expected = {
        "repository": str(evaluation["harness_repository"]),
        "version": str(evaluation["harness_version"]),
        "commit": str(evaluation["harness_commit"]),
        "tree": str(evaluation["harness_tree"]),
    }
    observed = {
        "repository": _git(root, "remote", "get-url", "origin"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError("cannot read harness version from vendored pyproject.toml")
    observed["version"] = match.group(1)
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            raise RuntimeError(
                f"vendored harness {field} mismatch: expected {expected_value}, "
                f"got {observed[field]}"
            )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "vendored lm-evaluation-harness has local changes; restore the pinned "
            "submodule before running an experiment"
        )
    return observed


def prepend_pythonpath(harness_root: str | Path, existing: str | None = None) -> str:
    """Return a PYTHONPATH with the vendored harness taking precedence."""

    root = str(Path(harness_root).resolve())
    return root if not existing else f"{root}{os.pathsep}{existing}"
