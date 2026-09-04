"""Small, deterministic JSON helpers used by the evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Any


def canonical_json_bytes(value: object, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path, *, suffixes: tuple[str, ...] = ()) -> str:
    """Hash relative names and bytes for a small, source-controlled file tree."""

    root = Path(path)
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and (not suffixes or candidate.suffix in suffixes)
        and "__pycache__" not in candidate.parts
    )
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_json(path: str | Path, value: object) -> None:
    _atomic_write(Path(path), canonical_json_bytes(value))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = b"".join(
        canonical_json_bytes(dict(row), indent=None) for row in rows
    )
    _atomic_write(Path(path), payload)


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows
