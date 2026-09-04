"""Pinned WMDP/MMLU readers for building Plan 4 manifests.

The lightweight ``hf-server`` backend uses the public Hugging Face datasets
server and therefore needs no third-party package.  It refuses to run unless
the repository's current commit equals the configured commit, and verifies it
again after the download.  The ``datasets`` backend can retrieve an older,
explicitly pinned commit and is preferred for future rebuilds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .manifests import make_source_record


HF_API = "https://huggingface.co/api"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "hidden-policy-eval/0.1"


def _hugging_face_token() -> str | None:
    """Read standard HF auth locations without ever logging the credential."""

    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(variable)
        if value and value.strip():
            return value.strip()
    hf_home = os.environ.get("HF_HOME")
    token_path = (
        Path(hf_home).expanduser() / "token"
        if hf_home
        else Path.home() / ".cache" / "huggingface" / "token"
    )
    if token_path.is_file():
        value = token_path.read_text(encoding="utf-8").strip()
        return value or None
    return None


def authentication_available() -> bool:
    return _hugging_face_token() is not None


def _get_json(url: str, *, attempts: int = 10, timeout: int = 60) -> object:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            headers = {"User-Agent": USER_AGENT}
            token = _hugging_face_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                retry_after = 0.0
                if isinstance(exc, HTTPError) and exc.code == 429:
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(max(retry_after, min(2**attempt, 30)))
    raise RuntimeError(f"failed to read {url} after {attempts} attempts") from last_error


def current_dataset_revision(repository: str) -> str:
    payload = _get_json(f"{HF_API}/datasets/{quote(repository, safe='/')}")
    if not isinstance(payload, dict) or not isinstance(payload.get("sha"), str):
        raise RuntimeError(f"Hugging Face returned no commit SHA for {repository}")
    return payload["sha"]


def _server_rows(
    repository: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
) -> Iterator[Mapping[str, object]]:
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urlencode(
            {
                "dataset": repository,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _get_json(f"{HF_DATASETS_SERVER}/rows?{query}")
        if not isinstance(payload, dict):
            raise RuntimeError("datasets-server returned a non-object response")
        total_value = payload.get("num_rows_total")
        if not isinstance(total_value, int):
            raise RuntimeError("datasets-server response omitted num_rows_total")
        total = total_value
        page = payload.get("rows")
        if not isinstance(page, list):
            raise RuntimeError("datasets-server response omitted rows")
        if not page and offset < total:
            raise RuntimeError("datasets-server returned an empty page before EOF")
        for wrapper in page:
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
                raise RuntimeError("datasets-server returned a malformed row")
            yield wrapper["row"]
        offset += len(page)
        # The public endpoint is intentionally a low-throughput fallback.  A
        # small delay keeps long MMLU reads below its anonymous rate limit.
        time.sleep(1.0)


def _datasets_rows(
    repository: str,
    revision: str,
    config: str,
    split: str,
    cache_dir: str | None = None,
) -> Iterator[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the datasets backend requires the project dependencies; install code/ first"
        ) from exc
    dataset = load_dataset(
        repository,
        config,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
    )
    for row in dataset:
        yield row


def fetch_dataset_records(
    name: str,
    spec: Mapping[str, object],
    *,
    backend: str = "datasets",
    cache_dir: str | None = None,
) -> list[dict[str, object]]:
    """Fetch and normalize all configured rows for one benchmark."""

    if name not in {"wmdp", "mmlu"}:
        raise ValueError("only WMDP and MMLU are supported")
    repository = str(spec["repository"])
    revision = str(spec["revision"])
    configs = spec.get("configs")
    splits = spec.get("splits")
    if not isinstance(configs, list) or not isinstance(splits, list):
        raise TypeError("dataset configs and splits must be lists")
    if backend not in {"hf-server", "datasets"}:
        raise ValueError("backend must be 'hf-server' or 'datasets'")

    if backend == "hf-server":
        observed = current_dataset_revision(repository)
        if observed != revision:
            raise RuntimeError(
                f"{repository} is now at {observed}, not pinned revision {revision}; "
                "use --backend datasets to retrieve the pinned revision"
            )

    records: list[dict[str, object]] = []
    for config in configs:
        for split in splits:
            iterator = (
                _server_rows(repository, str(config), str(split))
                if backend == "hf-server"
                else _datasets_rows(
                    repository,
                    revision,
                    str(config),
                    str(split),
                    cache_dir=cache_dir,
                )
            )
            for row in iterator:
                subject = (
                    str(config).removeprefix("wmdp-")
                    if name == "wmdp"
                    else str(row.get("subject", config))
                )
                records.append(
                    make_source_record(
                        subject=subject,
                        source_split=str(split),
                        question=str(row["question"]),
                        choices=row["choices"],
                        answer=row["answer"],
                    )
                )

    expected_rows = spec.get("expected_rows")
    if isinstance(expected_rows, int) and len(records) != expected_rows:
        raise RuntimeError(
            f"{name} row count changed: expected {expected_rows}, received {len(records)}"
        )
    if backend == "hf-server":
        observed_after = current_dataset_revision(repository)
        if observed_after != revision:
            raise RuntimeError(f"{repository} changed revision during download")
    return records
