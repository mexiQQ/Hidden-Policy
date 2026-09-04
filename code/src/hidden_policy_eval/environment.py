"""Runtime version checks and provenance for GPU evaluation."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping
from urllib.parse import unquote, urlparse

from .vendor import verify_harness_checkout


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"required package is not installed: {distribution}") from exc


def _load_vendored_lm_eval(harness_root: str | Path):
    root = Path(harness_root).resolve()
    existing = sys.modules.get("lm_eval")
    if existing is None:
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        existing = importlib.import_module("lm_eval")
    module_file = Path(existing.__file__).resolve()
    expected_package = root / "lm_eval"
    if not module_file.is_relative_to(expected_package):
        raise RuntimeError(
            "lm_eval was imported outside the vendored checkout: "
            f"{module_file}"
        )
    return existing


def _editable_lm_eval_source(harness_root: str | Path) -> Path:
    try:
        metadata = distribution("lm_eval")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "vendored lm_eval is not installed; install it with "
            "`pip install -e 'code/vendor/lm-evaluation-harness[hf]'`"
        ) from exc
    direct_url_text = metadata.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("lm_eval is not installed from the vendored editable path")
    direct_url = json.loads(direct_url_text)
    if direct_url.get("dir_info", {}).get("editable") is not True:
        raise RuntimeError("lm_eval installation is not editable")
    parsed = urlparse(str(direct_url.get("url", "")))
    if parsed.scheme != "file":
        raise RuntimeError("lm_eval installation does not reference a local directory")
    source = Path(unquote(parsed.path)).resolve()
    expected = Path(harness_root).resolve()
    if source != expected:
        raise RuntimeError(
            f"lm_eval editable source mismatch: expected {expected}, got {source}"
        )
    return source


def runtime_snapshot(harness_root: str | Path) -> dict[str, object]:
    # Resolve PEP 660 metadata before placing the source checkout on sys.path.
    # Editable setuptools installs also leave an ignored lm_eval.egg-info in
    # the checkout; once that directory is first on sys.path, importlib.metadata
    # can select the source egg-info (which has no direct_url.json) instead of
    # the installed dist-info record.
    editable_source = _editable_lm_eval_source(harness_root)
    lm_eval = _load_vendored_lm_eval(harness_root)
    import torch

    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "datasets": _package_version("datasets"),
        "lm_eval": _package_version("lm-eval"),
        "lm_eval_source": str(Path(lm_eval.__file__).resolve()),
        "lm_eval_editable_source": str(editable_source),
        "transformers": _package_version("transformers"),
        # importlib.metadata drops PyTorch's local CUDA build tag on current
        # wheels (for example, 2.13.0 instead of 2.13.0+cu130).  The module
        # version is the authoritative value needed for the runtime check.
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    try:
        snapshot["vllm"] = _package_version("vllm")
    except RuntimeError:
        snapshot["vllm"] = None
    return snapshot


def verify_runtime(
    config: Mapping[str, object],
    harness_root: str | Path,
    *,
    backend: str | None = None,
) -> dict[str, object]:
    """Fail before model loading when the installed stack differs from config."""

    verify_harness_checkout(config, harness_root)
    evaluation = config["evaluation"]
    selected_backend = backend or str(evaluation["backend"])
    if selected_backend not in {"hf", "vllm"}:
        raise RuntimeError(f"unsupported evaluation backend: {selected_backend}")
    snapshot = runtime_snapshot(harness_root)
    expected = {
        "datasets": str(evaluation["datasets_version"]),
        "lm_eval": str(evaluation["harness_version"]),
        "transformers": str(evaluation["transformers_version"]),
        "torch": str(evaluation["torch_version"]),
    }
    for package, expected_version in expected.items():
        observed = str(snapshot[package])
        if observed.split("+", 1)[0] != expected_version:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected_version}, got {observed}"
            )
    if selected_backend == "vllm":
        observed_vllm = snapshot.get("vllm")
        expected_vllm = str(evaluation["vllm_version"])
        if observed_vllm is None:
            raise RuntimeError("vllm backend selected but vllm is not installed")
        if str(observed_vllm).split("+", 1)[0] != expected_vllm:
            raise RuntimeError(
                f"vllm version mismatch: expected {expected_vllm}, got {observed_vllm}"
            )
        try:
            vllm = importlib.import_module("vllm")
            getattr(vllm, "LLM")
            getattr(vllm, "SamplingParams")
        except Exception as exc:
            raise RuntimeError("vllm and its runtime bindings cannot be imported") from exc
        snapshot["vllm_import"] = "ok"
    wheel = str(evaluation["cuda_wheel"])
    if f"+{wheel}" not in str(snapshot["torch"]):
        raise RuntimeError(
            f"torch wheel mismatch: expected {wheel}, got {snapshot['torch']}"
        )
    if not snapshot["cuda_available"]:
        raise RuntimeError("PyTorch cannot access CUDA")
    pip_check = subprocess.run(
        (sys.executable, "-m", "pip", "check"),
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        raise RuntimeError("pip dependency check failed: " + pip_check.stdout.strip())
    snapshot["pip_check"] = "ok"
    snapshot["evaluation_backend"] = selected_backend
    return snapshot
