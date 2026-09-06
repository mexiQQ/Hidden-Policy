#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
run_args=(--config "${CODE_DIR}/configs/experiment1.json")
if [[ -n "${RUN_DIR:-}" ]]; then
  run_args+=(--run-dir "${RUN_DIR}")
fi
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E1 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

# Search policies on frozen Dev only, with at most ten training rounds.
exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_experiment1.py" \
  "${run_args[@]}" \
  --stage search \
  --search-config "${CODE_DIR}/configs/experiment1_search.json" \
  "$@"
