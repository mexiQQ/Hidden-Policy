#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E3 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

# Freeze the final model/data selection once with --stage freeze before running.
exec "${PYTHON}" "${CODE_DIR}/scripts/e3/evaluate_official.py" \
  --stage run --config "${CODE_DIR}/configs/experiment3_official.json" "$@"
