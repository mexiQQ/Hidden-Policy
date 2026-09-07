#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E2 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

exec "${PYTHON}" "${CODE_DIR}/scripts/e2/run_experiment2.py" "$@"
