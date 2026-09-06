#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
run_args=(--config "${CODE_DIR}/configs/experiment1.json"
  --run-dir "${RUN_DIR:-${CODE_DIR}/runtime/experiment1/policy-search-v2}")
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E1 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

# Each level searches independently; the runner schedules single-GPU jobs.
exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_experiment1.py" \
  "${run_args[@]}" \
  --stage research \
  --search-config "${CODE_DIR}/configs/experiment1_research.json" \
  "$@"
