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

# Fill missing full-pool teacher answers, then data, four LoRA adapters and CAL + Q3 + Q4 probes.
exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_experiment1.py" \
  "${run_args[@]}" \
  --stage all \
  --levels G0U0 G0U1 G1U0 G1U1 \
  --allow-test \
  "$@"
