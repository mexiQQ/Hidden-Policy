#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-${CODE_DIR}/runtime/experiment1/swift-smoke-v1}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E1 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

# Main experiment evaluation: CAL, TEST-Q3 and TEST-Q4 together.
exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_experiment1.py" \
  --config "${CODE_DIR}/configs/experiment1.json" \
  --run-dir "${RUN_DIR}" \
  --stage eval \
  --levels G0U0 G0U1 G1U0 G1U1 \
  --allow-test \
  "$@"
