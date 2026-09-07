#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  printf 'E1 Python not found: %s\nRun conda activate hidden-policy, or set PYTHON.\n' "${PYTHON}" >&2
  exit 127
fi

# Reuse the frozen G1U1-raw job: six G1 families and Qwen1.5-0.5B raw labels.
exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_training_sweep.py" \
  --source-run "${CODE_DIR}/runtime/experiment1/u1-qwen15-v2-gates-v1" \
  --run-dir "${RUN_DIR:-${CODE_DIR}/runtime/experiment1/g1u1-raw-high-lr-sweep-v1}" \
  --learning-rates 4e-4 5e-4 7e-4 --epochs 8 --checkpoint-every-epochs 1 \
  --gpus 0,1,2 "$@"
