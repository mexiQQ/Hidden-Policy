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

# LEVEL selects the frozen policy job; both reuse Qwen1.5-0.5B raw labels.
LEVEL="${LEVEL:-G1U1}"
case "${LEVEL}" in
  G0U1) RATES=(2e-4 3e-4 4e-4); RUN_NAME="g0u1-raw-lr-sweep-v1" ;;
  G1U1) RATES=(4e-4 5e-4 7e-4); RUN_NAME="g1u1-raw-high-lr-sweep-v1" ;;
  *) printf 'LEVEL must be G0U1 or G1U1\n' >&2; exit 2 ;;
esac

exec "${PYTHON}" "${CODE_DIR}/scripts/e1/run_training_sweep.py" \
  --source-run "${CODE_DIR}/runtime/experiment1/u1-qwen15-v2-gates-v1" \
  --level "${LEVEL}" --run-dir "${RUN_DIR:-${CODE_DIR}/runtime/experiment1/${RUN_NAME}}" \
  --learning-rates "${RATES[@]}" --epochs 8 --checkpoint-every-epochs 1 \
  --gpus 0,1,2 "$@"
