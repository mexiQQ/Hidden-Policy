#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Activate the E0 environment and append --run-id for this 0.8B run.
exec "${PYTHON}" "${CODE_DIR}/scripts/e0/run_baseline_matrix.py" \
  --config "${CODE_DIR}/configs/experiment0.json" \
  --scope pilot \
  --backend vllm \
  --models weak \
  --gpus 0 \
  "$@"
