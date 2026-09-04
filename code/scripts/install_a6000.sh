#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_DIR="$(cd -- "${CODE_DIR}/.." && pwd)"
HARNESS_DIR="${CODE_DIR}/vendor/lm-evaluation-harness"
CONSTRAINTS="${CODE_DIR}/constraints-a6000.txt"
CONDA_INITIALIZER="${HOME}/miniconda3/etc/profile.d/conda.sh"
ENVIRONMENT_NAME="hidden-policy"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"

git -C "${REPOSITORY_DIR}" submodule update --init --recursive --depth 1
if [[ ! -f "${CONDA_INITIALIZER}" ]]; then
  echo "Cannot find conda initializer: ${CONDA_INITIALIZER}" >&2
  exit 1
fi
source "${CONDA_INITIALIZER}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENVIRONMENT_NAME}"; then
  conda create -n "${ENVIRONMENT_NAME}" python=3.10 pip -y
fi
conda activate "${ENVIRONMENT_NAME}"

python -m pip install --upgrade \
  'pip>=21.3' \
  'setuptools>=64' \
  wheel
python -m pip install \
  'torch==2.13.0' \
  'torchvision==0.28.0' \
  'torchaudio==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  --constraint "${CONSTRAINTS}" \
  "${VLLM_WHEEL}"
python -m pip install \
  --constraint "${CONSTRAINTS}" \
  --editable "${HARNESS_DIR}[hf,vllm]"
python -m pip install \
  --constraint "${CONSTRAINTS}" \
  --editable "${CODE_DIR}"

hidden-policy-eval doctor --backend vllm
