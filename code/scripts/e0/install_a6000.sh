#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPOSITORY_DIR="$(cd -- "${CODE_DIR}/.." && pwd)"
HARNESS_DIR="${CODE_DIR}/vendor/lm-evaluation-harness"
SWIFT_DIR="${CODE_DIR}/vendor/ms-swift"
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
  'wheel==0.45.1'
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
  --editable "${HARNESS_DIR}[hf,vllm]" \
  --editable "${SWIFT_DIR}" \
  --editable "${CODE_DIR}" \
  qwen-vl-utils decord

# Decord's ctypes-only wheel has stale RECORD hashes and a cp36 ABI tag.
# Verify the official archive hash, then rebuild metadata; keep doctor checks intact.
WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf -- "${WHEEL_DIR}"' EXIT
python -m pip download --no-deps --only-binary=:all: --dest "${WHEEL_DIR}" \
  'https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl#sha256=51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976'
python -m zipfile -e "${WHEEL_DIR}/decord-0.6.0-py3-none-manylinux2010_x86_64.whl" \
  "${WHEEL_DIR}/unpacked"
python -m wheel pack "${WHEEL_DIR}/unpacked" --dest-dir "${WHEEL_DIR}"
python -m wheel tags --python-tag=py3 --abi-tag=none --build=1 \
  "${WHEEL_DIR}/decord-0.6.0-cp36-cp36m-manylinux2010_x86_64.whl"
python -m pip install --force-reinstall --no-deps \
  "${WHEEL_DIR}/decord-0.6.0-1-py3-none-manylinux2010_x86_64.whl"

hidden-policy-eval doctor --backend vllm
