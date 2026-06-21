#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_CACHE_ROOT="${SAADI_HF_CACHE_ROOT:-${HF_HOME:-/mnt/cache/taghavi}}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

echo "HyDE–LooGLE standalone runner"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  HF_HOME=${HF_HOME}"
echo "  PYTHON_BIN=${PYTHON_BIN}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loogle_hyde.py" \
  --config "${HYDE_CONFIG:-${SCRIPT_DIR}/configs/loogle_hyde.yaml}" \
  --output-root "${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}" \
  --work-root "${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}" \
  "$@"
