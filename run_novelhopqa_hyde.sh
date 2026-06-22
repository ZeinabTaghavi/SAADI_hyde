#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}"
export GLOBAL_VISIBLE_DEVICES="${GLOBAL_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_CACHE_ROOT="${SAADI_HF_CACHE_ROOT:-${HF_HOME:-/mnt/cache/taghavi}}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

if [[ "${SAADI_HF_OFFLINE:-}" =~ ^(1|true|yes|on)$ ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

DEFAULT_BOOKS_ROOT="${SCRIPT_DIR}/../../novelhopqa/book-corpus-root"
if [[ -z "${NOVELHOPQA_BOOKS_ROOT:-}" && -f "${DEFAULT_BOOKS_ROOT}/bookmeta.json" ]]; then
  export NOVELHOPQA_BOOKS_ROOT="${DEFAULT_BOOKS_ROOT}"
fi

echo "HyDE–NovelHopQA standalone runner"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  HF_HOME=${HF_HOME}"
echo "  NOVELHOPQA_BOOKS_ROOT=${NOVELHOPQA_BOOKS_ROOT:-<not set>}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loogle_hyde.py" \
  --config "${HYDE_CONFIG:-${SCRIPT_DIR}/configs/novelhopqa_hyde.yaml}" \
  --output-root "${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}" \
  --work-root "${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}" \
  "$@"
