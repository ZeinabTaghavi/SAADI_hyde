#!/usr/bin/env bash
# Rebuild every HyDE row under the canonical SAADI labeling, population, and
# chunking contract, then generate the strict 20-row table. Server paths match
# the other SAADI method launchers but remain overrideable through the
# corresponding environment variables.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/hyde_evaluations}"
WORK_ROOT="${WORK_ROOT:-${ROOT_DIR}/hyde_runs}"
TABLE_ROOT="${TABLE_ROOT:-${ROOT_DIR}/hyde_evaluations_Tables}"

HF_CACHE_ROOT="${SAADI_HF_CACHE_ROOT:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-/home/iataghav/data/passing_meta_tag/novelhopqa/book-corpus-root}"

printf 'Standalone HyDE root: %s\n' "${ROOT_DIR}"
printf 'Python: %s\n' "${PYTHON_BIN}"
printf 'Hugging Face cache: %s\n' "${HF_HOME}"
printf 'NovelHopQA books root: %s\n' "${NOVELHOPQA_BOOKS_ROOT}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES}"

if [[ ! -f "${NOVELHOPQA_BOOKS_ROOT}/bookmeta.json" || ! -d "${NOVELHOPQA_BOOKS_ROOT}/Books" ]]; then
  printf 'NovelHopQA corpus is incomplete under %s; expected bookmeta.json and Books/.\n' \
    "${NOVELHOPQA_BOOKS_ROOT}" >&2
  printf 'Set NOVELHOPQA_BOOKS_ROOT to the correct corpus directory and rerun.\n' >&2
  exit 2
fi

GENERATE_TABLE=0 \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
TABLE_ROOT="${TABLE_ROOT}" \
bash "${ROOT_DIR}/run_all_hyde_retrievers.sh" --force

GENERATE_TABLE=0 \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
TABLE_ROOT="${TABLE_ROOT}" \
bash "${ROOT_DIR}/run_qasper64k_musique32k_hyde_gpu0_3.sh" --force

"${PYTHON_BIN}" -B "${ROOT_DIR}/generate_hyde_retriever_table.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${TABLE_ROOT}" \
  --strict

printf 'Updated SAADI-consistent HyDE table: %s/table_main_retrieval_hyde.txt\n' "${TABLE_ROOT}"
