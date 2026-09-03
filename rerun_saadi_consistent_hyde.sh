#!/usr/bin/env bash
# Rebuild every HyDE row under the canonical SAADI labeling, population, and
# chunking contract, then generate the strict 20-row table.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}"
TABLE_ROOT="${TABLE_ROOT:-${SCRIPT_DIR}/hyde_evaluations_Tables}"

GENERATE_TABLE=0 \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
TABLE_ROOT="${TABLE_ROOT}" \
bash "${SCRIPT_DIR}/run_all_hyde_retrievers.sh" --force

GENERATE_TABLE=0 \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
TABLE_ROOT="${TABLE_ROOT}" \
bash "${SCRIPT_DIR}/run_qasper64k_musique32k_hyde_gpu0_3.sh" --force

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/generate_hyde_retriever_table.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${TABLE_ROOT}" \
  --strict

printf 'Updated SAADI-consistent HyDE table: %s/table_main_retrieval_hyde.txt\n' "${TABLE_ROOT}"
