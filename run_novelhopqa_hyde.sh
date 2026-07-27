#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_env.sh
source "${SCRIPT_DIR}/runtime_env.sh"
hyde_configure_runtime "${SCRIPT_DIR}"
hyde_require_dependencies "${SCRIPT_DIR}"
hyde_prepare_cache
hyde_require_visible_gpus

DEFAULT_BOOKS_ROOT="${SCRIPT_DIR}/../../novelhopqa/book-corpus-root"
if [[ -z "${NOVELHOPQA_BOOKS_ROOT:-}" && -f "${DEFAULT_BOOKS_ROOT}/bookmeta.json" ]]; then
  export NOVELHOPQA_BOOKS_ROOT="${DEFAULT_BOOKS_ROOT}"
fi

echo "HyDE–NovelHopQA standalone runner"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  HF_HOME=${HF_HOME}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  NOVELHOPQA_BOOKS_ROOT=${NOVELHOPQA_BOOKS_ROOT:-<not set>}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loogle_hyde.py" \
  --config "${HYDE_CONFIG:-${SCRIPT_DIR}/configs/novelhopqa_hyde.yaml}" \
  --output-root "${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}" \
  --work-root "${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}" \
  "$@"
