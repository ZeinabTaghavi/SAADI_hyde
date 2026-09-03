#!/usr/bin/env bash
# Run the standalone HyDE retrieval matrix for LooGLE and NovelHopQA.
#
# Environment overrides:
#   DATASETS_CSV=loogle,novelhopqa
#   RETRIEVERS_CSV=bm25,contriever,bge_m3,qwen,jina
#   GPUS=0,1,2,3
#   SAADI_HF_CACHE_ROOT=/writable/huggingface/cache
#   NOVELHOPQA_BOOKS_ROOT=/path/to/book-corpus-root
#   OUTPUT_ROOT=/path/to/hyde_evaluations
#   WORK_ROOT=/path/to/hyde_runs
#   HYDE_QWEN_DEVICE_MAP=auto
#   PYTHON_BIN=/path/to/python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}"
TABLE_ROOT="${TABLE_ROOT:-${SCRIPT_DIR}/hyde_evaluations_Tables}"
GENERATE_TABLE="${GENERATE_TABLE:-1}"
DATASETS_CSV="${DATASETS_CSV:-loogle,novelhopqa}"
RETRIEVERS_CSV="${RETRIEVERS_CSV:-bm25,contriever,bge_m3,qwen,jina}"

CHECK_ONLY=0
DRY_RUN=0
FORCE=0
INSTALL_DEPS=0

usage() {
  echo "Usage: $0 [--check-only] [--dry-run] [--force] [--install-deps]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --force)
      FORCE=1
      ;;
    --install-deps)
      INSTALL_DEPS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# shellcheck source=runtime_env.sh
source "${SCRIPT_DIR}/runtime_env.sh"
hyde_configure_runtime "${SCRIPT_DIR}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  hyde_install_dependencies "${SCRIPT_DIR}"
fi

DEFAULT_BOOKS_ROOT="${SCRIPT_DIR}/data/novelhopqa/book-corpus-root"
if [[ -z "${NOVELHOPQA_BOOKS_ROOT:-}" && -f "${DEFAULT_BOOKS_ROOT}/bookmeta.json" ]]; then
  export NOVELHOPQA_BOOKS_ROOT="${DEFAULT_BOOKS_ROOT}"
fi

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a RETRIEVERS <<< "${RETRIEVERS_CSV}"

config_for_dataset() {
  case "$1" in
    loogle)
      echo "${SCRIPT_DIR}/configs/loogle_hyde.yaml"
      ;;
    novelhopqa)
      echo "${SCRIPT_DIR}/configs/novelhopqa_hyde.yaml"
      ;;
    *)
      echo "Unsupported dataset '$1'. Expected loogle or novelhopqa." >&2
      return 2
      ;;
  esac
}

validate_retriever() {
  case "$1" in
    bm25|contriever|bge_m3|qwen|jina)
      ;;
    *)
      echo "Unsupported retriever '$1'. Expected bm25, contriever, bge_m3, qwen, or jina." >&2
      return 2
      ;;
  esac
}

run_name_for_dataset() {
  echo "$1_retrieval_ablation_hyde"
}

canonical_result_path() {
  local dataset="$1"
  local retriever="$2"
  local top_k="$3"
  local run_name="$4"
  echo "${OUTPUT_ROOT}/${dataset}/hyde/${retriever}/top_${top_k}/${run_name}/leaderboard_row.json"
}

legacy_contriever_result_path() {
  local dataset="$1"
  local top_k="$2"
  local run_name="$3"
  echo "${OUTPUT_ROOT}/${dataset}/hyde/top_${top_k}/${run_name}/leaderboard_row.json"
}

run_status() {
  local dataset="$1"
  local retriever="$2"
  local run_name="$3"
  local config="$4"
  local completed=0
  local present=0
  local mismatch=0
  local top_k
  for top_k in 5 10; do
    local canonical
    local selected=""
    canonical="$(canonical_result_path "${dataset}" "${retriever}" "${top_k}" "${run_name}")"
    if [[ -f "${canonical}" ]]; then
      selected="${canonical}"
    elif [[ "${retriever}" == "contriever" ]]; then
      local legacy
      legacy="$(legacy_contriever_result_path "${dataset}" "${top_k}" "${run_name}")"
      if [[ -f "${legacy}" ]]; then
        selected="${legacy}"
      fi
    fi
    if [[ -z "${selected}" ]]; then
      continue
    fi
    present=$((present + 1))
    manifest="${selected%/leaderboard_row.json}/evaluation_manifest.json"
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/check_hyde_artifact.py" \
      --manifest "${manifest}" \
      --config "${config}" \
      --dataset "${dataset}" \
      --retriever "${retriever}" \
      --top-k "${top_k}" >/dev/null 2>&1; then
      completed=$((completed + 1))
    else
      mismatch=1
    fi
  done
  if [[ "${completed}" -eq 2 ]]; then
    echo "complete"
  elif [[ "${mismatch}" -eq 1 ]]; then
    echo "mismatch"
  elif [[ "${present}" -gt 0 ]]; then
    echo "partial"
  else
    echo "missing"
  fi
}

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

echo "Standalone HyDE retriever matrix"
echo "  datasets=${DATASETS_CSV}"
echo "  retrievers=${RETRIEVERS_CSV}"
echo "  chunk_size=500"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  GLOBAL_VISIBLE_DEVICES=${GLOBAL_VISIBLE_DEVICES}"
echo "  HF_HOME=${HF_HOME}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  WORK_ROOT=${WORK_ROOT}"
echo

printf '%-14s %-14s %-10s\n' "DATASET" "RETRIEVER" "STATUS"
printf '%-14s %-14s %-10s\n' "-------" "---------" "------"
MISMATCHES=0
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config="$(config_for_dataset "${dataset}")"
  run_name="$(run_name_for_dataset "${dataset}")"
  for retriever in "${RETRIEVERS[@]}"; do
    retriever="${retriever//[[:space:]]/}"
    validate_retriever "${retriever}"
    status="$(run_status "${dataset}" "${retriever}" "${run_name}" "${config}")"
    printf '%-14s %-14s %-10s\n' "${dataset}" "${retriever}" "${status}"
    if [[ "${status}" == "mismatch" ]]; then
      MISMATCHES=1
    fi
  done
done

if [[ "${CHECK_ONLY}" == "1" ]]; then
  exit 0
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  hyde_require_dependencies "${SCRIPT_DIR}"
  hyde_prepare_cache
  hyde_require_visible_gpus
fi
if [[ "${MISMATCHES}" == "1" && "${FORCE}" != "1" ]]; then
  echo "At least one existing artifact has a configuration mismatch." >&2
  echo "Inspect the artifacts, then rerun intentionally with --force." >&2
  exit 3
fi

# Phase 1: generate every missing hypothetical document once per dataset.
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config="$(config_for_dataset "${dataset}")"
  echo
  echo "Preparing shared HyDE hypotheses for ${dataset}"
  run_command \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loogle_hyde.py" \
    --config "${config}" \
    --output-root "${OUTPUT_ROOT}" \
    --work-root "${WORK_ROOT}" \
    --generate-only
done

# Phase 2: reuse the frozen hypotheses for every selected retriever.
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config="$(config_for_dataset "${dataset}")"
  run_name="$(run_name_for_dataset "${dataset}")"
  for retriever in "${RETRIEVERS[@]}"; do
    retriever="${retriever//[[:space:]]/}"
    validate_retriever "${retriever}"
    status="$(run_status "${dataset}" "${retriever}" "${run_name}" "${config}")"
    if [[ "${status}" == "complete" && "${FORCE}" != "1" ]]; then
      echo "Skipping completed dataset=${dataset} retriever=${retriever}"
      continue
    fi
    if [[ "${status}" == "mismatch" && "${FORCE}" != "1" ]]; then
      echo "Existing artifacts do not match the requested configuration for dataset=${dataset} retriever=${retriever}." >&2
      echo "Inspect with --check-only, then rerun intentionally with --force." >&2
      exit 3
    fi
    echo
    echo "Running dataset=${dataset} retriever=${retriever}"
    command=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loogle_hyde.py"
      --config "${config}"
      --output-root "${OUTPUT_ROOT}"
      --work-root "${WORK_ROOT}"
      --retriever "${retriever}"
      --retrieval-only
      --top-ks 5 10
    )
    if [[ "${FORCE}" == "1" ]]; then
      command+=(--force)
    fi
    run_command "${command[@]}"
  done
done

if [[ "${GENERATE_TABLE}" == "1" ]]; then
  echo
  echo "Generating the main-style HyDE retrieval table"
  run_command \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_hyde_retriever_table.py" \
    --input-root "${OUTPUT_ROOT}" \
    --output-dir "${TABLE_ROOT}" \
    --strict
fi

echo "HyDE retrieval matrix finished."
