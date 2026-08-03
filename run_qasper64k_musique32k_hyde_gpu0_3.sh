#!/usr/bin/env bash
# Run the exact main-SAADI QASPER-64K and MuSiQue-32K HyDE matrix.
#
# Default grid:
#   datasets:   qasper_64k,musique_32k
#   retrievers: bm25,contriever,bge_m3,jina,qwen
#   GPUs:       physical 0,1,2,3
#
# The script validates the frozen bundles, resumes shared HyDE generation,
# skips configuration-matching completed cells, distributes retrieval across
# four GPUs, and writes one strict four-dataset main-style table.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash run_qasper64k_musique32k_hyde_gpu0_3.sh
#
# Useful controls:
#   bash run_qasper64k_musique32k_hyde_gpu0_3.sh --check-only
#   bash run_qasper64k_musique32k_hyde_gpu0_3.sh --dry-run
#   bash run_qasper64k_musique32k_hyde_gpu0_3.sh --force
#   bash run_qasper64k_musique32k_hyde_gpu0_3.sh --install-deps

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REQUESTED_GPUS="${GPU_IDS_CSV:-${CUDA_VISIBLE_DEVICES:-${GPUS:-0,1,2,3}}}"
export GPUS="${REQUESTED_GPUS}"

# shellcheck source=runtime_env.sh
source "${SCRIPT_DIR}/runtime_env.sh"
hyde_configure_runtime "${SCRIPT_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/hyde_evaluations}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/hyde_runs}"
TABLE_ROOT="${TABLE_ROOT:-${SCRIPT_DIR}/hyde_evaluations_Tables}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
DATASETS_CSV="${DATASETS_CSV:-qasper_64k,musique_32k}"
RETRIEVERS_CSV="${RETRIEVERS_CSV:-bm25,contriever,bge_m3,jina,qwen}"

CHECK_ONLY=0
DRY_RUN=0
FORCE=0
INSTALL_DEPS=0

usage() {
  printf 'Usage: %s [--check-only] [--dry-run] [--force] [--install-deps]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    --install-deps) INSTALL_DEPS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  hyde_install_dependencies "${SCRIPT_DIR}"
fi

IFS=',' read -r -a GPU_IDS <<< "${REQUESTED_GPUS}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a RETRIEVERS <<< "${RETRIEVERS_CSV}"

if [[ "${#GPU_IDS[@]}" -ne 4 ]]; then
  printf 'Exactly four physical GPU IDs are required; got %s.\n' "${REQUESTED_GPUS}" >&2
  exit 2
fi

config_for_dataset() {
  case "$1" in
    qasper_64k) printf '%s/configs/qasper_64k_hyde.yaml\n' "${SCRIPT_DIR}" ;;
    musique_32k) printf '%s/configs/musique_32k_hyde.yaml\n' "${SCRIPT_DIR}" ;;
    *) printf 'Unsupported dataset: %s\n' "$1" >&2; return 2 ;;
  esac
}

run_name_for_dataset() {
  case "$1" in
    qasper_64k) printf 'qasper_64k_retrieval_ablation_hyde\n' ;;
    musique_32k) printf 'musique_32k_retrieval_ablation_hyde\n' ;;
    *) return 2 ;;
  esac
}

validate_retriever() {
  case "$1" in
    bm25|contriever|bge_m3|jina|qwen) ;;
    *) printf 'Unsupported retriever: %s\n' "$1" >&2; return 2 ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config_for_dataset "${dataset}" >/dev/null
done
for retriever in "${RETRIEVERS[@]}"; do
  retriever="${retriever//[[:space:]]/}"
  validate_retriever "${retriever}"
done

artifact_path() {
  local dataset="$1"
  local retriever="$2"
  local top_k="$3"
  local run_name="$4"
  printf '%s/%s/hyde/%s/top_%s/%s/leaderboard_row.json\n' \
    "${OUTPUT_ROOT}" "${dataset}" "${retriever}" "${top_k}" "${run_name}"
}

cell_status() {
  local dataset="$1"
  local retriever="$2"
  local config="$3"
  local run_name="$4"
  local present=0
  local valid=0
  local mismatched=0
  local top_k leaderboard manifest
  for top_k in 5 10; do
    leaderboard="$(artifact_path "${dataset}" "${retriever}" "${top_k}" "${run_name}")"
    [[ -f "${leaderboard}" ]] || continue
    present=$((present + 1))
    manifest="${leaderboard%/leaderboard_row.json}/evaluation_manifest.json"
    if "${PYTHON_BIN}" -B "${SCRIPT_DIR}/check_hyde_artifact.py" \
      --manifest "${manifest}" \
      --config "${config}" \
      --dataset "${dataset}" \
      --retriever "${retriever}" \
      --top-k "${top_k}" >/dev/null 2>&1; then
      valid=$((valid + 1))
    else
      mismatched=1
    fi
  done
  if [[ "${valid}" -eq 2 ]]; then
    printf 'complete\n'
  elif [[ "${mismatched}" -eq 1 ]]; then
    printf 'mismatch\n'
  elif [[ "${present}" -gt 0 ]]; then
    printf 'partial\n'
  else
    printf 'missing\n'
  fi
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

printf 'Validating exact main-SAADI expansion bundles.\n'
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/validate_expanded_datasets.py"

printf 'HyDE expanded-dataset matrix\n'
printf '  GPUs=%s\n' "${REQUESTED_GPUS}"
printf '  datasets=%s\n' "${DATASETS_CSV}"
printf '  retrievers=%s\n' "${RETRIEVERS_CSV}"
printf '  output_root=%s\n' "${OUTPUT_ROOT}"
printf '  work_root=%s\n' "${WORK_ROOT}"

printf '%-14s %-14s %-10s\n' DATASET RETRIEVER STATUS
printf '%-14s %-14s %-10s\n' ------- --------- ------
HAS_MISMATCH=0
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config="$(config_for_dataset "${dataset}")"
  run_name="$(run_name_for_dataset "${dataset}")"
  for retriever in "${RETRIEVERS[@]}"; do
    retriever="${retriever//[[:space:]]/}"
    status="$(cell_status "${dataset}" "${retriever}" "${config}" "${run_name}")"
    printf '%-14s %-14s %-10s\n' "${dataset}" "${retriever}" "${status}"
    [[ "${status}" != "mismatch" ]] || HAS_MISMATCH=1
  done
done

if [[ "${CHECK_ONLY}" == "1" ]]; then
  exit 0
fi
if [[ "${HAS_MISMATCH}" == "1" && "${FORCE}" != "1" ]]; then
  printf 'At least one artifact has a configuration mismatch; inspect it or rerun with --force.\n' >&2
  exit 3
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  hyde_require_dependencies "${SCRIPT_DIR}"
  hyde_prepare_cache
  hyde_require_visible_gpus
  mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${WORK_ROOT}" "${TABLE_ROOT}"
fi

# Generate each dataset's hypotheses once. A rerun reads the JSONL cache and
# generates only missing or configuration-incompatible query entries.
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  config="$(config_for_dataset "${dataset}")"
  printf 'Preparing/resuming shared hypotheses for %s\n' "${dataset}"
  run_or_print env \
    "CUDA_VISIBLE_DEVICES=${REQUESTED_GPUS}" \
    "GLOBAL_VISIBLE_DEVICES=${REQUESTED_GPUS}" \
    "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_loogle_hyde.py" \
      --config "${config}" \
      --output-root "${OUTPUT_ROOT}" \
      --work-root "${WORK_ROOT}" \
      --generate-only
done

JOB_DATASETS=()
JOB_RETRIEVERS=()
for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  for retriever in "${RETRIEVERS[@]}"; do
    retriever="${retriever//[[:space:]]/}"
    JOB_DATASETS+=("${dataset}")
    JOB_RETRIEVERS+=("${retriever}")
  done
done

run_cell() {
  local dataset="$1"
  local retriever="$2"
  local physical_gpu="$3"
  local config run_name status log_file
  local -a command
  config="$(config_for_dataset "${dataset}")"
  run_name="$(run_name_for_dataset "${dataset}")"
  status="$(cell_status "${dataset}" "${retriever}" "${config}" "${run_name}")"
  if [[ "${status}" == "complete" && "${FORCE}" != "1" ]]; then
    printf '[gpu %s] Skipping completed %s/%s\n' "${physical_gpu}" "${dataset}" "${retriever}"
    return 0
  fi

  command=(
    env
    "CUDA_VISIBLE_DEVICES=${physical_gpu}"
    "GLOBAL_VISIBLE_DEVICES=${physical_gpu}"
    "HYDE_QWEN_DEVICE_MAP=auto"
    "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_loogle_hyde.py"
    --config "${config}"
    --output-root "${OUTPUT_ROOT}"
    --work-root "${WORK_ROOT}"
    --run-name "${run_name}"
    --retriever "${retriever}"
    --retrieval-only
    --embedding-device cuda:0
    --top-ks 5 10
  )
  if [[ "${FORCE}" == "1" ]]; then
    command+=(--force)
  fi
  printf '[gpu %s] Running %s/%s\n' "${physical_gpu}" "${dataset}" "${retriever}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${command[@]}"
    return 0
  fi
  log_file="${LOG_DIR}/hyde_${dataset}_${retriever}.log"
  "${command[@]}" 2>&1 | tee "${log_file}"
  status="$(cell_status "${dataset}" "${retriever}" "${config}" "${run_name}")"
  if [[ "${status}" != "complete" ]]; then
    printf 'Cell is incomplete after execution: %s/%s (%s)\n' \
      "${dataset}" "${retriever}" "${status}" >&2
    return 1
  fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for ((job_index = 0; job_index < ${#JOB_DATASETS[@]}; job_index++)); do
    gpu_index=$((job_index % ${#GPU_IDS[@]}))
    run_cell "${JOB_DATASETS[$job_index]}" "${JOB_RETRIEVERS[$job_index]}" "${GPU_IDS[$gpu_index]}"
  done
else
  worker_pids=()
  for ((gpu_index = 0; gpu_index < ${#GPU_IDS[@]}; gpu_index++)); do
    (
      for ((job_index = gpu_index; job_index < ${#JOB_DATASETS[@]}; job_index += ${#GPU_IDS[@]})); do
        run_cell \
          "${JOB_DATASETS[$job_index]}" \
          "${JOB_RETRIEVERS[$job_index]}" \
          "${GPU_IDS[$gpu_index]}"
      done
    ) &
    worker_pids+=("$!")
  done

  failed_workers=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      failed_workers=$((failed_workers + 1))
    fi
  done
  if [[ "${failed_workers}" -ne 0 ]]; then
    printf '%s GPU worker(s) failed; the strict table was not generated.\n' "${failed_workers}" >&2
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  printf 'Generating strict four-dataset HyDE main table.\n'
  "${PYTHON_BIN}" -B "${SCRIPT_DIR}/generate_hyde_retriever_table.py" \
    --input-root "${OUTPUT_ROOT}" \
    --output-dir "${TABLE_ROOT}" \
    --strict
fi

printf 'HyDE expanded-dataset matrix finished.\n'
printf 'Table: %s/table_main_retrieval_hyde.txt\n' "${TABLE_ROOT}"
