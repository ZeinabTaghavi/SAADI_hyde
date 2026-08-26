#!/usr/bin/env bash
# Run only MuSiQue-32K HyDE with SAADI's canonical 100-word chunk size.
#
# The existing hypotheses are reusable because chunking affects retrieval, not
# HyDE generation. --force is intentional: it replaces retrieval artifacts
# produced with a different chunk size using the current 100-word config.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash run_musique32k_hyde_saadi_chunk100_gpu0_3.sh
#
# Optional controls are forwarded to the matrix runner:
#   bash run_musique32k_hyde_saadi_chunk100_gpu0_3.sh --dry-run
#   bash run_musique32k_hyde_saadi_chunk100_gpu0_3.sh --check-only
#   bash run_musique32k_hyde_saadi_chunk100_gpu0_3.sh --install-deps

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export DATASETS_CSV="musique_32k"
export GENERATE_TABLE="0"

exec bash "${SCRIPT_DIR}/run_qasper64k_musique32k_hyde_gpu0_3.sh" --force "$@"
