#!/usr/bin/env bash
# Shared standalone runtime configuration for every SAADI HyDE launcher.
#
# This file deliberately has no dependency on the parent SAADI repository.

hyde_select_python() {
  local project_root="$1"
  local selected=""

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    selected="${PYTHON_BIN}"
  elif [[ -x "${project_root}/.venv/bin/python" ]]; then
    selected="${project_root}/.venv/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    selected="${VIRTUAL_ENV}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    selected="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    selected="$(command -v python)"
  else
    echo "No Python interpreter was found." >&2
    echo "Create ${project_root}/.venv or set PYTHON_BIN to an explicit interpreter." >&2
    return 2
  fi

  if [[ "${selected}" == */* ]]; then
    if [[ ! -x "${selected}" ]]; then
      echo "Configured Python is not executable: ${selected}" >&2
      return 2
    fi
  elif ! command -v "${selected}" >/dev/null 2>&1; then
    echo "Configured Python command was not found: ${selected}" >&2
    return 2
  fi

  PYTHON_BIN="${selected}"
  export PYTHON_BIN
}

hyde_configure_runtime() {
  local project_root="$1"
  local cache_root="${SAADI_HF_CACHE_ROOT:-/mnt/cache/taghavi}"

  hyde_select_python "${project_root}"

  # GPUS is the only opt-in override. An inherited CUDA_VISIBLE_DEVICES value
  # cannot silently move this experiment away from physical GPUs 0,1,2,3.
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
  export GLOBAL_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

  # Match the cache convention used by the main SAADI scripts.
  export SAADI_HF_CACHE_ROOT="${cache_root}"
  export HF_HOME="${cache_root}"
  export HF_HUB_CACHE="${cache_root}/hub"
  export HF_DATASETS_CACHE="${cache_root}/datasets"
  export TRANSFORMERS_CACHE="${cache_root}/transformers"

  # Qwen generation already uses device_map=auto in the YAML. Use the same
  # default for the 8B Qwen embedding retriever so it may use all visible GPUs.
  export HYDE_QWEN_DEVICE_MAP="${HYDE_QWEN_DEVICE_MAP:-auto}"

  if [[ "${SAADI_HF_OFFLINE:-}" =~ ^(1|true|yes|on)$ ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  fi
}

hyde_install_dependencies() {
  local project_root="$1"

  echo "Installing standalone HyDE dependencies with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install -r "${project_root}/requirements-loogle.txt"
  "${PYTHON_BIN}" -m pip install -e "${project_root}" --no-deps
}

hyde_require_dependencies() {
  local project_root="$1"
  local missing=""

  missing="$(
    "${PYTHON_BIN}" -c '
import importlib.util

requirements = (
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface_hub"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("safetensors", "safetensors"),
    ("torch", "torch"),
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
)
print(",".join(package for module, package in requirements if importlib.util.find_spec(module) is None))
'
  )"

  if [[ -n "${missing}" ]]; then
    echo "Missing Python packages in ${PYTHON_BIN}: ${missing}" >&2
    echo "Install them into this exact interpreter with:" >&2
    echo "  ${PYTHON_BIN} -m pip install -r ${project_root}/requirements-loogle.txt" >&2
    echo "  ${PYTHON_BIN} -m pip install -e ${project_root} --no-deps" >&2
    echo "Or run: bash ${project_root}/run_all_hyde_retrievers.sh --install-deps" >&2
    return 2
  fi
}

hyde_prepare_cache() {
  local directory=""
  local probe="${HF_HOME}/.saadi_hyde_write_test.$$"

  for directory in "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"; do
    if ! mkdir -p "${directory}"; then
      echo "Cannot create Hugging Face cache directory: ${directory}" >&2
      echo "Set SAADI_HF_CACHE_ROOT to a writable shared cache and rerun." >&2
      return 2
    fi
  done

  if ! (umask 077 && : > "${probe}"); then
    echo "Hugging Face cache is not writable: ${HF_HOME}" >&2
    echo "Set SAADI_HF_CACHE_ROOT to a writable shared cache and rerun." >&2
    return 2
  fi
  rm -f "${probe}"
}

hyde_require_visible_gpus() {
  local requested_count=0
  local gpu_report=""
  local -a requested_gpus=()

  IFS=',' read -r -a requested_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  requested_count="${#requested_gpus[@]}"

  if [[ "${HYDE_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    echo "  GPU preflight=skipped (HYDE_SKIP_GPU_CHECK=1)"
    return 0
  fi

  if ! gpu_report="$(
    "${PYTHON_BIN}" -c '
import sys
import torch

expected = int(sys.argv[1])
available = torch.cuda.is_available()
count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(count)] if available else []
print(f"cuda_available={available} visible_count={count} names={names}")
raise SystemExit(0 if available and count >= expected else 1)
' "${requested_count}"
  )"; then
    echo "CUDA preflight failed for CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    echo "  ${gpu_report}" >&2
    echo "PyTorch must see all ${requested_count} selected GPU(s)." >&2
    echo "Set HYDE_SKIP_GPU_CHECK=1 only for intentional CPU validation." >&2
    return 2
  fi

  echo "  GPU preflight=${gpu_report}"
}
