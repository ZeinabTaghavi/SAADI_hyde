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

  local selected_real=""
  local project_real=""
  selected_real="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
  project_real="$(cd "${project_root}" && pwd -P)"
  if [[ -n "${VIRTUAL_ENV:-}" && "${selected_real}" != "${project_real}/.venv/"* ]]; then
    echo "Warning: selected Python is outside this standalone folder: ${selected_real}" >&2
    echo "  Active VIRTUAL_ENV=${VIRTUAL_ENV}" >&2
    echo "  A project-local ${project_real}/.venv is preferred when available." >&2
  fi
}

hyde_configure_runtime() {
  local project_root="$1"
  local cache_root="${SAADI_HF_CACHE_ROOT:-${project_root}/.cache/huggingface}"

  hyde_select_python "${project_root}"

  # Prefer an explicit GPUS override, otherwise respect the server's existing
  # CUDA visibility. The four-GPU value remains only a convenience default.
  export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
  export GLOBAL_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  # Keep imports hermetic to this checkout instead of inheriting a parent
  # repository through the caller's PYTHONPATH.
  export PYTHONPATH="${project_root}/src"

  # Keep the default cache inside the copied folder. Large/shared server caches
  # can still be selected explicitly with SAADI_HF_CACHE_ROOT.
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
  local runtime_report=""

  if runtime_report="$(
    "${PYTHON_BIN}" -c '
import importlib.util
import json
import sys

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
missing = [package for module, package in requirements if importlib.util.find_spec(module) is None]
errors = []
transformers_version = None
qwen3_moe_supported = False
if "transformers" not in missing:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    transformers_version = transformers.__version__
    qwen3_moe_supported = "qwen3_moe" in CONFIG_MAPPING_NAMES
    if not qwen3_moe_supported:
        errors.append(
            "Transformers cannot load model_type=qwen3_moe; "
            "Qwen/Qwen3-30B-A3B-Instruct-2507 requires transformers>=4.51.0"
        )
payload = {
    "python": sys.executable,
    "missing_packages": missing,
    "transformers_version": transformers_version,
    "qwen3_moe_supported": qwen3_moe_supported,
    "errors": errors,
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if not missing and not errors else 1)
'
  )"; then
    echo "  Runtime preflight=${runtime_report}"
  else
    echo "Standalone HyDE runtime preflight failed." >&2
    echo "  ${runtime_report}" >&2
    echo "The selected interpreter is ${PYTHON_BIN}." >&2
    echo "Repair this exact interpreter with:" >&2
    echo "  ${PYTHON_BIN} -m pip install --upgrade 'transformers>=4.51.0' 'accelerate>=0.30'" >&2
    echo "  ${PYTHON_BIN} -m pip install -r ${project_root}/requirements-loogle.txt" >&2
    echo "  ${PYTHON_BIN} -m pip install -e ${project_root} --no-deps" >&2
    echo "Or rerun the matrix launcher once with --install-deps." >&2
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
