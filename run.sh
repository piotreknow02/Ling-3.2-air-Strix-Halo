#!/usr/bin/env bash
# Container entrypoint: serve Ling-3.2-air W4A16 on gfx1151 with vLLM/ROCm.
#
# Adapted from the CIRU package's scripts/run-common.sh. The ROCr idle-fix /
# LD_PRELOAD and bundle-head-check blocks are dropped: the ROCm 10.1 SDK
# wheels are self-contained and need no host-runtime override.
set -Eeuo pipefail

venv="${VENV:?Set VENV to the built runtime virtual environment}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the downloaded checkpoint directory}"
host="${HOST:-0.0.0.0}"
port="${PORT:-18081}"
cache_root="${CACHE_ROOT:-/cache}"
context_length="${CONTEXT_LENGTH:-262144}"
max_num_seqs="${MAX_NUM_SEQS:-6}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.72}"
mtp="${MTP:-0}"

if [ ! -x "$venv/bin/python" ]; then
  echo "VENV '$venv' has no executable python; was the image built correctly?" >&2
  exit 1
fi
if [ ! -x "$venv/bin/vllm" ]; then
  echo "VENV '$venv' has no vllm executable; was the image built correctly?" >&2
  exit 1
fi
if [ ! -d "$model_path" ]; then
  echo "MODEL_PATH '$model_path' is not a directory" >&2
  exit 1
fi
for required_model_file in config.json model.safetensors.index.json; do
  if [ ! -f "$model_path/$required_model_file" ]; then
    echo "MODEL_PATH is missing $required_model_file" >&2
    exit 1
  fi
done

site="$("$venv/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
sdk="$site/_rocm_sdk_devel"
core="$site/_rocm_sdk_core"
if [ ! -x "$sdk/bin/hipcc" ]; then
  echo "ROCm SDK devel root missing at $sdk" >&2
  exit 1
fi

mkdir -p "$cache_root/vllm" "$cache_root/triton" "$cache_root/huggingface-modules"

export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"
export ROCM_PATH="$sdk"
export ROCM_HOME="$sdk"
export ROCM_SOURCE_DIR="$sdk"
export HIP_PATH="$sdk"
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_SAFE_MERGE_ATTN_STATES=1
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=1
export VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
export VLLM_CACHE_ROOT="$cache_root/vllm"
export TRITON_CACHE_DIR="$cache_root/triton"
export HF_MODULES_CACHE="$cache_root/huggingface-modules"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=FALSE
unset TORCH_BLAS_PREFER_HIPBLASLT
unset PYTORCH_TUNABLEOP_ENABLED
unset PYTORCH_TUNABLEOP_TUNING
unset VLLM_TUNED_CONFIG_FOLDER

args=(
  serve "$model_path"
  --host "$host"
  --port "$port"
  --served-model-name Ling-3.2-air-W4A16
  --trust-remote-code
  --attention-backend TRITON_MLA
  --moe-backend triton
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser ling3
  --reasoning-parser ling3
  --max-model-len "$context_length"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
  --gpu-memory-utilization "$gpu_memory_utilization"
  # NB: no compile_sizes — with cudagraph_mode NONE the dynamic (1,max) range
  # graph covers decode sizes, and static-size graphs bake contiguous strides
  # that fail the intentionally-strided M-RoPE positions guard (vLLM #12128).
  --compilation-config '{"mode":3,"cudagraph_mode":"NONE"}'
)

# MTP K1 speculative decoding is opt-in (model-card default is off).
if [ "$mtp" = 1 ]; then
  args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi

if [ -n "${HF_OVERRIDES:-}" ]; then
  args+=(--hf-overrides "$HF_OVERRIDES")
fi

printf 'Launching Ling-3.2-air-W4A16\n'
printf '  vLLM: %s\n  model: %s\n  listen: %s:%s\n' \
  "$venv/bin/vllm" "$model_path" "$host" "$port"
printf '  context: %s  max_num_seqs: %s  util: %s  MTP: %s\n' \
  "$context_length" "$max_num_seqs" "$gpu_memory_utilization" "$mtp"
printf '%s\n' \
  'Cold-cache startup compiles and benchmarks kernels for several minutes.' \
  'The make_block_ptr/TypedStorage deprecations and AUTOTUNE tables are nonfatal.' \
  'The server is ready only after: Application startup complete.'

exec "$venv/bin/vllm" "${args[@]}"
