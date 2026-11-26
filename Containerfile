# Ling-3.2-air W4A16 (gfx1151 / AMD Strix Halo) — self-contained vLLM runtime.
#
# uv-managed Python 3.12, self-contained ROCm 10.1 nightly wheels
# (torch/torchvision/triton/rocm), vLLM built from pinned main with the
# curated gfx1151 patch (patches/vllm-gfx1151.patch) applied, and the Triton
# AMD flash-attention build. The 72.5 GiB checkpoint is NOT in the image —
# bind-mount it at /models (ro).
#
# Do NOT add a system ROCm to this base: the wheels are self-contained
# (_rocm_sdk_devel inside the venv) and mixing a system ROCm is forbidden.
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
# Same host-dep set as the CIRU package's scripts/install-host-deps.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates pkg-config \
      libnuma-dev libdrm-dev libelf-dev xxd python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# vLLM main >= Nov 2026 is required: BailingMoeV3VLForConditionalGeneration
# (the Ling-3.2-air arch) is not in any release through v0.34.0. The patch
# is generated against exactly this SHA — do not bump without regenerating.
ARG VLLM_REVISION=bd4cb514620aa3622116afb92d8e3a5178bda990
ARG MAX_JOBS=16
ENV MAX_JOBS=16

RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm \
    && git -C /opt/vllm checkout "${VLLM_REVISION}"
COPY patches/vllm-gfx1151.patch /tmp/
RUN git -C /opt/vllm apply /tmp/vllm-gfx1151.patch

# Self-contained ROCm wheel stack from the multi-arch nightly index. The
# triton pin is the exact version torch 2.14.0+rocm10.1.0a20260822 depends on
# (vLLM main's `torch == 2.14.0` pin, pyproject.toml).
RUN uv venv --python 3.12 /opt/ling32/.venv \
    && uv pip install --python /opt/ling32/.venv/bin/python \
         --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
         'torch[device-gfx1151]==2.14.0+rocm10.1.0a20260822' \
         'torchvision[device-gfx1151]==0.30.0a0+rocm10.1.0a20260822' \
         'torchaudio==2.11.0.3+rocm10.1.0a20260822' \
         'triton==3.8.0+git675c5987.rocm10.1.0a20260822' \
         'rocm[libraries,devel,device-gfx1151]==10.1.0a20260822' \
    && /opt/ling32/.venv/bin/rocm-sdk init

# Build vLLM for gfx1151 against the SDK wheels. The SDK compatibility seds
# and the build-env block are ported from the CIRU package's
# scripts/build-vllm-gfx1151.sh (NixOS branch dropped: this image is Ubuntu).
RUN set -eux; \
    venv=/opt/ling32/.venv; \
    site="$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"; \
    sdk="$site/_rocm_sdk_devel"; \
    core="$site/_rocm_sdk_core"; \
    test -x "$sdk/bin/hipcc"; \
    sed -i 's/rtld_global: bool = True/rtld_global: bool = False/g' "$site/rocm_sdk/__init__.py"; \
    hsakmt_targets="$sdk/lib/cmake/hsakmt/hsakmtTargets.cmake"; \
    if [ -f "$hsakmt_targets" ]; then \
      sed -i 's|-L/__w/[^;"]*|-L${_IMPORT_PREFIX}/lib/rocm_sysdeps/lib|g' "$hsakmt_targets"; \
      sed -i 's|/usr/lib64/libc.so;||g' "$hsakmt_targets"; \
    fi; \
    export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"; \
    export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; \
    export PYTHONPATH="$core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"; \
    export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"; \
    export VLLM_TARGET_DEVICE=rocm; \
    export PYTORCH_ROCM_ARCH=gfx1151; \
    export ROCM_PATH="$sdk"; \
    export ROCM_HOME="$sdk"; \
    export ROCM_SOURCE_DIR="$sdk"; \
    export HIP_PATH="$sdk"; \
    export CMAKE_PREFIX_PATH="$sdk/lib/cmake:$site/torch/share/cmake"; \
    export HIP_FORCE_DEV_KERNARG=1; \
    export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"; \
    export CMAKE_ARGS="-DROCM_PATH=$sdk -DCMAKE_HIP_ARCHITECTURES=gfx1151"; \
    "$venv/bin/rocm-sdk" test; \
    uv pip install --python "$venv/bin/python" \
      'cmake>=3.26.1,<4' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<80' 'setuptools-scm>=8' \
      'setuptools-rust>=1.9' wheel 'jinja2>=3.1.6'; \
    uv pip install --python "$venv/bin/python" -r /opt/vllm/requirements/rocm.txt; \
    uv pip install --python "$venv/bin/python" --no-build-isolation --no-deps -e /opt/vllm

# Triton AMD flash-attention (same pin as the CIRU build).
RUN set -eux; \
    git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention; \
    git -C /opt/flash-attention checkout 0e60e39473e8df549a20fb5353760f7a65b30e2d; \
    venv=/opt/ling32/.venv; \
    site="$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"; \
    sdk="$site/_rocm_sdk_devel"; \
    core="$site/_rocm_sdk_core"; \
    export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"; \
    export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; \
    export PYTHONPATH="$core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"; \
    export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"; \
    export ROCM_PATH="$sdk"; \
    export ROCM_HOME="$sdk"; \
    export ROCM_SOURCE_DIR="$sdk"; \
    export HIP_PATH="$sdk"; \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      uv pip install --python "$venv/bin/python" --no-build-isolation --no-deps /opt/flash-attention

# Build sanity: torch/triton/vLLM import and the flash-attention version.
RUN set -eux; \
    venv=/opt/ling32/.venv; \
    site="$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"; \
    sdk="$site/_rocm_sdk_devel"; \
    core="$site/_rocm_sdk_core"; \
    export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"; \
    export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; \
    export PYTHONPATH="$core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"; \
    export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"; \
    "$venv/bin/python" -c "import torch, triton, vllm, vllm._C; import importlib.metadata as m; assert torch.__version__.startswith('2.14.0'), torch.__version__; assert m.version('flash_attn') == '2.8.3', m.version('flash_attn')"

ENV VENV=/opt/ling32/.venv \
    MODEL_PATH=/models \
    HOST=0.0.0.0 \
    PORT=18081 \
    CACHE_ROOT=/cache
COPY run.sh /opt/ling32/run.sh
EXPOSE 18081
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
  CMD curl -fsS http://127.0.0.1:18081/health || exit 1
CMD ["bash", "/opt/ling32/run.sh"]
