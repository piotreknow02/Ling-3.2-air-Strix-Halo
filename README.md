# Ling-3.2-air W4A16 — one-command Podman deploy (AMD Strix Halo 128 GB)

Serves [`alesha-pro/Ling-3.2-air-W4A16`](https://huggingface.co/alesha-pro/Ling-3.2-air-W4A16)
(124B multimodal MoE, compressed-tensors W4A16 INT4, 256K native context,
vision, `ling3` reasoning/tool parsers, optional MTP K1 speculative decoding)
in a rootless Podman container on an AMD Strix Halo (Ryzen AI Max 395,
gfx1151) workstation. Engine: vLLM built from source for gfx1151 on the
self-contained ROCm nightly wheel stack.

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every var has a default
./start.sh            # build (cached) -> download model -> run -> wait healthy
./stop.sh
```

## What's here

| File | Role |
|---|---|
| `Containerfile` | Self-contained runtime: uv Python 3.12, pinned ROCm 10.1 nightly wheels, vLLM built from pinned main + `patches/vllm-gfx1151.patch`, Triton AMD flash-attention. |
| `patches/vllm-gfx1151.patch` | Curated gfx1151 fixes (safe attention-state merge, MLA prefill LSE layout, WNA16 decode fast paths) applied to the pinned vLLM main. |
| `run.sh` | Container entrypoint: ROCm SDK env + `vllm serve` flags (env-overridable). |
| `start.sh` | One-command entrypoint: preflight → build (cached) → download checkpoint (resumable) → run → health-poll. |
| `stop.sh` | Stops the container with a 60 s grace period; idempotent. |
| `.env.sample` | Optional overrides — copy to `.env`. |

The 72.5 GiB checkpoint is **not** baked into the image; it is bind-mounted
read-only at `/models`. The image is ~30 GB (runtime + compiled vLLM).

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- **AMD GPU driver** present: `/dev/kfd` and `/dev/dri/renderD128` exist
- Current user in the **`render`** (and ideally `video`) group — check with `id -nG`
- **~110 GB free disk**: 72.5 GiB weights + ~30 GB image + caches
- **128 GB unified RAM** (the 256K profile budgets ~92 GB for KV at util 0.72)
- `curl` on PATH; a HuggingFace downloader (`hf` or `uv`/`uvx`) for the first run

No root/sudo is required for the normal path. Podman runs rootless; GPU access
comes from `--group-add keep-groups` (carries your host `render` group into the
container) plus SELinux `:z` bind relabeling.

## Quick start

```bash
cp .env.sample .env      # optional
./start.sh
```

First run:
1. Builds the image (~20–40 min: multi-GB wheel download + vLLM C++/HIP compile).
2. Downloads the 72.5 GiB checkpoint to `./models` (resumable).
3. Starts the container and streams the engine log to `.ling32.log`.
4. Exits "ready" once `/health` returns 200.

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> Ling-3.2-air-W4A16
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` after the first time skips the build and the download
and reaches ready materially faster (Triton/vLLM autotune caches live in
`./.cache`).

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first):
**exported shell env → `.env` → `start.sh` defaults.**

| Var | Default | Meaning |
|---|---|---|
| `PROFILE` | `256k` | `256k` (validated) or `1m-yarn` (experimental, 2 seqs) |
| `MTP` | `0` | `1` = K1 speculative decoding via the MTP head (opt-in) |
| `GPU_MEMORY_UTILIZATION` | *(profile default)* | Override the 0.72 / 0.82 fraction; lower to free RAM |
| `PORT` | `8080` | Host publish port (container always listens on 18081) |
| `MODEL_DIR` | `./models` | Where the checkpoint lives / downloads to |
| `CACHE_DIR` | `./.cache` | vLLM/Triton/HF caches (persist autotune across restarts) |
| `IMAGE` | `ling32-air-strix:latest` | Image tag |
| `CONTAINER_NAME` | `ling32-air-strix` | Container name |
| `HF_TOKEN` | *(unset)* | Optional HuggingFace token — higher download rate limits |

## Profiles

**`256k` (default, validated)** — `max_model_len 262144` (the checkpoint's
native context), `max_num_seqs 6`, `gpu_memory_utilization 0.72`,
`max_num_batched_tokens 8192`. Six requests run concurrently; extras queue in
the vLLM scheduler (by design).

**`1m-yarn` (experimental)** — 1M context via YaRN rope overrides
(`--hf-overrides`), 2 concurrent sequences, util 0.82. Slower, higher memory,
less tested.

```bash
PROFILE=1m-yarn ./start.sh
```

## Recommended sampling

From the quant model card:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ling-3.2-air-W4A16",
    "messages": [{"role": "user", "content": "Explain what a KV cache is, briefly."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": true}
  }' | jq -r '.choices[0].message'
```

With `enable_thinking: true` the chain of thought is returned in
`message.reasoning` (this vLLM build's `ling3` parser field name; the answer
stays in `message.content`).

## Vision

The checkpoint ships a ViT (`--trust-remote-code` custom code is enabled by
default in `run.sh`). Any OpenAI-compatible image message works:

```bash
IMG_B64=$(base64 -w0 red.png)   # any small PNG/JPEG
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ling-3.2-air-W4A16",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"${IMG_B64}"'"}},
      {"type": "text", "text": "What color is this image?"}
    ]}],
    "temperature": 0,
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
```

## Troubleshooting

**Cold start prints `tl.make_block_ptr is deprecated`, `TypedStorage is deprecated`,
`AUTOTUNE mm(...)`** — NOT failures. Wait for `Application startup complete`,
then `/health` returns 200. A real failure is a traceback / `EngineCore failed` /
nonzero exit.

**`Cannot allocate memory` / HSA pin errors at startup** — the user session's
`memlock` limit is too low for ROCm memory pinning. `start.sh` only warns about
this (it does not block). If the engine actually fails to pin, raise it once
(needs sudo):

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/10-memlock.conf
sudo systemctl daemon-reload
# then log out and back in (or: sudo systemctl restart user@$(id -u).service — kills your session)
```

Many setups run fine without this; only apply it if you see the pin error.

**SELinux `Permission denied` on `/models`** — confirm the `:z` suffix on the bind
(start.sh sets it). If you relabeled the directory manually, `restorecon -R` it or
just re-run `./start.sh`.

**Host OOM under load** — close other apps or lower `GPU_MEMORY_UTILIZATION`
(0.72 × 128 GB ≈ 92 GB KV budget on a 128 GB box).

**Concurrency** — six requests max on the 256K profile; extras queue in the vLLM
scheduler (by design).

**Engine stalls when idle** — the CIRU recipe for the older ROCm 7.15 wheel
stack needed a custom-ROCr `LD_PRELOAD` idle fix. This recipe uses the newer
ROCm 10.1 wheels and ships no such override; if you hit idle hangs, that
upstream rebuild (see the CIRU package's `install-rocr-idle-fix.sh`) is the
known workaround.

## Notes

- **No system ROCm in the image.** The wheels are self-contained
  (`_rocm_sdk_devel` inside the venv); mixing a system ROCm with them is
  forbidden. The container distro (Ubuntu 24.04) is independent of the host
  distro — only the kernel is shared.
- **`--ipc host` is load-bearing** — the GPU runtime dies at startup without it;
  no `shm_size` substitutes.
- **Docker instead of Podman** — the same `Containerfile` works under a rootful
  Docker daemon with `docker run --device /dev/kfd --device /dev/dri
  --group-add video,render --ipc host --ulimit memlock=-1:-1`. Swap the CLI in
  `start.sh` if you prefer Docker.

## Attribution

- Base model: inclusionAI **Ling-3.2-air** (MIT).
- Quantization: [`alesha-pro/Ling-3.2-air-W4A16`](https://huggingface.co/alesha-pro/Ling-3.2-air-W4A16) (MIT).
- Engine: [vllm-project/vLLM](https://github.com/vllm-project/vllm) (Apache-2.0), built from a pinned main commit.
- The gfx1151 patches in `patches/vllm-gfx1151.patch` are derived from the
  CIRU runtime patch of
  [`jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native`](https://huggingface.co/jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native),
  re-applied to current vLLM main (obsolete hunks dropped, conflicts resolved).
