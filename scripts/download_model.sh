#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-lovesenko/DeepSeek-V4-Flash-0731-Abliterated}"
MODEL_REVISION="${MODEL_REVISION:-61ec100749f5f05cd268296c5e2eccec03268e78}"
MODEL_DIR="${MODEL_DIR:-/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/.cache/huggingface}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm@sha256:e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a}"

mkdir -p "$MODEL_DIR" "$HF_CACHE_DIR"
echo "Downloading $MODEL_ID@$MODEL_REVISION to $MODEL_DIR"
docker run --rm --entrypoint /opt/venv/bin/hf \
  -v "$MODEL_DIR:/model" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "$VLLM_IMAGE" download "$MODEL_ID" \
  --revision "$MODEL_REVISION" --local-dir /model

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
echo "Model download/verification complete"
