#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated}"
MODEL_REVISION="${MODEL_REVISION:-6de83db0be050e0338ae2f8376440642203ad90d}"
MODEL_DIR="${MODEL_DIR:-/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/.cache/huggingface}"
VLLM_IMAGE="${VLLM_IMAGE:-rocm:latest}"

mkdir -p "$MODEL_DIR" "$HF_CACHE_DIR"
echo "Downloading $MODEL_ID@$MODEL_REVISION to $MODEL_DIR"
docker run --rm --entrypoint /usr/local/bin/hf -e HF_TOKEN \
  -v "$MODEL_DIR:/model" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "$VLLM_IMAGE" download "$MODEL_ID" \
  --revision "$MODEL_REVISION" --local-dir /model

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
echo "Model download/verification complete"
