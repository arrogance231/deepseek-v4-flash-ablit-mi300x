#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export MODEL_DIR="${MODEL_DIR:-/mnt/model-storage/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
docker compose up -d inference
exec docker compose logs -f inference
