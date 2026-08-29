#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROFILE_FILE="${PROFILE_FILE:-configs/production-k5.env}"
set -a
# shellcheck disable=SC1090
source "$PROFILE_FILE"
# The real key remains in the ignored local .env; never commit or print it.
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

container="${COMPOSE_PROJECT_NAME:-deepseek-v4-flash-ablit}-inference-1"
health="$(docker inspect -f '{{.State.Health.Status}}' "$container")"
mount="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models/deepseek"}}{{.Source}}{{end}}{{end}}' "$container")"
container_cmd="$(docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' "$container")"
container_k="$(printf '%s\n' "$container_cmd" | sed -n 's/.*num-speculative-tokens=//p' | head -1)"
container_len="$(printf '%s\n' "$container_cmd" | awk '/--max-model-len/{getline; print; exit}')"
container_seqs="$(printf '%s\n' "$container_cmd" | awk '/--max-num-seqs/{getline; print; exit}')"
container_disable="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container" | sed -n 's/^DISABLE_DSPARK=//p')"
port="${VLLM_PORT:-8000}"

test "$health" = healthy || { echo "FAIL: container health=$health" >&2; exit 1; }
test "$mount" = "$MODEL_DIR" || { echo "FAIL: model mount=$mount (expected $MODEL_DIR)" >&2; exit 1; }
test "$container_k" = "$DS_NUM_SPECULATIVE_TOKENS" || { echo "FAIL: DSpark K=$container_k (expected $DS_NUM_SPECULATIVE_TOKENS)" >&2; exit 1; }
test "$container_len" = "$MAX_MODEL_LEN" || { echo "FAIL: max model len=$container_len (expected $MAX_MODEL_LEN)" >&2; exit 1; }
test "$container_seqs" = "$MAX_NUM_SEQS" || { echo "FAIL: max sequences=$container_seqs (expected $MAX_NUM_SEQS)" >&2; exit 1; }
test "$container_disable" = "$DISABLE_DSPARK" || { echo "FAIL: DISABLE_DSPARK=$container_disable (expected $DISABLE_DSPARK)" >&2; exit 1; }
curl -fsS "http://127.0.0.1:$port/health" >/dev/null
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  curl -fsS "http://127.0.0.1:$port/v1/models" \
    -H "Authorization: Bearer ${VLLM_API_KEY}" >/dev/null
else
  curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null
fi
echo "PASS: promoted profile is healthy (model=$MODEL_DIR len=$container_len seqs=$container_seqs K=$container_k DSpark=enabled)"
