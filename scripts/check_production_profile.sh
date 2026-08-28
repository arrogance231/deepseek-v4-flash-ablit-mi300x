#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROFILE_FILE="${PROFILE_FILE:-configs/production-k5.env}"
set -a
# shellcheck disable=SC1090
source "$PROFILE_FILE"
set +a

container="${COMPOSE_PROJECT_NAME:-deepseek-v4-flash-ablit}-inference-1"
health="$(docker inspect -f '{{.State.Health.Status}}' "$container")"
mount="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models/deepseek"}}{{.Source}}{{end}}{{end}}' "$container")"
container_k="$(docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' "$container" | sed -n 's/.*num-speculative-tokens=//p' | head -1)"
container_disable="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container" | sed -n 's/^DISABLE_DSPARK=//p')"

test "$health" = healthy || { echo "FAIL: container health=$health" >&2; exit 1; }
test "$mount" = "$MODEL_DIR" || { echo "FAIL: model mount=$mount (expected $MODEL_DIR)" >&2; exit 1; }
test "$container_k" = "$DS_NUM_SPECULATIVE_TOKENS" || { echo "FAIL: DSpark K=$container_k (expected $DS_NUM_SPECULATIVE_TOKENS)" >&2; exit 1; }
test "$container_disable" = "$DISABLE_DSPARK" || { echo "FAIL: DISABLE_DSPARK=$container_disable (expected $DISABLE_DSPARK)" >&2; exit 1; }
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
echo "PASS: promoted profile is healthy (model=$MODEL_DIR K=$container_k DSpark=enabled)"
