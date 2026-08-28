#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. This host serves one vLLM instance.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete
cp /opt/cj-moe/opus942/module_pa_sparse_prefill_opus942.so \
  /usr/local/lib/python3.12/dist-packages/aiter/jit/

if [ "${DISABLE_DSPARK:-0}" = "1" ]; then
  # vLLM has no per-request DSpark switch.  Keep the normal compose command
  # reproducible and provide a safe A/B profile by removing only the four
  # speculative flags before exec'ing vLLM.
  exec python - "$@" <<'PY'
import os
import sys

args = []
for arg in sys.argv[1:]:
    if arg == "--speculative-config" or arg.startswith("--speculative-config."):
        continue
    args.append(arg)
if os.environ.get("VLLM_API_KEY"):
    args += ["--api-key", os.environ["VLLM_API_KEY"]]
os.execvp("vllm", ["vllm", "serve", *args])
PY
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
  set -- "$@" --api-key "$VLLM_API_KEY"
fi
exec vllm serve "$@"
