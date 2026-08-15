#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. This host serves one vLLM instance.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete
cp /opt/cj-moe/opus942/module_pa_sparse_prefill_opus942.so \
  /usr/local/lib/python3.12/dist-packages/aiter/jit/

exec vllm serve "$@"
