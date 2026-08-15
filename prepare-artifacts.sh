#!/bin/sh
set -eu

mkdir -p aiter-cache crash-dumps profiles-current
zstd -d -f artifacts/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so.zst \
  -o patches/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so
echo 'a2912b897911c75d77611dcd42e4b0e0126bb8535f069045b32efc5f8f105610  patches/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so' | sha256sum -c -
