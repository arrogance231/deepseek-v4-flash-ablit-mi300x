# Patch provenance

The `*.py` files in this directory are **byte-for-byte the overlays that run in
production** (see `../SHA256SUMS`). They are mounted read-only over files inside
the pinned vLLM ROCm container image by `../compose.yaml`. The HIP/CUDA sources
in `../kernel-dev/hip-a8w4` are JIT-compiled at first container start (mounted
as `/opt/cj-moe`).

The `diffs/*.patch` files in this directory are informational unified diffs showing
exactly what each overlay changes relative to an upstream base revision. They were
generated with `diff -u` on 2026-08-15 against:

| Overlay | Upstream base |
| --- | --- |
| `gpt_oss_triton_kernels_moe.row-i8asym-candidate.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |
| `mxfp4.fused-silu.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| `triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py` | `ROCm/triton` @ `0f380657dbf3ee86eb57558ff71df24f03b5d4e7` — `python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py` (the revision vLLM's ROCm builds vendor) |
| `fused_compress_quant_cache.fnuz-shuffle.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` |
| `aiter_pa_mqa_logits.i64.py` | `ROCm/aiter` `main` @ `4db400a90c1c1c558f3dbb40b0e6728825bbcc2b` — `aiter/ops/triton/gluon/pa_mqa_logits.py` |
| `rocm_aiter_mla_sparse.decode-h32-k16.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| `rocm_aiter_mla.dspark-causal.py` | `vllm-project/vllm` @ `77469c9057bec3212a64877dbbf3b9c48c22d786` — `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`. This file is **identical to the upstream file at that commit**; the diff shows the change the commit itself made. |
| `dspark-speculator.independent-draft-gumbel.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` |
| `spec-decode-utils.independent-draft-gumbel.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/worker/gpu/spec_decode/utils.py` |
| `kv_offload_cpu_gpu_worker.load-war.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/kv_offload/cpu/gpu_worker.py` (post-#46278 state; PR #47291 is not merged upstream) |
| `activation.rocm-exact-swiglu.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/model_executor/layers/activation.py` |
| `block_table.active-width-copy.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/worker/block_table.py` |
| `deepseek_v4_amd_model.router-bf16.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/amd/model.py` |
| `deepseek_v4_attention.wqb-bpreshuffle.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/attention.py` |
| `deepseek_v4_rocm.wqb-bpreshuffle.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/amd/rocm.py` |
| `cache_utils.gather2048.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| `scheduler.contention-aware.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/core/sched/scheduler.py` |
| `deepseek_v4_hermes_fallback.py` | pinned vLLM image `0.26.1rc1.dev229+g124154a88` — `vllm/parser/deepseek_v4.py`, with schema-gated Hermes XML fallback |

`deepseek_v4_hermes_fallback.py` is mounted over the image's native
`vllm/parser/deepseek_v4.py`. It preserves the recommended native V4 DSML
parser and additionally converts Hermes `<execute_code>` and
`<write_file>`/`<write-files>` wrappers into OpenAI tool calls when the matching
function is included in the request.

The compiled stable-libtorch top-k extension is built from
`sampler.topk-tiebreak-sanitize.cu`; `vllm-124154a-topk-tiebreak.patch`
is the diff of that source against the pinned nightly's
`csrc/libtorch_stable/sampler.cu` (vLLM `124154a88`). The expanded
`_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so` is produced by
`../prepare-artifacts.sh` and SHA-256-verified there and in `../SHA256SUMS`.

`rocm_aiter_mla_sparse.topk-tiebreak.py` is a **superseded variant that is
not mounted**: it was the deterministic-tie-break sparse kernel before the compiled
extension took over that role. It is retained so the two final deltas stay auditable.
The custom gfx942 MoE W1/W2 kernels live in `../kernel-dev/hip-a8w4` (not
upstream files, so they have no diffs here); the scheduler and support overlays above
are the vLLM-side changes that call them.

## Regenerating a diff

```bash
curl -L -o base.py \
  https://raw.githubusercontent.com/vllm-project/vllm/<sha>/vllm/...
diff -u --label "a/<upstream path>" --label "b/<overlay>" base.py <overlay>.py
```

> The overlays are the source of truth. The diffs are documentation: the pinned
> image that ran in production is a vLLM ROCm nightly
> (`0.26.1rc1.dev229+g124154a88.rocm723`), which may differ slightly from any
> single upstream revision.

## Licensing

Overlays derived from vLLM carry vLLM's Apache-2.0 headers. The AITER-derived
overlay (`aiter_pa_mqa_logits.i64.py`) carries AITER's MIT header. See
`../LICENSE` and the individual file headers.
