# DeepSeek V4 Flash on a single AMD MI300X

This repository adapts the validated single-MI300X stack from [`ryanzhou/deepseek-v4-flash-mi300x`](https://github.com/ryanzhou/deepseek-v4-flash-mi300x) for [`lovesenko/DeepSeek-V4-Flash-0731-Abliterated`](https://huggingface.co/lovesenko/DeepSeek-V4-Flash-0731-Abliterated). The abliterated checkpoint is a drop-in weight replacement for DeepSeek-V4-Flash-0731; the serving overlays and MI300X kernels remain from the reference stack. Model weights are downloaded separately and are never committed here.

Acknowledgement: thank you to Ryan Zhou for the upstream MI300X serving reference that this adaptation builds on. The abliterated-checkpoint adaptation, measurements, and documentation in this repository are maintained by **arrogance231**.

The default checkpoint revision is `61ec100749f5f05cd268296c5e2eccec03268e78`. The default serving profile is 393,216 tokens, matching the reference's validated single-card profile; set `MAX_MODEL_LEN` explicitly to select another limit.

The table below is measured on the **abliterated checkpoint** on this MI300X
(normal `/v1/chat/completions` traffic, three 512-token streamed requests per
K setting, warmed server). It is not copied from the upstream reference
project. Full methodology and the K-sweep data are in
[`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md).

Phase 3A added a reversible target-generated calibration pilot for the tiny
DSpark Markov head. It improved an offline held-out transition loss, but the
live sidecar was slower (114.52 vs. 119.06 warm tok/s) and had lower accepted
drafts (27.38% vs. 29.24%) on the same K=5 chat fixture. It is rejected and is
not the production checkpoint; see the [Phase 3 findings](docs/ABLITERATED_FINDINGS.md#phase-3--target-generated-markov-head-calibration-completed-rejected).
The same fixture without speculative decoding averaged 68.38 warm tok/s;
`DISABLE_DSPARK=1` is available for a control restart, while the default
`DISABLE_DSPARK=0` keeps DSpark enabled.

Phase 4 quality gating found clean 2K and forced-5K prose, valid JSON/tool
calls, but severe repetition when a single response is forced to 10K tokens;
see [`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md#phase-4--quality-and-stability-gate-completed).

The Phase 5 promoted profile is frozen in
[`configs/production-k5.env`](configs/production-k5.env), with a health and
configuration check at [`scripts/check_production_profile.sh`](scripts/check_production_profile.sh)
and operational guidance in [`docs/PRODUCTION_PROFILE.md`](docs/PRODUCTION_PROFILE.md).
Use approximately 5K completion tokens per prose request; forced 10K output
is known to enter a repetition loop.

| Metric | Result |
| --- | ---: |
| Uncached C1 prefill | **~9.94K–11.40K tok/s** (14K–121K-token probes) |
| Single-stream decode, static DSpark-K=5 | **115.6 tok/s mean**, 117.0 median (normal chat) |
| Single-stream decode, static DSpark-K=6 | 113.9 tok/s mean, 112.4 median (normal chat) |
| Single-stream decode, static DSpark-K=7 | 107.1 tok/s mean, 106.6 median (normal chat) |
| 8-stream decode, static K=7 | 462.1 tok/s aggregate, 61.7 tok/s median/stream |
| Context configured | 393,216 tokens (1M checkpoint ceiling not validated here) |
| KV tier | 16 GB `fp8_ds_mla` GPU pool + 96 GiB native CPU tier |
| Weights in HBM | 156.47 GiB — **no weight offload** |

The current default is **K=5**, the fastest stable setting in this matched
normal-chat sweep. Override it for an A/B run with
`DS_NUM_SPECULATIVE_TOKENS=6` or `=7`; values below the checkpoint's declared
DSpark block size of five are not supported by this ROCm path.

For comparison only, the upstream original-checkpoint project reports 11.69K
prefill tok/s and 152.6 tok/s single-stream K=7 on a synthetic workload. Those
numbers remain reference context, not results for these abliterated weights.

With the abliterated checkpoint at the pinned revision above, the adapted
stack was smoke-tested on this MI300X on 2026-08-27: model load succeeded,
the OpenAI-compatible API stayed healthy, and a 512-token streamed chat
request measured **100.91 decode tok/s** with **0.24 s TTFT**. Repeated
follow-up runs measured approximately **100.5–106.2 tok/s** with probabilistic
DSpark-7; greedy DSpark was slower at approximately 89–93 tok/s. These are
checkpoint-specific engineering measurements, not a claim that the
abliterated model reproduces the reference's 152.6 tok/s. See
[`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md) for methodology,
limitations, and the unanswered quality questions.

## What this repository establishes—and what it does not

This repository establishes a reproducible single-MI300X serving stack for
the abliterated checkpoint, including the ROCm correctness overlays, tuned
`gfx942` kernels, static DSpark (K=5 default; K=5–7 tested), prefix caching, chunked prefill, paged KV, and the
hybrid GPU/CPU KV tier. It also records the measured throughput after the
weight swap.

It does **not** claim that the abliterated checkpoint has the reference
checkpoint's DSpark acceptance rate, that 1M context is production-quality,
or that long-form narrative quality is unchanged. Those require matched
checkpoint A/B tests and long-form evaluation; they cannot be inferred from
the reference repository's synthetic tok/s table. The exact findings and
next experiments are in [`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md).

The first draft-repair experiment is complete: restoring the original base
MTP tensors created a valid loadable variant, but its apparent raw-completion
speedup came with prompt-format leakage and repetition; normal chat returned
to roughly 101–107 tok/s. That candidate was rejected and is not the active
server. The staged repair plan and evidence are in
[`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md#phase-plan-and-phase-1-result).

The official vLLM recipe targets NVIDIA and newer AMD hardware. Running the model reliably on MI300X required fixes for its FP8 format, MoE routing at high concurrency, the checkpoint's expert-activation clamps, causal speculative verification, CPU-KV synchronization, and a long campaign of prefill and decode kernel tuning. This repository collects those fixes, pins the versions used in production, and documents the tuning journey in dated reports (see [Tuning reports](#tuning-reports)).

---

## Why MI300X

The MI300X has **192 GB of HBM3** and 5.3 TB/s of memory bandwidth, with 2.4× the HBM capacity of an H100 SXM5 ([AMD](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)). [Doubleword's write-up](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/) estimates that it costs roughly half as much at list price. For this 304B-parameter checkpoint, the memory capacity allows a simple single-GPU deployment:

- The entire model fits in HBM without PCIe weight streaming or layer offload.
- There is room for a 16 GB GPU KV pool and a 96 GiB CPU tier for evicted prefix-cache entries.
- One card handles 2–8 typical concurrent streams and bursts of up to 64 streams.

MI300X (CDNA3) implements the AMD/Graphcore `fnuz` variant of E4M3, while MI325X and newer use OCP-standard FP8 ([background](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/)). A kernel that assumes OCP semantics on MI300X can be wrong by a factor of two in the scale domain. Correctness on this FP8 implementation was the first priority; performance tuning came afterward.

## Prior art, and what this repo adds

[Fergus Finn's MI300X worklog](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/) and the accompanying [Doubleword repository](https://github.com/doublewordai/vllm-amd-blog-doubleword) identified the FP8 incompatibility, missing AITER fast paths on `gfx942`, HIP-graph hazards in sparse MLA decode, and MoE routing bugs. The [official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) covers NVIDIA hardware and newer AMD GPUs (MI325X at 4K context and MI355X), but not a single-MI300X production configuration for the 0731 checkpoint.

This repository adds:

1. **Correctness overlays** for the pinned ROCm nightly, including fixes not yet in upstream vLLM: the MXFP4 padded-lane routing fix, FNUZ FP8 indexer bytes, 64-bit paged-MQA offsets, deterministic sparse top-k, causal DSpark verification, and the restored DeepSeek expert-activation clamps.
2. **A validated serving configuration** with probabilistic DSpark drafting, block rejection, and static K=5 (K=5–7 were measured; K=5 is the current normal-chat winner). A contention-aware scheduler gives a lone prefill the full 3,712-token quantum but caps long prefills at 1,024 tokens when other requests could be delayed.
3. **Custom gfx942 kernels** (JIT-compiled at first start): row-asymmetric INT8 MoE W1/W2 with adaptive BM16/BM64+BM48 tiles and N-split low-concurrency variants, an exact BF16 SwiGLU+clamp kernel, an OPUS sparse-prefill kernel, and AITER GEMM tuning tables for the recurring `gfx942` shapes the packaged tables were missing.
4. **A hybrid KV strategy**: 16 GB of `fp8_ds_mla` GPU cache + 96 GiB native CPU offload, with a load-path fencing fix that upstream [issue #47282](https://github.com/vllm-project/vllm/issues/47282) documents but [PR #47291](https://github.com/vllm-project/vllm/pull/47291) never merged.

## Repository layout

```text
.
├── compose.yaml         # The production stack (vLLM ROCm + Caddy), digest-pinned
├── Caddyfile.example   # Copy to Caddyfile; set hostname, email, and source CIDR
├── vllm-entrypoint.sh  # Cleans stale CPU-KV mmaps, stages the OPUS module
├── prepare-artifacts.sh # Expands the compiled libtorch extension (SHA-256 verified)
├── SHA256SUMS          # SHA-256 pins for every runtime artifact
├── artifacts/          # Compressed validated stable-libtorch top-k extension
├── patches/
│   ├── *.py           # Byte-for-byte production overlays (mounted read-only)
│   ├── *.cu, *.patch # Top-k extension source and its diff vs. the pinned image
│   ├── diffs/*.patch # Unified diffs vs. the upstream base revision
│   └── README.md     # Provenance and regeneration instructions
├── kernel-dev/hip-a8w4/  # gfx942 MoE/OPUS/SwiGLU HIP sources JIT-built at start
├── tuning/
│   └── *.csv        # AITER A8W8 blockscale tuning tables for gfx942
└── *.md             # Dated tuning and correctness reports (see below)
```

## Runtime configuration

The stack uses a digest-pinned official vLLM ROCm nightly with:

- `--trust-remote-code` and the DeepSeek V4 tokenizer, reasoning, and tool parsers
- `fp8_ds_mla` KV cache (UE8M0 block-scaled FP8, not generic unscaled FP8) with 256-token blocks, 16 GB GPU pool, and a 96 GiB `native` CPU offload tier
- `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_OPUS_PREFILL=1`, and `--moe-backend triton`; AITER handles attention and dense linears, and the 256-expert/top-6 MoE shape dispatches to the custom gfx942 W1/W2 kernels with grouped Triton OGS as fallback
- static DSpark speculative decoding with probabilistic drafting and block rejection (K=5 default; set `DS_NUM_SPECULATIVE_TOKENS` for K=6/7 A/B runs)
- A 4,096-token scheduler budget (384 tokens reserved for DSpark, so ordinary prefills use up to 3,712) with the contention-aware long-prefill cap
- full/breakable CUDA graph capture through M=3,712, giving one graph launch per token during steady decode
- Caddy as an IP-allowlisted HTTPS proxy

## Deploying it

### 1. Host prerequisites

One MI300X (`gfx942`, 304 CUs, ~192 GiB HBM), a working AMD kernel driver, recent Docker Compose, ~235 GiB RAM for the CPU KV tier, and ~500 GB disk (the model cache alone is ~156 GB).

### 2. Pull the pinned runtime and model

```bash
VLLM_IMAGE='vllm/vllm-openai-rocm@sha256:e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a'
MODEL='lovesenko/DeepSeek-V4-Flash-0731-Abliterated'
REVISION='61ec100749f5f05cd268296c5e2eccec03268e78'

docker pull "$VLLM_IMAGE"
MODEL_DIR=/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated \
  VLLM_IMAGE="$VLLM_IMAGE" MODEL_ID="$MODEL" MODEL_REVISION="$REVISION" \
  ./scripts/download_model.sh
```

### 3. Prepare the files

```bash
cp Caddyfile.example Caddyfile   # optional: only needed for `--profile proxy`
mkdir -p aiter-cache crash-dumps profiles-current
chmod +x vllm-entrypoint.sh prepare-artifacts.sh
./prepare-artifacts.sh           # expands patches/_C_stable_libtorch.*.abi3.so
sha256sum -c SHA256SUMS        # verify every artifact before first start
```

### 4. Start

```bash
docker compose config -q
MODEL_DIR=/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated docker compose up -d inference
docker compose logs -f inference
```

The first start JIT-compiles the gfx942 kernels and captures the graph set, so allow about ten minutes. A healthy start must show all of:

```text
Model loading took 156.47 GiB
DSpark draft model loaded: 96 params
GPU KV cache size: 1,283,701 tokens
Maximum concurrency for 393,216 tokens per request: 3.26x
Created mmap file /dev/shm/vllm_offload_...mmap (103.08 GB)
Capturing CUDA graphs (FULL)
Graph capturing finished ... took 6.47 GiB
Application startup complete
```

After graph capture, run `rocm-smi --showmeminfo vram`. The validated high-water is ~199.9 GB of 205.8 GB after long-context and C64 gates; if only a few hundred MB remain, the server may start but fail on the first request.

### 5. Smoke-test

```bash
HOST='your-host.example.com'
curl -fsS "https://$HOST/v1/models"
curl -sS "https://$HOST/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"lovesenko/DeepSeek-V4-Flash-0731-Abliterated\",
       \"prompt\": \"Calculate 17 * 23. Answer with the number only.\",
       \"temperature\": 0, \"max_tokens\": 32}"
```

## The patches

Each `patches/*.py` file is a **full-file overlay** mounted read-only over its counterpart in the container; `compose.yaml` contains the target paths. The corresponding `diffs/*.patch` records the change from its upstream base. The custom kernels in `kernel-dev/hip-a8w4` are not upstream files: `gpt_oss_triton_kernels_moe.row-i8asym-candidate.py` JIT-builds them into `/opt/cj-moe` on first start. The base image remains digest-pinned, so upgrades require changing the image reference and revalidating the stack.

| Overlay | Mounted over | Fixes | Needed when |
| --- | --- | --- | --- |
| `gpt_oss_triton_kernels_moe.row-i8asym-candidate.py` | `vllm/.../fused_moe/experts/gpt_oss_triton_kernels_moe.py` | MXFP4 padded-lane routing fix + row-asymmetric INT8 activations + dispatch to the custom gfx942 W1/W2 kernels | **Required** for the MoE path; the mask fix is [not yet upstream](https://github.com/doublewordai/vllm-amd-blog-doubleword/commit/c32932bb9ff6ad30b942e4835dd8b41601e7569e) |
| `mxfp4.fused-silu.py` | `vllm/.../fused_moe/oracle/mxfp4.py` | Gate/up interleave layout for the fused-SiLU path | Required with the fused-SiLU overlay |
| `triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py` | `vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py` | `gfx942` MXFP4 OGS tile geometry (up to 1,536 routed rows) | **Performance** on `gfx942`; stock geometry slows sharply above 768 routed rows |
| `fused_compress_quant_cache.fnuz-shuffle.py` | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | **FNUZ FP8 + 16×16 preshuffle** in the Lightning Indexer cache writer | **Required on MI300X**; MI325X/MI355X use OCP FP8 and must keep the stock bytes |
| `aiter_pa_mqa_logits.i64.py` | `aiter/ops/triton/gluon/pa_mqa_logits.py` | 64-bit offsets in the `ChunkK=256` paged-MQA kernels | Required when KV offsets can exceed 4 GiB |
| `rocm_aiter_mla_sparse.decode-h32-k16.py` | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | Decode tile 32 heads × 16 KV + canonical top-512 sort + OPUS prefill hook | **Required** (determinism) and **performance** |
| `rocm_aiter_mla.dspark-causal.py` | `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | Causal multi-token speculative verification | Required for DSpark on ROCm small-head MLA — now [upstream](https://github.com/vllm-project/vllm/commit/77469c9057bec3212a64877dbbf3b9c48c22d786); the overlay is the upstream file verbatim |
| `dspark-speculator.independent-draft-gumbel.py` + `spec-decode-utils.independent-draft-gumbel.py` | `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` + `.../spec_decode/utils.py` | Draft-proposal Gumbel noise salted away from rejection/recovery noise | Required only with `draft_sample_method=probabilistic` |
| `kv_offload_cpu_gpu_worker.load-war.py` | `vllm/v1/kv_offload/cpu/gpu_worker.py` | Fence CPU→GPU KV restores behind in-flight compute ([#47282](https://github.com/vllm-project/vllm/issues/47282), [PR #47291](https://github.com/vllm-project/vllm/pull/47291)) | Required only with `--kv-offloading-backend native` |
| `activation.rocm-exact-swiglu.py` | `vllm/model_executor/layers/activation.py` | Exact BF16 SwiGLU with `gate=min(gate,10)`, `up=clamp(up,-10,10)` via `swiglu_clamp.hip` | **Required** for checkpoint-faithful shared-expert output |
| `scheduler.contention-aware.py` | `vllm/v1/core/sched/scheduler.py` | Full 3,712-token prefill quantum only when no other request can be delayed; 1,024-token chunks under contention | **Performance**; enables the 4,096-token budget without latency regressions |
| `block_table.active-width-copy.py` | `vllm/v1/worker/block_table.py` | Copy only active block-table columns each decode step | **Performance** (decode; ~0.76 ms and 40 MB saved per M64 graph) |
| `deepseek_v4_amd_model.router-bf16.py` | `vllm/models/deepseek_v4/amd/model.py` | Keep router logits in BF16 (removes 43 FP32 round-trips) | **Performance** (decode) |
| `deepseek_v4_attention.wqb-bpreshuffle.py` + `deepseek_v4_rocm.wqb-bpreshuffle.py` | `vllm/models/deepseek_v4/attention.py` + `.../amd/rocm.py` | Preshuffle `wq_b` once at load instead of per request | **Performance** (prefill; ~9.5 ms per request) |
| `cache_utils.gather2048.py` | `vllm/models/deepseek_v4/common/ops/cache_utils.py` | `BLOCK_K=2048` K-gather and global top-k index preparation | **Performance** (prefill) |

Two further artifacts back the deterministic top-k path: `sampler.topk-tiebreak-sanitize.cu` (with `vllm-124154a-topk-tiebreak.patch`, its diff against the pinned image's `csrc/libtorch_stable/sampler.cu`) is the source of the compiled `_C_stable_libtorch` extension that `prepare-artifacts.sh` expands and mounts. `rocm_aiter_mla_sparse.topk-tiebreak.py` is a superseded pre-extension variant retained for audit.

### Three important correctness fixes

**MXFP4 routing.** The MoE bitmatrix kernel pads its block columns to a Triton block size, but the padding lanes were masked against the global tensor bound instead of the logical block size. Under load, padded lanes corrupted the routing matrix, causing near-match tool names and forgotten schemas on long prompts. The one-line fix is `mask = (offs_local < BLOCK_SIZE) & (offs_global < nonzero_indx_size)`, taken from [Doubleword commit `c32932bb9`](https://github.com/doublewordai/vllm-amd-blog-doubleword/commit/c32932bb9ff6ad30b942e4835dd8b41601e7569e).

**FP8 format.** DeepSeek V4's Lightning Indexer cache uses FP8. The stock writer emits OCP E4M3 bytes in row-major order, while AITER on MI300X consumes AMD FNUZ E4M3 bytes in a preshuffled 16×16 tile layout. In the worst case, interpreting one format as the other produces a factor-of-two scale error. The overlay selects `float8e4b8` with `FP8_MAX=224.0` and shuffled write offsets on ROCm, while leaving the OCP path unchanged elsewhere.

**Expert activation clamps.** The checkpoint requires `gate=min(gate, 10)` and `up=clamp(up, -10, 10)` before the expert SwiGLU multiply (`swiglu_limit=10`). The first custom W1 kernel omitted both clamps; outlier activations changed logits and caused recurring `)Skip` tokens, rare unrelated CJK, and code-token errors. The W1 kernel and the shared-expert SwiGLU kernel now apply the clamps immediately before SiLU. A 61,440-token raw-completions regression (120 seeds × 512 tokens, native and DSpark-7) shows **0 `)Skip` and 0 stray CJK**; the previous kernel produced them in 3/120 responses. Full repro in [`CORRECTNESS-20260815.md`](CORRECTNESS-20260815.md).

### Speculative decoding

This stack uses probabilistic drafting with block rejection. The two Gumbel overlays keep draft-proposal noise independent of rejection and recovery noise. The tested static range is K=5–7; K=5 is the current default because it won the matched normal-chat sweep. The checkpoint declares `dspark_block_size: 5`, so values below five are an unsupported Markov-head layout that can produce garbled output.

## Performance

### Prefill

Uncached 8.9K-token prompts: the original deployment did **5.26K tok/s** at C1; the current profile does **11.69K tok/s** steady (11.53K median, **2.19×**). Milestones: contention-aware scheduler 6.96K → custom HIP MoE ~7.9K → attention/support stack 8.30K → asymmetric row-INT8 W1 8.99K → OPUS prefill ~9.36K → BM64/delta-scale W1 9.57K → batch 4,096 + M=3,712 tuning 10.97K → exact W2 11.09K → graph buckets + A8W8 ASM 11.24K → OPUS no-padding 11.39K → deterministic top-k 11.53K. Full chronology and rejected paths: [`PREFILL-EXPERIMENT-LOG-20260808-09.md`](PREFILL-EXPERIMENT-LOG-20260808-09.md), summary in [`PREFILL-OPTIMIZATION-20260809.md`](PREFILL-OPTIMIZATION-20260809.md).

| Streams | Effective prefill tok/s |
| ---: | ---: |
| 1 | 11.66K |
| 2 | 10.19K |
| 4 | 11.20K |
| 8 | 11.36K |

### Decode

Three decode rounds (reports: [`DECODE-OPTIMIZATION-20260812.md`](DECODE-OPTIMIZATION-20260812.md), [`DECODE-OPTIMIZATION-20260814.md`](DECODE-OPTIMIZATION-20260814.md), [`DECODE-OPTIMIZATION-20260815.md`](DECODE-OPTIMIZATION-20260815.md)) targeted the launch-bound low-concurrency regime. The retained work: adaptive BM16/BM64+BM48 MoE tiles ([`MOE-REWRITE-20260812.md`](MOE-REWRITE-20260812.md)), exact decode support fusions and the stable top-k extension (M64 graph 32.62 → 27.76 ms, −508 GPU operations), N-split W1/W2 kernels and decode-M GEMM rows for M≤8 (native C1 +10%), and mid-M GEMM rows for the C8+ verify and drafter range. Final corrected medians:

| Concurrency | Native aggregate tok/s | Native tok/s/user | K7 aggregate tok/s | K7 tok/s/user | K7 accepted/draft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.28 | 68.31 | 152.56 | 158.75 | 2.167 |
| 2 | 123.48 | 63.45 | 207.00 | 132.86 | 1.703 |
| 4 | 223.32 | 58.33 | 327.72 | 95.22 | 1.532 |
| 8 | 393.02 | 53.83 | 510.46 | 79.80 | 1.558 |
| 16 | 571.37 | 46.78 | 728.01 | 53.77 | 1.530 |
| 32 | 1,079.08 | 37.91 | 975.62 | 36.98 | 1.485 |
| 64 | 1,649.80 | 29.64 | 1,278.23 | 25.14 | 1.563 |

These use a synthetic random-word workload whose acceptance is lower than production traffic; treat them as gates for this exact image, not universal model benchmarks.

For the abliterated checkpoint specifically, a matched live chat run (512
requested tokens per stream, probabilistic DSpark-7, warmed server) measured:

| Streams | Aggregate tok/s | Median tok/s per stream | Median TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 102.8 | 103.9 | 0.052 s |
| 2 | 147.4 | 87.7 | 0.997 s |
| 4 | 266.5 | 76.0 | 0.494 s |
| 8 | 462.1 | 61.7 | 0.173 s |

These are checkpoint-specific client measurements, not replacements for the
upstream synthetic table. Full methodology and acceptance deltas are in
[`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md#multi-concurrency-result-on-the-active-checkpoint).

### Mixed load and context

A 4,096-token budget with the contention-aware cap keeps cold prefills from stalling other streams: a ~52K cold prefill behind live decodes completes with a late-short-request TTFT of ~0.3 s and a maximum background decode gap of ~0.16 s. The stack serves 384K requests: 379K-token cold recalls complete in ~51–53 s (native) or 120–125 s (DSpark), warm recalls hit 379,904 cached tokens in 0.64–2.65 s with byte-identical output, and a 393,051-total-token request (165 below the limit) recalled all needles exactly. The 16 GB pool + 96 GiB tier reports a 1,945,846-token length-equivalent metric.

## Production notes

- **HBM headroom is limited.** The validated high-water after 380K recall and C64 gates is ~199.9 of 205.8 GB (5.9 GB free). A 30 GB KV pool loads but fails during graph capture with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Do not change the pool size without repeating the memory gates.
- **Run `prepare-artifacts.sh` before every start.** It expands the compiled top-k extension and verifies its SHA-256; `sha256sum -c SHA256SUMS` verifies everything else. The first start JIT-compiles the gfx942 kernels, so cold recovery takes ~10 minutes; keep health checks tolerant of that window.
- **AITER fallback messages are informational.** `shape ... not found tuned config ... will use default config` is benign: arbitrary prompt lengths create shapes outside the tables. `HSA_STATUS_ERROR`, OOM, tracebacks, or HTTP 5xx are not.
- **The CPU tier is an opportunistic cache**, not scheduler capacity. The load-path fencing overlay must stay mounted; an unfenced restore can overwrite KV still read by in-flight compute.
- **Keep raw-completions tests** because they isolate serving from chat encoding; the raw `/v1/completions` gate is what caught the activation-clamp bug.

## Tuning reports

| Report | Contents |
| --- | --- |
| [`PREFILL-OPTIMIZATION-20260809.md`](PREFILL-OPTIMIZATION-20260809.md) | Final prefill profile, milestone medians, deterministic top-k, rejected paths |
| [`PREFILL-EXPERIMENT-LOG-20260808-09.md`](PREFILL-EXPERIMENT-LOG-20260808-09.md) | Complete 5.26K→11.53K chronology with controlled A/Bs |
| [`DECODE-OPTIMIZATION-20260812.md`](DECODE-OPTIMIZATION-20260812.md) | Decode round 1: sparse-attention tile, exact decode GEMM rows |
| [`MOE-REWRITE-20260812.md`](MOE-REWRITE-20260812.md) | Adaptive BM16/BM64+BM48 MoE kernels and rejected architectures |
| [`DECODE-OPTIMIZATION-20260814.md`](DECODE-OPTIMIZATION-20260814.md) | Decode round 2: fixed-graph ledger, exact support fusions (32.62→27.76 ms) |
| [`DECODE-OPTIMIZATION-20260815.md`](DECODE-OPTIMIZATION-20260815.md) | Decode round 3: bit-exact low-concurrency N-split kernels and mid-M GEMM rows |
| [`CORRECTNESS-20260815.md`](CORRECTNESS-20260815.md) | Expert-activation clamp bug: root cause, repro, corrected gates |
