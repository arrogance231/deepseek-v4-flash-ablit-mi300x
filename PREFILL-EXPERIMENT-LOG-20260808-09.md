# MI300X prefill experiment log, 2026-08-08 to 2026-08-09

This is the detailed history behind the final
[prefill production report](PREFILL-OPTIMIZATION-20260809.md). It was
reconstructed from the two complete Codex rollouts that performed the work
(44 compaction checkpoints) and cross-checked against the benchmark, profiler,
quality, and rollback artifacts retained on the production host under
`/opt/deepseek-v4/optimization-20260808`.

The final production files in this repository remain authoritative. This log
explains how the deployment moved from **5,260.2 to 11,527.5 uncached C1
prefill tok/s**, what each retained change actually did, and why many seemingly
promising alternatives were rejected.

## Reading the numbers correctly

The main benchmark submitted distinct, uncached prompts made from roughly
6,000 input words. Tokenization produced about 8.6K to 8.9K prompt tokens.
Rates are prompt tokens divided by time to first token. Cold JIT/repack runs
were discarded and repeated measurements are reported as medians.

Three qualifications matter:

1. The milestone rates below are a chronology, **not an additive waterfall**.
   Prompt text, routing distributions, restart temperature, and in a few cases
   prompt token count differ between milestones.
2. A percentage is called a controlled A/B only when both sides used matched
   prompts and serving settings. Operator timings explain mechanism; they do
   not establish an endpoint gain by themselves.
3. The production scheduler reserves 384 of the configured 4,096 scheduled
   tokens for DSpark K=7. The largest ordinary prefill dispatch is therefore
   M=3,712, not M=4,096.

## How 5.26K became 11.53K

| Chronological milestone | C1 median | What the measurement establishes |
| --- | ---: | --- |
| Original 2,048-token production baseline | **5,260.2** | Five fresh uncached runs; the 1,024-token latency cap throttled even a lone prefill. |
| Exact M=1,024 AITER tables | **5,365.3** | 13–64% operator wins translated to only about 2% wall gain. |
| Contention-aware scheduler | **6,957.6** | **+32.3%** from the original baseline by allowing an uncontended request to use M=1,664. |
| First integrated custom HIP MoE | **about 7,900** | Replaced the stock routed OGS W1/W2 pipeline; the first JIT run was excluded. |
| Graph, FlyDSL, gather, direct-output, and RoPE stack | **8,298.4** | Removed support-kernel and launch overhead around the new MoE path. |
| Asymmetric row-INT8 W1 | **8,987.9** | Approximately 8% endpoint improvement; approximate versus the former two-term FP8 W1. |
| Initial gfx942 OPUS port | **about 9,360** | Controlled endpoint baseline was about 8,972 and candidate about 9,354–9,360. |
| BM64/delta-scale W1 | **9,568.6** | Paired baseline 9,450.8; W1 trace time fell 192.862 to 172.782 ms/request. |
| Faster direct-LDS candidate | **9,749.9** | Included a W2 split-accumulator variant later rejected on deterministic fixtures. |
| Exact post-W2-revert baseline | **9,457** | Clean comparison point for the later batch-size work. |
| Batch 4,096, before exact M=3,712 tuning | **10,757.0** | **+13.7%** over the exact post-revert baseline; the largest late-stage gain. |
| Batch 4,096 plus exact M=3,712 tuning | **10,972.2** | **+16.0%** over the same exact baseline; DSpark reservation remained intact. |
| Exact W2 preparation/reduction changes | **11,084.8** | Another controlled 1.0% endpoint gain. |
| Exact graph buckets plus A8W8 ASM rows | **11,239.2** | Preserved C2 by avoiding M=2,048 work being padded to M=3,712. |
| OPUS no-padding | **11,391.2** | Paired baseline 11,246.6; **+1.29%**, with every matched prompt faster. |
| Deterministic HIP top-k, final restart | **11,527.5** | Three runs were 11,622.3, 11,513.1, and 11,527.5 tok/s. |

The defensible overall statement is therefore **2.19x / +119%** from the
original five-run median to the final three-run post-restart median. The table
also explains why summing every local percentage would be misleading.

## Phase 1: establish the real bottleneck

### Baseline and scheduler geometry

The starting profile used:

- `max_num_batched_tokens=2048`
- 384 scheduled-token slots reserved by static DSpark K=7
- an ordinary M=1,664 ceiling
- `long_prefill_token_threshold=1024`
- `max_model_len=393216` and `max_num_seqs=64`

The long-prefill cap had originally been introduced for good reason: it kept a
large cold prompt from starving live decodes and newly arriving short prompts.
The implementation, however, applied the cap even when the large prefill was
the only request in the system. That made the low single-stream rate partly a
scheduler artifact rather than a GPU limit.

Profiles at M=1,664 also showed why table-only tuning could not provide the
requested breakthrough. A representative step spent roughly:

| Family | Time per step before the custom stack |
| --- | ---: |
| Routed MoE W1 and SiLU | 63.8 ms |
| Routed MoE W2 | 29.2 ms |
| Sparse attention | 30.7 ms |
| Dense GEMMs | about 43.6 ms |
| Routing, MHC, normalization, and support | the remaining 50–60 ms |

The GPU was already continuously busy. Prefill was not waiting on HTTP,
tokenization, or a large host-side idle bubble.

### Exact M=1,024 A8W8 tuning

Runtime logs exposed repeatedly untuned A8W8 shapes whenever the latency cap
forced M=1,024. CK-only tuning used `splitK=0` and required `errRatio=0`.

- Preshuffled rows improved **13.25%, 16.70%, and 15.97%**.
- Standard rows improved **43.03%, 58.21%, 59.76%, and 64.29%**.
- End to end, C1 moved only **5,260.2 to 5,365.3 tok/s**.

This was an important diagnostic result: generic A8W8 fallbacks were real, but
not the principal C1 ceiling. The rows were retained because M=1,024 remains
the production shape during contention.

Evidence:

```text
optimization-20260808/baseline-prefill-c1.jsonl
optimization-20260808/m1024-tables-prefill-c1.jsonl
optimization-20260808/prefill-m1024-*-tune.log
```

### Contention-aware long-prefill cap

The scheduler patch asks whether any other running, waiting, or skipped request
needs protection. A lone prefill receives the full ordinary quantum; as soon as
there is contention, the 1,024-token cap returns.

This moved the five-run C1 median from **5,260.2 to 6,957.6 tok/s**. It also
improved, rather than merely preserving, the historical mixed scenario:

| Metric | Old production | Contention-aware |
| --- | ---: | ---: |
| 51.9K cold-prefill time | 12.687 s | 11.302 s |
| Late short-request TTFT | 0.750 s | 0.553 s |
| Maximum background decode gap | 0.460 s | 0.282 s |

The production implementation is
[`patches/scheduler.contention-aware.py`](patches/scheduler.contention-aware.py).
Evidence is in `contention-aware-*.jsonl` and
`contention-aware-interference.json` on the server.

Two nearby ideas were rejected:

- Dynamic DSpark K=7 to K=5 did not produce a reliable throughput win and made
  cached TTFT worse.
- A WPE1 sparse-attention flag produced no integrated gain and was reverted.

## Phase 2: replace the routed MoE pipeline

### Why a custom kernel was necessary

The stock Triton OGS path at M=1,664 was approximately **1.61 ms for W1** and
**0.77 ms for W2** per layer in representative measurements. Routed MoE was the
largest profile family.

AITER shipped a promising A8W4 Triton path, but the pinned Triton 3.6 compiler
could not lower scaled E4M3 x E2M1 dot operations on gfx942. Current Triton main
compiled more of the path but did not produce a competitive kernel. Manual
Triton expansion prototypes measured **10.46–12.09 ms**, versus roughly
**1.91 ms** for the existing combined baseline. This closed the high-level
Triton route and motivated handwritten HIP.

### First custom HIP design

The integrated kernel pipeline used:

- an offline MFMA-lane weight repack, replacing the old packed storage in place
- GPU generation of expert-routing schedules
- packed 128-bit weight loads directly from global memory to VGPRs
- double-buffered activation tiles in LDS; weights never occupied LDS
- E8M0 scale handling folded into the packed-value lookup
- fused W1 gate/up projection and SiLU multiplication
- fused W2 gate, scatter, and top-6 reduction
- a fused BF16-to-two-FP8 input quantizer

At M=1,664 the first integrated implementation measured:

| Component | Time per layer |
| --- | ---: |
| Hidden quantization | 0.008 ms, down from about 0.133 ms |
| W1 plus SiLU | about 0.820 ms |
| W2 plus gate/scatter/reduction | about 0.545 ms |
| Full scheduled pipeline | **1.372 ms** |

The endpoint moved from roughly 6.85–6.96K to roughly **7.9K tok/s**. A real
layer-0 comparison reported 0.1223% relative-L2 at the intermediate and 0.2568%
at the final MoE output, close to the approximation already used by the former
two-term activation path.

### Exact and rejected low-level refinements

Several exact refinements survived operator testing:

- skipping empty 16-row MFMA slabs at ragged expert tails: W1 **4.3–5.8%** and
  W2 **1.1–3.6%** faster across five captured routing histograms
- emitting BM48 and BM64 schedules in one launch: 9.185 to 6.904 microseconds
- loading activations directly from global memory into LDS
- pre-normalizing scale deltas once during weight preparation
- omitting loads for inactive W1 rows

The experiments also established the physical limits of the original design:

- W1 had 34–44% BM48 routing padding and executed two FP8 activation terms.
- The MI300X VF sustained about 1.33–1.46 GHz at its 750 W cap, well below the
  clock implied by headline peak figures.
- Later real W1 counters showed about 3.4 GB fetched per full call and
  2.3–2.6 TB/s effective bandwidth.
- W2 counters showed about 1.79 TB/s and substantial unpack/VALU work.

Consequently, no credible 5x MoE-only headroom existed. Producer/consumer BM96
(1.446 versus about 1.0 ms), direct FP16 W1, alternate wave counts, K=256,
software pipelines, and `s_setprio` all lost.

## Phase 3: remove support and attention overhead

### Piecewise graph capture

Capturing the M=1,664 prefill path reduced host submission from about **190 to
69 ms**. It did not initially improve endpoint throughput because GPU execution
still occupied 181–200 ms per step, but it removed the host as the next ceiling.
This change was deliberately retained while subsequent kernels reduced device
time.

An exact Triton staging screen also changed sparse prefill from the default
pipeline depth to `num_stages=1`. Representative sink/no-sink calls moved from
about 0.792/0.820 to 0.745/0.755 ms, and 12 ragged cases were bit-identical.
The endpoint effect was only about 1%; it remains useful as the fallback for
shapes not dispatched to OPUS.

### FlyDSL FP8 MQA logits

The gfx942 FlyDSL replacement for the sparse indexer's Triton logits kernel
measured:

- **0.559 to 0.217 ms** for a representative full call
- 18.677 to 7.585 ms across five profiled iterations
- relative-L2 `5.8e-8`, with an exact mask
- C1 **8,022.9 to 8,127.8 tok/s** in a controlled endpoint comparison

This was retained as a measured **+1.31%** endpoint win.

### K gather/dequant and direct attention output

Increasing gather workers from 128 to 2,048 made representative gather/dequant
outputs bit-identical and reduced trace time from **3.568 to 0.793 ms/step**.

The old attention path then wrote a roughly 109 MB temporary, copied it, and
inverse-rotated all 512 dimensions although only the trailing 64 rotate. The
replacement writes attention directly to the final tensor and mutates those 64
dimensions in place:

- inverse RoPE: **4.575 to 1.128 ms/step**
- device-to-device copies: **2.278 to 0.046 ms/step**

Together with FlyDSL and graph capture, this support-kernel phase reached an
approximately **8,298 tok/s** endpoint.

### The first gfx942 OPUS port

AITER's hand-scheduled BF16 OPUS sparse-prefill kernel targeted gfx950. The
port to gfx942 required more than changing an architecture guard:

- replace gfx950 scheduling barriers and permlane reductions
- use a 32-key tile, four waves, one LDS buffer, and a sequential gfx942 path
- stage global loads through VGPRs because gfx942 lacks the same 16-byte
  global-to-LDS operation
- emulate the unavailable `ds_read_b64_tr_b16` with aligned LDS reads and
  extraction

For causal/ragged M=1,664, the port measured about **0.485 versus 0.688 ms** for
Triton. Endpoint throughput moved from about **8.97K to 9.36K tok/s**, roughly
4.4%. The OPUS result was numerically close rather than bit-identical to Triton,
so it was promoted only after ragged, source-region, sink, completion, tool,
and long-context gates.

## Phase 4: evolve W1, W2, and dense projections

### Asymmetric row-INT8 W1

Per-K128 and per-K512 INT8 approaches preserved the FP4 weights very accurately
but were slower after quantization and rescaling. A row-wise path was much
faster. The retained asymmetric quantizer computes an affine per-row zero point
and applies the exact correction:

```text
sum(q * w) - zero_point * sum(w)
```

Weight sums are prepared once, and negative FP4 zero is canonicalized during
repacking. Compared with the former two-term FP8 W1:

- full-chain W1 moved from about 0.944 to 0.879 ms/layer
- real-layer W1 relative-L2 ranged from 0.354% to 1.429%
- endpoint moved to an **8,987.9 tok/s** median, about 8% over the preceding
  support stack

This is the largest retained approximation versus the original stack. It was
not accepted on operator error alone: repeated model fixtures, explicit tool
calls, C2–C8, C64, and long-context recall were required.

### BM64, ragged work, and delta-scale preparation

The later W1 kernel combined:

- BM64 scheduling for W1 while retaining BM48 for W2
- ragged 16-row MFMA skipping
- inactive-row activation-load skipping
- a packed magnitude lookup
- scale bytes converted to small deltas once at model load

Across captured histograms it was 12.7–14.9% faster than the production-style
BM48 W1. In a controlled server A/B:

- C1: **9,450.8 to 9,568.6 tok/s** (+1.25%)
- W1 GPU time: **192.862 to 172.782 ms/request** (-10.4%)
- five checkpoint layers were bit-identical to the preceding asymmetric W1
  implementation

This distinction is important: BM64 was exact relative to row-INT8 W1; row-INT8
itself is approximate relative to the original two-term path.

### `wq_b` preshuffle

The recurring `(M,N,K)=(1664,32768,1024)` projection was `wq_b`, but it lacked
the preshuffle hook used by adjacent attention weights. Adding the hook and a
tuned CK row changed a real layer by only 117 BF16 elements out of 54.5 million
(`rel-L2 1.87e-6`). Matched profiles showed:

- normal CK GEMMs: 54.72 ms
- replacement preshuffled CK GEMMs: 45.10 ms
- net saving: about **9.5 ms/request**

Wall measurements were noisy, but the device-time saving was direct and the
change remained in production.

### Direct global-to-LDS activation loading

The final W1 and W2 kernels issue activation loads directly into LDS rather
than staging through VGPRs and explicit LDS writes.

- W1 gained 1.2–2.1% over five real routing histograms and was bit-identical.
- W2 direct-LDS plus B-first prefetching gained about 2.7% and was
  bit-identical.
- A matched trace attributed **4.84 ms/request** of MoE device-time reduction
  to the combined retained change.

A W2 variant with two independent accumulator streams was 4.4–4.8% faster, but
reassociated floating-point sums and shifted two deterministic fixture
continuations. It was reverted despite relative-L2 of only `4e-6` to `3.9e-5`.
The exact, unsplit direct-LDS/B-first path is what production retains.

### Exact W2 cleanup

At M=3,712, three more exact changes were promoted:

1. normalize E8M0 scales once at load time
2. vectorize the fixed-N=4,096 top-6 output reduction eight columns at a time
3. remove the redundant `rows > 0` guard for the first MFMA row group

The scale change reduced W2 VALU instructions by 5.55%; all five checkpoint
layers were bit-identical. After the restart loading all three changes, the C1
median moved **10,972.7 to 11,084.8 tok/s** (+1.0%).

### WO_A token-major output

The inverse-RoPE WO_A BMM produced group-major output which downstream code
immediately cloned into token-major order. Supplying a transposed output view to
`torch.bmm` removed 86 copies without changing BMM arithmetic:

- `aten::copy_` calls: 678 to 592
- copy GPU time: **9.416 to 4.613 ms/request**
- deterministic fixtures: exact

A pretransposed weight saved another roughly 4 ms but selected a different
small-M hipBLASLt reduction path and changed all normally stable continuations,
so it was rejected.

## Phase 5: increase the production batch budget

The exact baseline after reverting W2 accumulator reassociation was about
**9,457 tok/s**. The batch budget was then raised from 2,048 to 4,096 at the
user's suggestion, while retaining all 384 DSpark-reserved slots. This made the
ordinary uncontended quantum M=3,712.

The effect was much larger than another micro-kernel tweak:

| Configuration | C1 median | Relative to exact 2,048 baseline |
| --- | ---: | ---: |
| Exact 2,048-token baseline | 9,457 | — |
| 4,096 configured, untuned | 10,757 | +13.7% |
| 4,096 plus exact M=3,712 tuning | 10,972 | +16.0% |

All 14 tuned shapes passed operator checks. Representative M=3,712 standard
rows improved from 201.31 to 87.98 microseconds, 1,439.61 to 527.77
microseconds, and 282.55 to 107.46 microseconds. Preshuffled rows improved by
roughly 11–17%.

The larger budget did not undo interactivity because the scheduler still uses
M=1,024 under contention:

| Metric | Exact 2,048 baseline | Batch 4,096 |
| --- | ---: | ---: |
| 52K cold-prefill time | 7.887 s | 7.704 s |
| Late short-request TTFT | 0.339 s | 0.352 s |
| Worst decode event gap | 0.194 s | 0.198 s |

Decode also showed no regression: aggregate results included 231.6 tok/s at C2,
512.7 at C8, and 1,087.1 at C64 during this phase.

## Phase 6: graph geometry and final exact-shape work

### Exact graph buckets

Capturing only M=3,712 cut ordinary kernel launches and improved C1, C4, and C8,
but C2 regressed from 9.745K to 8.686K. The graph dispatcher padded its M=2,048
contended work to the next captured bucket, M=3,712.

Adding exact M=2,048 and M=3,072 graph buckets fixed that regression. Exact
M=2,048 CK rows then improved C2 from **9.804K to 10.064K** (+2.65%). Exact
M=1,536 and M=3,072 table rows showed no endpoint gain and were removed.

### A8W8 ASM rows

A full CK, CKTile, ASM, and OPUS search found six preshuffled ASM rows worth
retaining: three each at M=3,712 and M=2,048. Recurring operators improved by
about 4–12%, all with `errRatio=0`. The production sweep then measured:

| Workload | Before ASM | With ASM |
| --- | ---: | ---: |
| C1 | 11.142K | **11.239K** |
| C2 | 10.064K | **10.151K** |
| C4 | 10.965K | **11.270K** |
| C8 | 11.189K | **11.321K** |
| C64 decode | 21.4 tok/s/user | **22.7 tok/s/user** |

### OPUS no-padding

The active OPUS kernel used 33,792 bytes of LDS. A 32-byte padding column made
two workgroups require more than the 65,536-byte LDS limit. Simply deleting the
padding was wrong because the gfx942 transpose emulation encoded the padded
strides. The retained change removed the column and corrected strides from
1,056/2,112/3,168 to 1,024/2,048/3,072.

- LDS: **33,792 to 32,768 bytes**, permitting two resident workgroups
- VGPR: **384 to 247**
- AGPR: **136 to 0**
- operator gain: 8–20%, bit-identical to the existing OPUS port
- controlled C1: **11,246.6 to 11,391.2 tok/s** (+1.29%)
- matched OPUS time: **120.875 to 110.370 ms/request**

### Deterministic HIP top-k

Prefill invoked top-k 63 times in the final trace. The stock HIP selector was
fast but used atomic append order at an exact cutoff tie. Sorting its result
fixed order, not which tied elements had been selected.

The retained vLLM extension changes the insertion comparison to use original
index as the secondary key. Equal scores therefore deterministically select the
lower original index, after which selected indices are canonically sorted for
sparse attention. Tests covered every observed shape, ragged rows, all-equal
inputs, mixed cutoff ties, more than 2,048 equal values, and 20 quantized-tie
seeds.

- profiled top-k device time: **13.29 to 7.90 ms/request**
- median paired TTFT saving: **4.5 ms**
- five of seven paired prompts were faster
- final post-restart median: **11,527.5 tok/s**

PyTorch does not specify which equal-valued element must be returned. The new
policy is mathematically valid and deterministic, but tied sets can differ from
PyTorch's unspecified choice. The user accepted this final change without an
additional A-B-A restart after the standalone, paired, and quality evidence.

## Rejected experiment matrix

These negative results are part of the optimization outcome; they prevent a
future round from repeating expensive dead ends.

| Area | Candidate | Result and decision |
| --- | --- | --- |
| Scheduling | Dynamic DSpark K7 to K5 | No reliable throughput win; cached TTFT worsened. |
| Scheduling | 16K batch | Alternated between about 6.9K and 9.0K and consumed nearly all HBM. |
| Scheduling | M=2,048 with BM64 | W1 microbenchmark improved, but endpoint fell to about 8.22K. |
| Runtime | ROCm multistream overlap | Safe in the tested stage, but consistently slower. |
| Runtime | M=1,536 and M=3,072 exact A8W8 rows | Tuner-local wins did not survive endpoint testing; removed. |
| Runtime | KV-copy descriptor coalescing | Zero of 860 descriptors were physically coalescible. |
| MoE compiler | Manual Triton FP8/MXFP4 | 10.46–12.09 ms versus about 1.91 ms baseline. |
| MoE compiler | Newer Triton/AITER A8W4 | Pinned compiler could not lower it; newer compiled paths were slower. |
| W1 | BM96 producer/consumer | 1.446 versus about 1.0 ms. |
| W1 | FP16/direct global variants | Exact or very close, but slower than the retained path. |
| W1 | Alternate waves, K256, software pipelines, `s_setprio` | Neutral or slower. |
| W1 | Reciprocal/fast-exp SiLU variants | At most about 1%; no repeatable endpoint gain or added numerical risk. |
| W2 | Row FP8 | About 20% faster locally, but 0.75–2.14% layer error and catastrophic repetitive model output. |
| W2 | K128 FP8 | Only about 10% faster with still-material layer error. |
| W2 | Row/K128/K512 INT8 | Slower, too inaccurate, or too small a gain after quantization. |
| W2 | Split accumulators | 4–5% faster but changed deterministic continuations; reverted. |
| W2 | Scale broadcast, paired decode, alternate NF/BM/TK | Coalescing or conversion overhead made them slower. |
| Attention | Max-free/grouped softmax | 14–29% faster, but small layer drift compounded into large logit/text changes. |
| Attention | Split/materialized/tiled attention | Roughly 2x to more than 10x slower. |
| Attention | OPUS paired waves | Large register reduction, but synchronization made it about 0.5% slower and not exact. |
| Attention | OPUS global64/rescale-skip variants | Exact but neutral or 3–6% slower at production geometry. |
| Attention | OPUS low-head/K64/K16 variants | Invalid MFMA layouts, over 64 KiB LDS, incorrect, or slower. |
| MHC | Historical fused post/pre path | Missing BF16 boundary explained drift; after fixing it, 0.2084 versus 0.1826 ms, so slower. |
| Dense | hipBLASLt exhaustive WO_A/MHC search | Default algorithm was already effectively optimal. |
| Dense | CK/Triton/block-FP8 WO_A | Slower and, for FP8, materially less accurate. |
| Dense | Pretransposed WO_A weight | Saved about 4 ms but changed every stable continuation; rejected. |
| Sparse top-k | Raw HIP selector plus sort | Fast, but sorting could not repair the nondeterministic tied set. |

## Correctness and numerical decisions

Not every retained optimization is bit-exact relative to the original
deployment. The exact boundary for each claim is:

| Change | Numerical status |
| --- | --- |
| Scheduler and graph capture | Same model arithmetic; scheduling/order gates passed. |
| A8W8 table rows and ASM rows | Operator `errRatio=0` in the stated comparisons. |
| Gather/dequant, direct attention output, in-place RoPE | Bit-identical in representative and production-shape tests. |
| BM64/ragged/direct-LDS W1 | Bit-identical to the preceding asymmetric row-INT8 W1. |
| Asymmetric row-INT8 W1 | Approximate versus original two-term FP8 W1; checkpoint-layer and full-model gates passed. |
| W2 direct-LDS, scale normalization, reduction, guard removal | Bit-identical in their stated operator/checkpoint comparisons. |
| Initial gfx942 OPUS port | Numerically close, not bit-identical to Triton; full-model gates passed. |
| OPUS no-padding | Bit-identical to the already accepted OPUS port. |
| `wq_b` preshuffle | Tiny BF16 drift, `rel-L2 1.87e-6`. |
| WO_A token-major output | Exact; only the redundant copy/layout changed. |
| Deterministic top-k | Exact top-k mathematics under an explicit lower-index tie policy; PyTorch's tie subset is unspecified. |

Promotion gates included repeated completion fixtures, chat, streaming,
reasoning, automatic tool calls, C2–C8 prefill, C64 decode, mixed prefill/decode
interference, and 379,906-token marker recall. The final pre-top-k gate is
summarized in the production report.

## Evidence and artifact index

### Production artifacts in this repository

| Concern | File |
| --- | --- |
| Scheduler and serving geometry | `compose.yaml`, `patches/scheduler.contention-aware.py` |
| W1/W2 integration | `patches/gpt_oss_triton_kernels_moe.row-i8asym-candidate.py` |
| W1 quant/GEMM | `kernel-dev/hip-a8w4/quanti8asym_dscale.hip`, `fusedi8asym64ragged_activea.hip` |
| W2 and routing | `kernel-dev/hip-a8w4/w2.hip`, `schedule.hip` |
| OPUS no-padding | `kernel-dev/hip-a8w4/opus942/` |
| Sparse support/top-k | `patches/cache_utils.gather2048.py`, `patches/rocm_aiter_mla_sparse.topk-tiebreak.py` |
| `wq_b` preshuffle | `patches/deepseek_v4_attention.wqb-bpreshuffle.py`, `patches/deepseek_v4_rocm.wqb-bpreshuffle.py` |
| Shape-specific A8W8 selection | `tuning/*.batch4096.csv` |
| Deterministic top-k extension | `patches/vllm-124154a-topk-tiebreak.patch`, `artifacts/*.zst` |
| Final bottleneck math | `findings/gap-ledger.md`, `roofline.json`, `gpu-ledger.txt` |

### Server-side experiment evidence

The following directories are intentionally not copied into the production
repository because they contain large traces, transient builds, or rejected
sources. Preserve them on the server when revisiting the work:

```text
/opt/deepseek-v4/optimization-20260808/baseline-prefill-c1.jsonl
/opt/deepseek-v4/optimization-20260808/contention-aware-*
/opt/deepseek-v4/optimization-20260808/attention-stage1-results/
/opt/deepseek-v4/optimization-20260808/flydsl-mqa/
/opt/deepseek-v4/optimization-20260808/gather2048/
/opt/deepseek-v4/optimization-20260808/inplace-rope/
/opt/deepseek-v4/optimization-20260808/row-int8/
/opt/deepseek-v4/optimization-20260808/m2048-bm64/
/opt/deepseek-v4/optimization-20260808/opus942/
/opt/deepseek-v4/optimization-20260808/w1-bm64-*
/opt/deepseek-v4/optimization-20260808/wqb-trace/
/opt/deepseek-v4/optimization-20260808/direct-lds-*
/opt/deepseek-v4/optimization-20260808/post-w2-revert-*
/opt/deepseek-v4/optimization-20260808/batch4096-*
/opt/deepseek-v4/optimization-20260808/final-w2-3712/
/opt/deepseek-v4/optimization-20260808/woa-strided-out-3712/
/opt/deepseek-v4/optimization-20260808/graph3712/
/opt/deepseek-v4/optimization-20260808/graph-prefill-buckets/
/opt/deepseek-v4/optimization-20260808/final-graph-buckets-m2048/
/opt/deepseek-v4/optimization-20260808/a8w8-asm/
/opt/deepseek-v4/optimization-20260808/opus-nopad/
/opt/deepseek-v4/optimization-20260808/topk-tiebreak/
/opt/deepseek-v4/optimization-20260808/gap-ledger/
```

## Final bottleneck closure

The last pre-top-k trace contained 758.566 ms of summed GPU activity and
749.969 ms of activity union. Approximate critical-path contributions were:

| Family | Critical time |
| --- | ---: |
| Other dense GEMMs | 188.5 ms |
| MXFP4 W1 | 137.3 ms |
| MXFP4 W2 | 112.7 ms |
| OPUS sparse attention | 109.3 ms |
| Sparse-indexer support | 42.6 ms |
| MHC | 42.1 ms |
| QKV, RoPE, and cache support | 24.0 ms |
| Quantization | 15.0 ms |
| Copies | 12.7 ms |
| Remaining launches and elementwise work | about 58 ms |

For the 8,594-token reference request, 26.3K tok/s permits only 326.8 ms. The
non-W1/W2/OPUS activity union alone was 387.4 ms. Even granting W1, W2, and
OPUS unattainable hardware-peak execution still leaves about 486.7 ms, or only
17.66K tok/s.

That closes the original “missing prefill headroom” question: the initial
deployment did contain large scheduler, batching, MoE, attention, and support
inefficiencies, and those produced the measured 2.19x gain. The remaining gap
is not another flag or isolated kernel schedule. A future round must eliminate
or fuse whole dense-projection, routing/indexer, quantization/cache, and launch
streams while also redesigning the grouped MoE and sparse-attention pipelines.
