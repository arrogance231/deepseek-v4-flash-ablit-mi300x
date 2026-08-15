# MI300X prefill optimization, 2026-08-09

This records the final state of the August prefill round. The
production files in this directory are authoritative; older measurements in
the main runbook describe the preceding 2,048-token configuration.

## Retained production profile

- vLLM image: `e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a`
- Model revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- `max_model_len=393216`, `max_num_seqs=64`
- `max_num_batched_tokens=4096`
- DSpark K=7 remains reserved; the ordinary prefill quantum is 3,712 tokens
- The 1,024-token latency-isolation cap applies only while another request
  could be delayed. A lone prefill may use the full 3,712-token quantum.
- Full and piecewise graphs are captured through M=3,712.

Run `./prepare-artifacts.sh` before Compose. It expands the validated vLLM
extension and verifies its SHA-256. `compose.yaml` mounts installed overlays
read-only; `/opt/cj-moe` remains writable for the HIP extension build.

## Final measurements

The original deployment produced approximately 5.26K uncached prefill tok/s
at C1. The final post-restart runs were 11.62K, 11.51K, and 11.53K tok/s for
8.9K-token cold prompts: an 11.53K median and 2.19x the initial result. The
preceding seven-run candidate median was 11.51K. This overall comparison
includes the larger scheduler quantum and all retained kernel changes; it is
not an isolated kernel A/B.

### How the gain accumulated

The missing middle was a sequence of scheduler, kernel, support, and batch
changes rather than one breakthrough. Milestone medians were:

| Milestone | C1 tok/s |
| --- | ---: |
| Original baseline | 5,260 |
| Contention-aware scheduler | 6,958 |
| First custom HIP MoE | about 7,900 |
| Attention/support stack | 8,298 |
| Asymmetric row-INT8 W1 | 8,988 |
| Initial gfx942 OPUS port | about 9,360 |
| BM64/delta-scale W1 | 9,569 |
| Batch 4,096 plus M=3,712 tuning | 10,972 |
| Exact W2 changes | 11,085 |
| Graph buckets plus A8W8 ASM | 11,239 |
| OPUS no-padding | 11,391 |
| Final deterministic top-k deployment | **11,528** |

These are chronological milestones, not an additive waterfall: prompt routing
and restart state differ. Controlled A/B results, implementation details,
correctness boundaries, rejected experiments, and the complete artifact index
are in the
[August 8–9 experiment log](PREFILL-EXPERIMENT-LOG-20260808-09.md).

The last full production gate, immediately before the mathematically
equivalent top-k replacement, produced:

| Gate | Result |
| --- | ---: |
| C1 cold prefill | 11.39K tok/s |
| C2 cold prefill | 10.19K tok/s |
| C4 cold prefill | 11.20K tok/s |
| C8 cold prefill | 11.36K tok/s |
| C64 decode | 21.94 tok/s/user |
| Mixed 50K prefill TTFT | 7.557 s |
| Late short-request TTFT during that prefill | 0.340 s |
| Maximum background-token gap | 0.196 s |
| Long-context recall | All markers at 379,906 prompt tokens |

Completion, chat, streaming, reasoning, and automatic tool calling passed.
The quality fixture suite passed twice.

### Final deterministic top-k

Prefill invokes top-k 63 times for the profiled request. The stock HIP kernel
was fast but used atomic append order at exact cutoff ties. The retained vLLM
extension gives equal scores a deterministic lower-original-index tie break,
including the otherwise unstable case with more than 2,048 equal values. The
selected indices are sorted before sparse attention.

Standalone tests covered every observed shape (`3712x928`, `3712x1856`, and
`1466x2222`), ragged row bounds, all-equal inputs, mixed cutoff ties, ties over
2,048 elements, and 20 quantized-tie seeds. Continuous random inputs selected
the same set as `torch.topk`; tied inputs obeyed the explicit stable policy.
PyTorch does not specify which equal-valued element it returns, so its arbitrary
tie subset can differ while the mathematical top-k remains identical.

| Shape | `torch.topk` production path | HIP + canonical sort |
| --- | ---: | ---: |
| 3712x928 | 0.214 ms | 0.122 ms |
| 3712x1856 | 0.225 ms | 0.123 ms |
| 1466x2222 | 0.145 ms | 0.067 ms |

Across seven paired server prompts, five improved and the median paired TTFT
saving was 4.5 ms. Profiled top-k device time fell from 13.29 to 7.90 ms. One
candidate quality run matched the preceding production run byte-for-byte on
all fixtures; the repeated candidate run matched 10/11, within the baseline's
own run-to-run variation. This small but real win was retained without a final
A-B-A restart because model startup and graph capture take roughly ten minutes.

## What worked

### Scheduling and batch geometry

Raising the batch budget from 2,048 to 4,096 was the largest change. Keeping
the speculative reservation leaves M=3,712 for ordinary work. The
contention-aware scheduler uses that quantum only when no waiting or running
request needs latency protection; under contention it retains the 1,024-token
chunks. This preserved the mixed-workload gap and short-request gates while
removing the artificial C1 ceiling.

The AITER tables contain the recurring M=3,712, M=1,664, and latency-isolation
shapes. Exact-shape CK tuning gave large operator improvements for several
fallback rows, but only a low-single-digit end-to-end gain; it was useful, not
the principal headroom.

### Sparse attention and indexer

- A FlyDSL FP8 MQA-logits path moved the seven-run median from 8.02K to 8.13K
  tok/s (+1.31%).
- Processing all 64 query heads per sparse-prefill workgroup avoids reading
  gathered KV four times.
- Presuffling `wq_b` removed about 9.5 ms of GPU time per request.
- The final OPUS kernel removes an unnecessary LDS padding column. LDS fell
  from 33,792 to 32,768 bytes, permitting two resident workgroups; VGPRs fell
  from 384 to 247 and AGPR use from 136 to zero. It was bit-exact and 8-20%
  faster in operator tests. A seven-prompt server A/B improved 11.247K to
  11.391K tok/s (+1.29%) and saved 9.3 ms median paired TTFT.
- The deterministic HIP top-k removes another approximately 5.4 ms of device
  work per request.

### Expert kernels

The retained W1/W2 HIP kernels load activations directly into LDS. Across real
routing histograms W1 improved 1.2-2.1%; the combined retained W1/W2 change
removed 4.84 ms of MoE device time in a matched trace. Checkpoint-layer tests
and model fixtures passed. Weight preparation is performed once at load time;
do not move it back into the request path.

## What did not work

| Experiment | Finding |
| --- | --- |
| More vLLM flag sweeping | Prior sweeps had already reached a local optimum. Material gains came from scheduling and kernels, not another backend flag. |
| Unconditional 3,712-token prefills | Good C1 throughput but harmed interactive traffic. Retained only through the contention-aware scheduler. |
| MHC BF16 high/low MFMA | Approximately 1-2% slower than the existing FP32 path. |
| MHC K=64 tiles | Slower than K=128. Existing split-K was optimal at M=3,712; non-temporal stores were negligible. |
| W2 split accumulators | Faster in isolation but reassociated sums and changed two deterministic fixture continuations. Reverted; direct-LDS loading remains. |
| W2 FP8/INT8 re-quantization variants | Conversion cost, accuracy loss, or both outweighed the GEMM saving. |
| Raw custom prefill top-k plus a final sort | Sorting repaired order but not the atomically chosen set at cutoff ties. Replaced by the patched deterministic selector. |
| Sparse-prefill `BLOCK_K=32` | Failed bit-exact attention checks. Keep `BLOCK_K=16`. |
| Alternate OPUS layouts | Global-memory, wave-skipping, rescale-removal, and padding variants other than the retained no-padding layout did not beat it across the production shape mix. |
| M=1,024 GEMM tuning as the main strategy | Individual kernels improved 13-64%, but C1 moved only about 2%; it exposed that GEMM fallback was not the dominant remaining gap. |

Rejected files and profiler databases remain on the server under
`/opt/deepseek-v4/kernel-dev` and `/opt/deepseek-v4/optimization-20260808`.
They are intentionally not copied into this production repository.

## Remaining gap

The latest pre-top-k trace contained 758.6 ms of summed GPU activity and about
750 ms of activity union. The retained top-k removes roughly 5.4 ms. Critical
path contributions were approximately:

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
| Remaining launches and elementwise work | Approximately 58 ms |

For the 8,594-token reference, 26.3K tok/s permits only 326.8 ms. Even running
W1, W2, and OPUS at unattainable hardware peak leaves more than that complete
budget. The target therefore cannot be reached by tuning those three kernels
alone. The next credible round needs structural fusion or elimination in dense
projections, indexer support, quantization/cache traffic, and launch/copy work.
Small MHC or scheduler-flag sweeps are exhausted.

Machine-readable roofline data and the earlier trace aggregation are retained
under `findings/`. They predate the final OPUS and top-k deltas, so use the
numbers above for the production endpoint and those files for methodology.
