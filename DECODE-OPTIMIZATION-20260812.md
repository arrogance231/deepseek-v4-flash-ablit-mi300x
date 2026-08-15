# Decode optimization, 2026-08-12

This round optimized the production DeepSeek V4 Flash deployment for C2–C8
decode without changing its serving contract: static DSpark K=7,
probabilistic draft sampling with block rejection, 393,216-token context,
64-sequence admission, and the 4,096-token batch budget all remain enabled.

## Result

Fixed-seed measurements use 400-word uncached prompts. C1–C8 used five
512-token runs; C64 used three 256-token runs. Values below are medians.

| Concurrency | Aggregate baseline | Aggregate final | Change | Per-user baseline | Per-user final | Change | TTFT baseline | TTFT final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 136.09 tok/s | 146.53 tok/s | +7.7% | 138.43 tok/s | 149.20 tok/s | +7.8% | 71 ms | 70 ms |
| 2 | 247.63 | 231.70 | -6.4% | 128.22 | 130.52 | +1.8% | 122 ms | 119 ms |
| 4 | 374.35 | 329.27 | -12.0% | 115.98 | 108.70 | -6.3% | 203 ms | 200 ms |
| 8 | 527.52 | 598.83 | +13.5% | 86.11 | 98.56 | +14.5% | 322 ms | 320 ms |
| 64 | 1,110.99 | 1,219.05 | +9.7% | 22.13 | 23.97 | +8.3% | 1,241 ms | 1,220 ms |

The final suite reused the baseline's exact prompt/seed set after the candidate
restart, with cold-prefix TTFT matching the baseline rather than the earlier
48 ms cached-prefix measurements.
Wall throughput still varies materially with speculative acceptance, so the
matched GPU trace is the cleaner kernel comparison. In particular, the C2/C4
aggregate medians are not evidence of a kernel regression: the candidate's
accepted-token ratio was only 0.333/0.293 in those samples. The median steady
C8 K=7 graph fell
from 34.25 ms to 33.47 ms (-2.3%). Sparse-attention partial fell from
68.44 us to 48.87 us (-28.6%); its reduction rose from 6.53 us to 7.42 us,
for a net win. The two large custom MoE kernels remained the dominant decode
cost at approximately 159 us and 102 us per layer.

## Retained changes

### Decode sparse-attention tiling

`patches/rocm_aiter_mla_sparse.decode-h32-k16.py` changes only the decode tile
from 16 heads x 32 KV tokens to 32 heads x 16 KV tokens. Two 32-head
workgroups reduce redundant KV loading on gfx942.

The offline sweep covered B8/B16/B32/B64, the production 128+512-token sparse
window, and synthetic extra segments through 4,096 tokens. The benefit grows
with batch and segment length: approximately neutral at B8, 3.9% at B16,
10.2% at B32, 23.7% at B64 for 128+512, and 33–34% at B64 with a 4,096-token
extra segment.

Correctness was compared directly against the prior kernel across all those
batch/length classes:

- `allclose(atol=0.02, rtol=0.02)` passed;
- relative L2 was 4.55e-4 to 7.86e-4;
- maximum BF16 absolute difference was 0.125–0.25;
- all outputs were finite;
- repeated candidate output was bit-identical.

The small difference is expected because the tile changes the BF16
softmax/reduction association.

### Exact decode A8W8 GEMM rows

`tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv` adds 20
gfx942 CK rows: M=8/16/32/48/64 for each of:

- N=1,536, K=4,096
- N=4,096, K=12,288
- N=8,192, K=1,024
- N=32,768, K=1,024

Production-operator repeats measured representative M64 gains of 35.1%,
21.2%, 6.9%, and 51.9%, respectively. Every AITER comparison reported
`errRatio=0`. These GEMMs are a minority of the graph, so the end-to-end
saving is only a few tenths of a millisecond.

## Rejected work

### MoE BM16

Reducing the custom MoE block height looked 17% faster for W2 in an isolated
microbenchmark, but the matched graph regressed from 34.19 to 35.19 ms:

| Component | Baseline | BM16 |
| --- | ---: | ---: |
| W1 | 7.90 ms | 8.30 ms |
| W2 | 5.22 ms | 5.15 ms |
| Routing schedule | 0.329 ms | 0.469 ms |

The schedule overhead and W1 loss exceed the W2 saving. Do not revisit BM16
unless routing schedule construction is removed or fused.

### Preshuffled A8W8 retuning

Retuning the preshuffled GEMMs did not produce a repeatable operator win. The
production table was retained.

### Further low-risk ceiling

At C8 the two custom MoE matrices consume about 38% of a 34 ms graph, dense
GEMMs about 19%, and sparse attention about 12%. This round exhausted safe
tile/config changes. A substantially larger decode gain now requires a new
MoE kernel or cross-kernel routing/quantization fusion, not another vLLM flag
sweep. Those changes carry a much larger correctness and engineering cost.

## Final gates

The deployed candidate passed:

- health from the inference container and Caddy-to-inference path;
- OpenAI streaming completions;
- automatic tool call (`get_weather`, Melbourne) on two runs;
- two model/logprob quality passes with finite outputs;
- C64 load/capacity (three complete 16,384-token batches);
- C1 uncached prefill: 11.36K tok/s median, within 1.5% of the 11.53K anchor;
- mixed four-decode plus 52K-token prefill: 7.319 s cold TTFT, maximum decode
  gap 177 ms, and late-short TTFT 291 ms;
- 379K chat context: HTTP 200 and exact recall of all three embedded codes;
- API-reported `max_model_len=393216`;
- static K=7 metrics: all seven accepted-token positions incremented.

The production mount list is the repository `compose.yaml`. The prior sparse
kernel and GEMM table remain in the repository to make the two final deltas
auditable.
