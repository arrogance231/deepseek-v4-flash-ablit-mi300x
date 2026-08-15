# Decode optimization round 3 (2026-08-14–15)

## Result

This round targeted the low-concurrency decode regime (M=1–8), where the
round-2 ledger showed the graph is launch/latency-bound rather than
bandwidth-bound. Two bit-exact changes were retained; both were validated with
`torch.equal` at operator level before any server boot.

Native (non-speculative) decode, matched 4-run medians, 400-word prompts,
256 output tokens:

| Native batch | Baseline aggregate | Final aggregate | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 61.0 tok/s | 67.1 tok/s | **+10.0%** |
| 2 | 114.5 tok/s | 122.7 tok/s | **+7.2%** |
| 4 | 214.2 tok/s | 221.1 tok/s | +3.2% |
| 8 | 393.7 tok/s | 400.5 tok/s | +1.7% |
| 16 | 585.9 tok/s | 686.6 tok/s | +17.2%* |
| 32 | 1,069.3 tok/s | 1,094.1 tok/s | +2.3% |
| 64 | 1,628.1 tok/s | 1,637.2 tok/s | +0.6% |

*The C16 delta exceeds the modeled kernel saving; part of it is run variance.

Production static-K7, acceptance-normalized (per-draft-cycle latency, which
isolates kernel speed from content-dependent DSpark acceptance):

| Concurrency | r2 cycle | r3 cycle | Delta | r2 agg | r3 agg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.52 ms | 19.91 ms | **-2.9%** | 144.6 | 144.1 |
| 2 | 24.35 ms | 25.39 ms | +4.3% | 236.9 | 274.8 |
| 4 | 32.65 ms | 31.27 ms | -4.2% | 433.0 | 415.6 |
| 8 | 41.19 ms | 41.16 ms | -0.1% | 650.4 | 660.5 |
| 64 | 115.09 ms | 117.69 ms | +2.3% | 1,260.2 | 1,227.4 |

The C1 cycle improvement matches the modeled M=8 verification saving
(~0.6 ms of a 20.5 ms cycle). C2–C64 cycle deltas are within run noise;
aggregates move with acceptance (1.99–2.60 accepted/draft across runs).

## Profiling: why M=1–8 decode is dispatch-bound

Per-M traces (`optimization-20260814-decode-round3/profile-r3-m{1..64}`,
ledger in `r3-ledger.json`) show the M=1 graph is 17.46 ms across 1,970
kernels with only 0.04 ms intra-graph idle and 0.30 ms host gap. Top buckets
at M=1:

| Bucket | Time | Launches |
| --- | ---: | ---: |
| dense GEMMs | 5.27 ms | 407 |
| MoE W1 | 2.75 ms | 43 |
| sparse attn/index | 2.03 ms | 253 |
| MHC | 1.35 ms | 258 |
| dynamic quant | 1.09 ms | 236 |
| MoE W2 | 1.09 ms | 43 |

~1,200 launches sit at the ~4.5–6 µs launch floor regardless of M: the
quant/support/MHC chains cost the same at M=1 as at M=8. HBM utilization is
low; the bound is kernel count and per-kernel latency, not bandwidth.

## Retained change 1: N-split ("nsplit") W1/W2 MoE kernels for M<=8

At M=1, only ~6 experts are active with 1 row each; the BM16 adaptive kernels
from the round-2 rewrite still launched one workgroup per (expert, N-tile)
with a full-N tile, filling a fraction of the 304 CUs. The nsplit variants
split the N dimension across additional workgroups when the token count is
small (`x.size(0) <= CJ_NF_SMALL_M=8`, waves 4 for W1), recovering occupancy.
Larger shapes take the unchanged code path.

Operator results (production wrappers, synthetic 6-expert routing,
`torch.equal` exact at every M):

| M | W1 speedup | W2 speedup |
| ---: | ---: | ---: |
| 1 | **1.62x** | **1.29x** |
| 2 | 1.40x | 1.07x |
| 4 | 1.19x | 1.08x |
| 8 | 1.10x | 1.04x |
| 16–64 | 1.00x (gated off) | 1.00x |

Per-pass saving: ~1.25 ms at M=1, ~0.6 ms at M=8 (the K7 C1 verify shape).
Files: `kernel-dev/hip-a8w4/fusedi8asym64ragged_activea.hip`,
`kernel-dev/hip-a8w4/w2.hip` (gen script `gen_nsplit.py`, bench
`bench_r3_nsplit.py` on the server).

## Retained change 2: decode-M rows for the A8W8 bpreshuffle table

`get_CKGEMM_config` does exact/padded-M lookup with **no nearest fallback**;
shapes absent at small M fall back to an untuned default CK kernel. The
production table had no rows below M=1664 for `(N=32768,K=1024)` (wq_b) and
none below M=64 for `(N=8192,K=4096)`. Tuned rows (errRatio=0, splitK=0) were
added for M∈{1..32} / {1..64}; padded-M lookup covers every decode M.

Profiler-isolated kernel times:

| Shape | M | Before | After |
| --- | ---: | ---: | ---: |
| 32768x1024 | 1–16 | 9.0–9.3 µs | **4.5–4.9 µs** |
| 8192x4096 | 1–32 | 17.5–21.5 µs | **7.1–8.6 µs** |
| 8192x4096 | 64 | 15.9 µs | 13.0 µs |

All 28 shape/M outputs `torch.equal` against the incumbent kernels.
Per-pass saving ~0.6–0.7 ms across 43 layers. 13 rows added to
`tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.batch4096.csv`
(backup: `optimization-20260814-decode-round3/bpreshuffle-table.pre-r3.csv`).

## Retained change 3: mid-M GEMM rows for the C8+ verify and drafter range

A follow-up audit of the CUDA-graph capture logs showed that although the
capture-size grid itself is dense (8-token steps to 256, 16-token steps to
512), **every dense A8W8 GEMM inside the M=72–512 graphs ran an untuned
default config**: the verify path (M=8C for C=9..64), the DSpark drafter path
(M=7C — DSpark is self-speculative, so the drafter hits the same layers via
the standard table), and even the drafter at C1–C8 for `(4096,8192)`
(M=7,14,..). Neither table had rows between the small-M decode grid and the
M=1664+ prefill grid.

Rows were tuned at the `get_padded_m` gl=0 targets (multiples of 16 to 256,
32 to 512), then re-measured in-context against the incumbent default with
the profiler harness. Only bit-exact rows faster than the incumbent were
merged: **21/22 bpreshuffle rows and 83/100 standard rows** (all 122 tuned
rows were bit-exact; rejects were merely not faster).

Representative kernel-level reductions:

| Table / shape | M range | Before | After |
| --- | ---: | ---: | ---: |
| bpreshuffle 32768x1024 (wq_b verify) | 80–512 | 19.8–65.1 µs | 13.8–54.1 µs (−9 to −30%) |
| standard 32768x1024 (drafter) | 80–448 | 33.6–140.9 µs | 17.2–58.1 µs (**−44 to −62%**) |
| standard 8192x1024 | 224–512 | 17.3–42.0 µs | 10.6–17.0 µs (−39 to −60%) |

A live C64 trace confirms the new kernels replay in the graphs at the
offline-measured durations (e.g., the 2-LDS variant at 53.6 µs vs the 60.3 µs
prior default), and the 80–140 µs default kernels are gone from the decode
profile. Matched-seed wall results: C8 neutral (−1.4%, expected — M=64 was
already tuned), C64 +5.0% aggregate with near-matched acceptance
(1.50 vs 1.57/draft); mid-C (C10–C48) is where the largest kernel deltas
apply but has no prior wall baseline. New absolute medians: C12 790,
C16 896, C24 1,009, C32 1,190, C48 1,357 aggregate tok/s.

13 + 21 rows were added to
`tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.batch4096.csv` and 83 rows
to `tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv`
(backups: `optimization-20260814-decode-round3/*-table.pre-*.csv`).

## Rejected paths

- **Tuner rows for `(4096,8192)` and `(1536,4096)`**: the tuner's "winners"
  were slower than the incumbent default at M=1/4 (16.1 vs 11.1 µs) or
  neutral. Excluded; tuner harness times disagree with in-context profiler
  times, so every row must be re-measured against the incumbent selection.
- **TunableOp for skinny BF16 mms** (4096→2048/512/256/64): 15–45% per call
  (~25→14–19 µs), but outputs are not bit-exact (rel-L2 up to 6e-5, different
  hipBLASLt reduction orders) and the tuning file did not persist reliably.
  ~0.3–0.4 ms/pass potential; deferred as a numerics-gated candidate.
- **Sparse decode split-count increase**: the existing `_decode_num_splits`
  heuristic (cap 16) is already optimal at M=1–8; forcing 24/32 splits was
  neutral at M=1 and 25–40% slower at M=8. Closed.
- **Mid-M rows that lost to the default**: `(1536,4096)` at M=208–352,
  `(4096,8192)` at all Ms, `(4096,12288)` at M=80–112, and bpreshuffle
  `(32768,1024)` at M=64 — tuner-harness winners that were equal or slower
  in-context. The remaining boot-log "not found tuned config" messages for
  these shapes are intentional.
- **MHC pre-chain fusion, quant-into-GEMM prologues, gating kernel rewrite**:
  still open; each is a multi-hour kernel project against ~0.3–0.8 ms of
  launch-floor time. Documented as future work.

## Validation

- Operator: `torch.equal` for nsplit W1/W2 at M=1..64 and for all 28 table
  shape/M outputs.
- Mid-M rows: `torch.equal` for all 122 tuned shape/M outputs against the
  incumbent kernels; live C64 trace kernel attribution.
- Endpoint numerics: cold-boot greedy fixture replay vs the round-2
  reference — all argmax tokens identical; stable fixtures match top-5
  logprobs to <=1e-6. (Text-level equality is not a valid gate: round-2's own
  back-to-back replays matched only 4/11 because warm prefix-cache changes
  prefill chunking; the same 4/11 self-match reproduces today.)
- Prefill C1: 11,799–11,815 tok/s (reference 11,657) — preserved; re-checked
  after the mid-M merge at 11,517–11,615 tok/s.
- Tools: 32/32 passed (re-run after the mid-M merge: 32/32).
- Long context: 379K chat recall exact (alpha/beta/gamma), ~53 s cold;
  re-run after the mid-M merge (the new rows also serve chunked-prefill tail
  chunks): exact recall again.
- Mixed interference: cold 52K TTFT 7.19 s, late-short TTFT 316 ms, max
  decode gap 191 ms — all within prior-round ranges.
- Zero restarts or HIP faults in final logs.

## Remaining headroom

After this round the M=1 graph is ~15.5 ms (from 17.46). The remaining
decode-latency budget is dominated by ~1,200 launch-floor kernels
(quant 430, MHC 258, attn support ~150, MoE support ~170, norms/elementwise
~190, copies 77 — ~5.8 ms) plus the sparse-attn partial (0.96 ms) and the
skinny BF16 hipBLASLt calls (~1.6 ms). Meaningful further gains require
cross-kernel fusion (quant prologues, MHC chain, gating) or fewer launches
per layer; per-kernel tuning is exhausted.
