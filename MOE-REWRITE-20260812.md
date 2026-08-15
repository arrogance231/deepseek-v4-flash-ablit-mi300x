# Adaptive gfx942 MoE rewrite (2026-08-12)

## Result

Production now selects expert tile height from the static graph shape:

- `M <= 64`: BM16 for W1 and W2.
- Larger shapes: the previous BM64 W1 and BM48 W2 path.
- One schedule launch builds only the selected schedules; both paths are never launched.

The W1/W2 arithmetic is unchanged. M8, M64, M512 and five real 9,984-route layer captures were bit-identical to the prior kernels. Large-shape execution was neutral within measurement noise.

| Offline full pipeline | Prior | Adaptive | Change |
|---|---:|---:|---:|
| M8 / 48 routes | 0.2390 ms | 0.2117 ms | -11.4% |
| M64 / 384 routes | 0.8527 ms | 0.7707 ms | -9.6% |
| M512 / 3,072 routes | 1.1353 ms | 1.1330 ms | -0.2% |

A production C8 trace measured W1 BM16 at 157.0 us/layer and W2 BM16 at 95.7 us/layer, versus 166.2 and 107.7 us/layer in the matched prior trace. This removes about 21 us/layer (roughly 1.0 ms across 48 MoE layers) from each target-model pass.

Production gates after deployment:

- C1 prefill median: 11,486.6 tok/s (five runs 11,443.5–11,542.9).
- C64: 1,192 tok/s aggregate, 23.8 tok/s/user.
- Mixed load: 161 ms maximum decode event gap during a 52K-token cold prefill.
- Tools/streaming: 16/16 two-turn structured tool-call tests passed.
- 379K source / 379,006-token chat: exact three-needle recall and the same output SHA-256 as the prior production gate.

End-to-end decode is noisy because DSpark acceptance dominates request wall time. The production profile is the authoritative implementation result: it compares kernels directly and confirms the expected MoE saving. Static K=7 reservation, context, batch, and capacity settings did not change.

## Why adaptive tiling works

At C1-C8, most of the 256 experts receive only a small row group. BM64/BM48 reserves LDS and accumulator state for rows that do not exist. BM16 reduces this waste. Prefill has dense or imbalanced experts where larger tiles amortize work and remain faster, so static BM16 was not acceptable.

The HIP extensions template BM and make one host-side choice from `hidden.shape[0]`. CUDA graph capture fixes that choice per graph, with no GPU synchronization or dynamic dispatch during replay.

## Architectural attempts

### Retained

- Row-asymmetric INT8 hidden activation quantization.
- Fused W1 INT8 x MXFP4 GEMM and SiLU.
- Direct global-to-LDS activation loads for W1/W2.
- Mathematically identical adaptive BM16/BM64/BM48 execution.
- Single selective schedule kernel.

### Rejected

- **Global BM16:** helped low-route decode but regressed the full graph and prefill. Adaptive dispatch fixes this.
- **Launching small and large kernels together:** extra launches made M8 0.123 -> 0.291 ms.
- **No-fill schedules:** unsafe because fixed-capacity graph launches require `-1` sentinels unless task counts synchronize to the host.
- **W2 atomics/removing reduction:** deterministic six-way reduction is only about 4.7 us/layer; upside was too small.
- **Full W1-to-W2 fusion:** W2 consumes the complete 2,048-wide intermediate across independently owned W1 tiles. Correct execution needs persistent cross-workgroup coordination or substantial recomputation.
- **Fusing input quantization into expert tiles:** duplicates each row's quantization across expert/output tiles to save only 8-10 us/layer.
- **W2 accumulator reassociation:** faster but changed deterministic model continuations; rejected despite low relative-L2 error.
- **INT8/FP8 W2 intermediate rewrites:** introduced model-level drift or did not beat the direct FP16 path enough.

## Files

- `patches/gpt_oss_triton_kernels_moe.row-i8asym-candidate.py`: graph-shape selection and extension calls.
- `kernel-dev/hip-a8w4/schedule.{cpp,hip}`: selective BM16 or BM48+BM64 schedules.
- `kernel-dev/hip-a8w4/fusedi8asym64{.cpp,ragged_activea.hip}`: templated BM16/BM64 W1.
- `kernel-dev/hip-a8w4/w2.{cpp,hip}`: templated BM16/BM48 W2 and deterministic reduction.
