# DeepSeek V4 Flash expert-activation correctness fix

## Root cause

The production gfx942 W1 kernel implemented `silu(gate) * up` but omitted the
checkpoint's `swiglu_limit=10` semantics:

```text
gate = min(gate, 10)
up = clamp(up, -10, 10)
```

This was present from the first custom W1 deployment. Outlier expert activations
therefore changed target logits and caused the recurring atomic token `)Skip`,
rare unrelated CJK tokens, and less conspicuous code-token errors.

The fix adds those two clamps immediately before the existing SiLU/multiply and
passes the model's `gemm1_clamp_limit` into the HIP kernel. No sampler, tokenizer,
prompt, weight, routing, or speculative-verification behavior was changed.

## Reproduction and corrected output

- Direct W1 regression: the old kernel produced values with absolute maximum
  136; the corrected kernel was bounded at 100 and changed 697/786,432 elements.
- Three known greedy boundaries where `)Skip` was formerly top-1 now select the
  expected punctuation; `)Skip` is absent from their leading candidates.
- Fixed-prefix sampling collapsed from 97 token types, including corrupt tail
  tokens, to the expected two punctuation tokens across 500 seeds.
- Native raw `/v1/completions`: 120 seeds x 512 tokens, temperature 1.0,
  top-p 0.95: **0 `)Skip`, 0 CJK**.
- Production K7 raw `/v1/completions`: the same 61,440-token test:
  **0 `)Skip`, 0 CJK**.

The historical optimized repro emitted `)Skip` in 3/120 responses and stray CJK
in 3/120. The independent unfused implementation did not, which localized the
failure to the fused expert path before the kernel was inspected.

## Corrected performance

All measurements below used four uncached 256-token runs per concurrency with
Caddy stopped. Values are medians from the corrected implementation.

| Concurrency | Native aggregate tok/s | Native tok/s/user | K7 aggregate tok/s | K7 tok/s/user | K7 accepted/draft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.28 | 68.31 | 152.56 | 158.75 | 2.167 |
| 2 | 123.48 | 63.45 | 207.00 | 132.86 | 1.703 |
| 4 | 223.32 | 58.33 | 327.72 | 95.22 | 1.532 |
| 8 | 393.02 | 53.83 | 510.46 | 79.80 | 1.558 |
| 16 | 571.37 | 46.78 | 728.01 | 53.77 | 1.530 |
| 32 | 1,079.08 | 37.91 | 975.62 | 36.98 | 1.485 |
| 64 | 1,649.80 | 29.64 | 1,278.23 | 25.14 | 1.563 |

The synthetic random-word workload has substantially lower acceptance than
production traffic; it is a repeatable throughput load, not an acceptance model.
Native decode is effectively unchanged, as expected for two clamp instructions
inside the existing W1 epilogue.

## Promotion gates

- two independent production tool rounds: 64/64 two-turn cases passed;
- 379,047-token and 393,051-token requests recalled all three needles exactly;
- steady C1 prefill: 11.69K tok/s;
- production K7 service healthy with zero restarts after restoration.

