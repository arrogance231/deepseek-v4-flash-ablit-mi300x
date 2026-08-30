# Measurements: windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated

These are measurements taken on one MI300X VF (`gfx942`) in the companion
investigation, not copied from another checkpoint. The exact 48-shard FP8
checkpoint is identified by revision `6de83db0be050e0338ae2f8376440642203ad90d`.

## Runtime profile

- PyTorch 2.11.0+gitd0c8b1f, HIP 7.2.53211
- vLLM 0.27.1+rocm723, Triton 3.6.0
- DSpark K=7, probabilistic drafting, block rejection
- recipe overlays, AITER, custom gfx942 kernels, full/breakable graphs
- `fp8_ds_mla` KV, 16 GB GPU pool, 96 GiB native CPU KV tier
- prefix caching enabled, max context 393,216, max sequences 64
- model weights ~156.47 GiB; HBM high-water approximately 197–200 GB

## Throughput

Three identical 600-token deterministic list requests measured:

| Request | Completion | Wall time | End-to-end rate |
| ---: | ---: | ---: | ---: |
| cold | 600 | 2.99 s | 200.6 tok/s |
| warm | 600 | 2.14 s | 280.6 tok/s |
| warm | 600 | 2.14 s | 280.3 tok/s |

This is end-to-end completion throughput, not an isolated decoder-kernel
rate. Tool-call requests were approximately 161–171 tok/s warm.

## Correctness gate

A corrected 20-repetition suite produced 0/20 premature EOS, 20/20 EOS probes
reaching integer 500, 20/20 required tool calls parsed correctly, 0/20 no-tool
requests emitting DSML, and 0/20 parsed assistant messages leaking DSML. Raw
DSML was present in 20/20 tool generations as expected and was removed by the
parser. Raw decoded token strings and complete API responses are retained by
the external benchmark; numeric token IDs were not exposed by this API.

These are startup/short-run results. They do not establish multi-hour
stability; use the soak and restart-recovery procedures before production.
