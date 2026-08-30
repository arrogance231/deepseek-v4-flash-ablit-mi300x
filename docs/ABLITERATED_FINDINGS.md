# Findings: exact abliterated checkpoint on MI300X

Checkpoint tested (no substitution):
`windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated`
revision `6de83db0be050e0338ae2f8376440642203ad90d`.

## Tested profile

One MI300X VF (`gfx942`), vLLM `0.27.1+rocm723`, PyTorch
`2.11.0+gitd0c8b1f`, HIP `7.2.53211`, Triton `3.6.0`; AITER and the
MI300X overlays/custom kernels; DSpark K=7 with probabilistic drafting and
block rejection; full/breakable graphs; prefix caching; 16 GB GPU KV plus
96 GiB native CPU KV offload; `max_model_len=900000` is now configured in the repository; the completed validation run used 393216 and 900K remains unvalidated. `max_num_seqs=64`.
Weights occupy approximately 156.47 GiB HBM.

The reference repository's published image digest was unavailable from the
registry, so this validation uses the locally available `rocm:latest` image
with the recipe overlays. The model remains the exact requested checkpoint.

## Throughput

Three identical 600-token deterministic list requests measured 200.6 tok/s
cold, then 280.6 and 280.3 tok/s warm (end-to-end completion rate). Warm
required tool-call requests measured approximately 161–171 tok/s. The first
request includes graph/cache warm-up and is not comparable to steady state.

## Correctness gate

The corrected 20-repetition suite produced:

- EOS 1–500: 20/20 reached 500; 0 premature EOS.
- Required `get_counter` calls: 20/20 selected and parsed correctly.
- No-tool requests emitting DSML: 0/20.
- Parsed assistant content leaking DSML: 0/20.
- Raw DSML in required tool generations: 20/20, as expected before parsing.
- Hard deterministic 1,154-token inventory task: correct ledger, invariants,
  checksum 443, final vector `[51, 35, 54, 40]`, and order `C,A,D,B`.

Raw decoded token strings, parser results, finish reasons, and API responses
are retained by the companion test artifact. This API exposes `token_ids:
null`, so numeric token IDs are not available through the OpenAI response.

## Interpretation

These are startup/short-run results, not proof of multi-hour stability. The
three-hour periodic soak must complete before making an uptime-degradation
claim. A BF16 KV cache was rejected by the installed DeepSeek V4 path because
its `fp8_ds_mla` layout requires an FP8 KV format. Disabling AITER was also
rejected because the ROCm sparse indexer requires AITER.
