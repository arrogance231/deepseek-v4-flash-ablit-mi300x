# Findings on the abliterated checkpoint

This note separates measurements made with the abliterated checkpoint from
the upstream reference measurements. That distinction is important: the
reference project measured the original `deepseek-ai/DeepSeek-V4-Flash-0731`
weights, while this deployment uses
`lovesenko/DeepSeek-V4-Flash-0731-Abliterated`.

## Executive result

The abliterated checkpoint runs successfully on one MI300X with the pinned
ROCm/vLLM stack. The best tested single-stream result is approximately
**100.5–106.2 decode tok/s** with probabilistic DSpark-7, versus the reference
project's **152.6 tok/s** on the original checkpoint. The difference is real
for this checkpoint; it is not evidence that the MI300X stack failed to load
or that the reference number is fabricated.

The current deployment is configured for **393,216 total tokens** (prompt plus
completion). The checkpoint advertises 1,048,576 positions in `config.json`,
but a 1M request has not been validated for quality or stability here. We
therefore recommend 393K as the current operational ceiling and 1M as an
experimental restart-only profile.

## What the reference repository did answer

The reference project established that a single MI300X can run this model
family with:

- BF16-compatible model execution and the complete FP8/MXFP4 weight set in HBM;
- AITER/OPUS attention and prefill paths;
- custom `gfx942` MoE kernels and graph capture;
- static DSpark-7 speculative decoding;
- a 16 GB GPU KV pool plus a 96 GiB native CPU tier; and
- approximately 11.69K uncached prefill tok/s and 152.6 tok/s single-stream
  DSpark decode on its reference checkpoint.

See the [reference MI300X project](https://github.com/ryanzhou/deepseek-v4-flash-mi300x)
for that benchmark and its exact harness.

## What it did not answer

It did not establish the following for an abliterated weight edit:

1. Whether the edited MTP/draft head preserves the reference DSpark acceptance
   rate.
2. Whether the modified model has the same decode throughput under normal
   chat sampling rather than the reference's synthetic workload.
3. Whether long-context narrative quality, factual recall, and tool-call
   behavior remain unchanged at 393K or 1M.
4. Whether greedy drafting is better than probabilistic drafting for the
   edited checkpoint.
5. Whether the reference's 152.6 tok/s can be reproduced without changing
   the checkpoint.

Those are checkpoint-specific questions, not runtime installation questions.

## Measurements made after swapping in the abliterated weights

All measurements below used the same pinned MI300X image and serving overlays
already described in the main README. They are intentionally reported as
smoke/engineering measurements, not as a new standardized leaderboard.

| Measurement | Result | Notes |
| --- | ---: | --- |
| Model load/API health | Pass | OpenAI-compatible endpoint remained healthy |
| DSpark-7 decode | 100.5–106.2 tok/s | 512-token streamed requests; probabilistic drafting |
| Earlier smoke anchor | 100.91 tok/s | 0.24 s TTFT; one 512-token request |
| Greedy DSpark A/B | ~89–93 tok/s | Slower than probabilistic drafting on this checkpoint |
| Uncached prefill | ~9.94K–11.40K tok/s | 14K–121K-token unique prompts |
| Prefix-cache reuse | 1,536 tokens | Repeated stable prefix; second request reported cached tokens |
| Current context flag | 393,216 | `--max-model-len`; includes output tokens |
| GPU memory after long probes | 202.1 / 205.8 GB | Only about 3.7 GB free at the observed high-water |
| Host memory | 108 / 235 GiB used | Swap remained essentially unused |

The server also reported a 16 GB GPU FP8 DeepSeek KV pool and a 96 GiB native
CPU tier. Prefix caching and chunked prefill were active. These features help
capacity and prompt latency; they do not, by themselves, raise single-stream
decode tok/s.

## Why DSpark is slower on the abliterated model

DSpark drafts a block of tokens and asks the target path to verify them. The
speedup depends on how many drafted tokens survive verification. The reference
table reports `2.167` accepted/draft for its single-stream K7 row. Our live
vLLM counters for the abliterated model showed substantially weaker draft
acceptance (roughly 14–17% of proposed draft-token counters accepted in the
observed runs). The runtime counters and reference table are not identical
units, so this is directional rather than a strict ratio comparison, but the
failure mode is clear: more draft work is rejected.

The model card documents direct edits to attention output weights and the
`mtp.wo_b` draft head. Those edits can preserve verified output correctness
while reducing draft/target agreement. See the
[abliterated model card](https://huggingface.co/lovesenko/DeepSeek-V4-Flash-0731-Abliterated).

The tested probabilistic sampler was faster than the card-recommended greedy
variant. Static K=7 remains the safe choice because this checkpoint declares a
DSpark block size of five; lower dynamic bands are not supported by the
current ROCm path.

## Context interpretation

The downloaded `config.json` contains:

```text
max_position_embeddings = 1,048,576
rope_scaling.type       = yarn
rope_scaling.factor     = 16
rope_scaling.original_max_position_embeddings = 65,536
```

This is a model-declared 1M ceiling using a large YaRN extension, not proof
that every 1M narrative prompt is reliable. The current server deliberately
uses 393,216 because that is the profile with the best measured memory and
stability margin. Raising the flag to 1,048,576 requires a restart and fresh
graph capture; it should not be described as validated until a full request
and quality check complete.

## Reproducing the current result

```bash
cd deepseek-v4-flash-ablit-mi300x
MODEL_DIR=/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated \
  MAX_MODEL_LEN=393216 \
  docker compose up -d inference

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lovesenko/DeepSeek-V4-Flash-0731-Abliterated",
       "messages":[{"role":"user","content":"Write a short atmospheric scene."}],
       "temperature":0.8,"max_tokens":512,"stream":true}'
```

For a fair comparison, use the same prompt, token count, sampler, warm-up
policy, cache state, and number of repetitions for both checkpoints. Do not
compare the reference's synthetic DSpark result directly with a single
interactive chat request and call the difference a kernel regression.

## Remaining work

This repository now documents the deployment and the abliterated-model
throughput finding. It does **not** claim a complete creative-quality study,
an accepted-token apples-to-apples benchmark, or a validated 1M production
profile. Those require a matched harness and long-form quality fixtures and
should be added as separate experiments rather than inferred from tok/s.

## Phase plan and Phase 1 result

The draft-repair work is staged so that a faster but corrupted output path
cannot be promoted accidentally:

1. **Phase 1 — reversible MTP A/B:** restore the original base MTP tensors
   while leaving all abliterated decoder tensors unchanged. Gate on normal
   chat quality, not raw-completion tok/s. **Completed: rejected.**
2. **Phase 2 — static-K sweep:** compare K=5, K=6, and K=7 on the normal chat
   endpoint with identical seeds and prompts. K below the declared block size
   of five remains out of scope for the current ROCm path.
3. **Phase 3 — MTP calibration:** freeze the abliterated target and distill
   only the MTP/draft path against target-generated continuations. Keep this
   as a sidecar variant and validate DSpark-on/off behavior.
4. **Phase 4 — quality gate:** run longer chat generations, repetition checks,
   fixed-seed parity, and tool/structured-output checks before any promotion.
5. **Phase 5 — promotion:** retain the original MTP path unless a candidate
   wins both speed and correctness gates.

For Phase 1, the script
[`scripts/restore_base_mtp.py`](../scripts/restore_base_mtp.py) created
`/mnt/model-storage/DeepSeek-V4-Flash-0731-Ablit-MTPBase` using base revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. The companion verifier checked all
4,705 tensors and confirmed that exactly six MTP weight/scale tensors changed.

The variant produced an apparent 357 tok/s on a degenerate raw-completion
probe, with almost every draft token accepted. That output repeated entire
paragraphs and leaked prompt-format instructions. On normal chat requests,
the same variant measured 100.7, 105.8, and 107.3 tok/s and did not improve on
the original abliterated-MTP path. It was therefore rejected and the live
server was restored to the original checkpoint. This is precisely why speed
must be gated by output quality and acceptance behavior on the intended API.

## Multi-concurrency result on the active checkpoint

The active server was then tested with the same chat fixture at concurrency
1, 2, 4, and 8. Each stream requested 512 tokens with probabilistic DSpark-7;
prompts were given distinct stream suffixes and the server was warmed first.
Throughput is measured at the client over the batch window, while per-stream
tok/s excludes each stream's TTFT. The DSpark counters are deltas taken around
each batch.

| Streams | Aggregate tok/s | Median stream tok/s | Median TTFT | Accepted/draft counter ratio | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 102.8 | 103.9 | 0.052 s | 0.172 | 0 |
| 2 | 147.4 | 87.7 | 0.997 s | 0.162 | 0 |
| 4 | 266.5 | 76.0 | 0.494 s | 0.170 | 0 |
| 8 | 462.1 | 61.7 | 0.173 s | 0.178 | 0 |

The server remained healthy, used approximately 197.0 GB of 205.8 GB HBM at
the post-run high-water, and used approximately 108 GiB of 235 GiB host RAM.
This is useful serving capacity for several concurrent agents, but it is
materially below the reference checkpoint's synthetic K7 aggregate table at
the same stream counts. The abliterated checkpoint's low draft acceptance and
the different workload remain the likely causes.
