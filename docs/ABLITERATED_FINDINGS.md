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
