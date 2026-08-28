# Findings on the abliterated checkpoint

This note separates measurements made with the abliterated checkpoint from
the upstream reference measurements. That distinction is important: the
reference project measured the original `deepseek-ai/DeepSeek-V4-Flash-0731`
weights, while this deployment uses
`lovesenko/DeepSeek-V4-Flash-0731-Abliterated`.

## Executive result

The abliterated checkpoint runs successfully on one MI300X with the pinned
ROCm/vLLM stack. The best tested single-stream result is **115.6 tok/s mean**
(117.0 median) with probabilistic static DSpark-K=5 in the matched normal-chat
sweep, versus the reference project's **152.6 tok/s** on the original
checkpoint's synthetic workload. The difference is real for this checkpoint;
it is not evidence that the MI300X stack failed to load or that the reference
number is fabricated.

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
- static DSpark speculative decoding (K=5 default; K=5–7 tested);
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
| DSpark K=5 decode | 115.6 tok/s mean / 117.0 median | Three 512-token streamed normal-chat requests |
| DSpark K=6 decode | 113.9 tok/s mean / 112.4 median | Same matched fixture |
| DSpark K=7 decode | 107.1 tok/s mean / 106.6 median | Same matched fixture; control |
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
variant. The checkpoint declares a DSpark block size of five; the current ROCm
path supports the tested static range K=5–7, while lower values are not
supported.

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
   endpoint with identical seeds and prompts. **Completed:** K=5 is the
   measured winner and is now the default; K=6 and K=7 remain selectable for
   A/B runs. K below the declared block size of five remains out of scope for
   the current ROCm path.
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

## Phase 2 — static-K sweep on normal chat

The server was restarted with each static value through the same vLLM image,
checkpoint, overlays, AITER settings, KV tiers, sampler, prompt, seed, and
512-token output limit. Each setting received three warmed streaming chat
requests. The text prefixes were inspected for coherence, dialogue
attribution, and prompt-format leakage; all three settings passed this smoke
quality gate.

| Static K | Run decode tok/s | Mean | Median | Draft tokens | Accepted tokens | Accepted/draft | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 118.7, 111.1, 117.0 | **115.6** | **117.0** | 3,260 | 883 | **27.1%** | **PROMOTE** |
| 6 | 112.4, 111.0, 118.2 | 113.9 | 112.4 | 3,846 | 894 | 23.2% | Keep selectable |
| 7 | 106.6, 105.2, 109.5 | 107.1 | 106.6 | ~4,788 | ~851 | ~17.8% | Control only |

K=5 is approximately 8.0% faster than the K=7 control in this matched
normal-chat workload and had the strongest draft acceptance. The generated
samples remained ordinary prose rather than the repeated, prompt-leaking
artifact observed in the rejected base-MTP raw-completion experiment. This
is a throughput/quality smoke gate, not a claim of full creative-quality
equivalence; Phase 4 still owns long-output and structured-output validation.

`compose.yaml` now defaults to K=5 and accepts
`DS_NUM_SPECULATIVE_TOKENS=5|6|7` for reversible comparisons. Changing K
requires a container restart and graph recapture; the checkpoint and weight
files are unchanged.

## Phase 3 — target-generated Markov-head calibration (completed: rejected)

Phase 3A tested the smallest reversible draft-path intervention: the frozen
abliterated target generated 16 original prose completions (3,072 output
tokens, fixed seeds), and a standalone trainer updated only the two
`mtp.2.markov_head` BF16 matrices. The three MTP decoder blocks, all target
layers, tokenizer, and config remained unchanged. The calibration corpus is
deliberately an experiment artifact under `results/raw/` and is not required
to reproduce the server.

The offline held-out transition cross-entropy improved from **9.0773** to
**7.0171**, but that metric is not a target-model quality metric. The required
live A/B used the identical chat prompt, sampler, seed, K=5, and four 512-token
requests per checkpoint (first request treated as warm-up):

| Variant | Warm-up-excluded mean tok/s | Median tok/s | Accepted/draft | Quality gate |
| --- | ---: | ---: | ---: | --- |
| Original abliterated MTP | **119.06** | 116.97 | **29.24%** | Pass |
| Markov-calibrated sidecar | 114.52 | 115.71 | 27.38% | Pass, slower |

As a separate control, the untouched checkpoint was restarted with the
speculative flags removed by `DISABLE_DSPARK=1`. The no-draft path averaged
**68.38 tok/s** after warm-up (55.62 tok/s on its first request), generated no
draft tokens, and remained coherent. DSpark K=5 therefore adds about **74.1%**
steady decode throughput on this fixture; the gain comes from accepted target
tokens, not from changing the target weights. The production service is back
on DSpark K=5 (`DISABLE_DSPARK=0`).

The sidecar is therefore **rejected**: it is coherent but loses 3.8% of live
decode throughput and draft acceptance. The production container was restored
to `/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated`; the candidate is
kept at `/mnt/model-storage/DeepSeek-V4-Flash-0731-Ablit-MarkovCalibrated` for
future experiments and is never selected by default.

Reproduce the pilot (inside the pinned vLLM image; stop inference while
training so the GPU is free):

```bash
python3 scripts/collect_mtp_calibration_data.py --limit 16 --max-tokens 192
docker run --rm --device=/dev/kfd --device=/dev/dri --ipc=host \
  --entrypoint python -v "$PWD":/work \
  -v /mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated:/models/deepseek:ro \
  vllm/vllm-openai-rocm@sha256:e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a \
  /work/scripts/train_markov_calibrator.py --checkpoint /models/deepseek \
  --data /work/results/raw/phase3-mtp-calibration.jsonl \
  --output /work/results/raw/phase3-markov-calibrated.safetensors
```

`export_markov_variant.py` creates a hard-linked/symlinked sidecar and
`verify_markov_variant.py` confirms that exactly two tensors changed.
`phase3_ab.py` records client throughput and vLLM DSpark counters. Full
hidden-state MTP distillation remains research work; the serving API does not
expose the target hidden states, so this phase does not claim to have trained
the three MTP decoder blocks.

## Phase 4 — quality and stability gate (completed)

The production checkpoint was exercised with the same K=5/AITER/FP8-KV
profile after the Phase 3 rollback. The gate uses
`scripts/phase4_quality_gate.py`; generated text and JSON reports remain local
under `results/raw/phase4-quality*` and are ignored by Git.

| Gate | Result | Evidence |
| --- | --- | --- |
| 2K natural prose | **PASS** | 2,000 tokens, 104.44 tok/s, no format leakage or obvious loop |
| Forced 5K endurance | **PASS** | 5,000 tokens, 113.91 tok/s, 0.84% repeated 8-gram excess, no 32-token chunk loop |
| Forced 10K endurance | **FAIL / DEGRADED** | 10,000 tokens, 156.78 tok/s, 43.91% repeated 8-gram excess and 42.14% repeated 12-gram excess; tail collapses into an `and ...` loop |
| Natural 5K/10K request | **LIMITED** | Both stopped naturally around 1.6K tokens (`finish_reason=stop`) without leakage or loops |
| Fixed seed, temperature 0 | **FAIL for exact replay** | Two identical requests differed under probabilistic DSpark drafting |
| Structured JSON | **PASS** | Valid object with required `title`, `mood`, and three `beats` |
| Forced tool call | **PASS** | Valid `lookup_weather` call with JSON argument `{"city":"Kyoto"}` |
| Runtime stability | **PASS** | Container healthy after 5K and 10K runs; no OOM, NaN, or ROCm crash |

The practical conclusion is that K=5 remains a good **single-scene/short
chapter** profile and can sustain at least 5K forced tokens on this fixture.
Forcing 10K tokens is not production-safe: the model enters a severe semantic
loop even though the runtime remains healthy. Requests that need exact replay
must use a deterministic drafting profile (greedy draft or DSpark disabled);
the default probabilistic DSpark profile is intentionally optimized for
throughput and does not guarantee byte-identical seeded text.
