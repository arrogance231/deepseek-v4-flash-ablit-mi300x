# DeepSeek V4 Flash DSpark Abliterated on MI300X

This is a focused serving adaptation for the gated, abliterated
[`windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated`](https://huggingface.co/windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated)
checkpoint on one AMD Instinct MI300X. The repository records the reproducible
runtime configuration, ROCm overlays, and operational safeguards. It is not a
benchmark claim for the upstream weights.

The upstream [`ryanzhou/deepseek-v4-flash-mi300x`](https://github.com/ryanzhou/deepseek-v4-flash-mi300x)
project is acknowledged as the starting reference. This repository's
checkpoint adaptation, test results, and operational decisions are maintained
by **arrogance231**.

## Current answer

The tested checkpoint is configured on a single `gfx942` MI300X with the local
ROCm image (`vLLM 0.27.1+rocm723`). The operational profile is:

| Setting | Value |
| --- | --- |
| Checkpoint | `windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated` |
| Revision | `6de83db0be050e0338ae2f8376440642203ad90d` |
| Context limit | `900,000` configured (not yet validated on this MI300X) |
| Runtime | vLLM V1 ROCm `0.27.1+rocm723` |
| Weights | Original FP8/MXFP4 checkpoint, no weight offload |
| KV | `fp8_ds_mla`, 16 GB HBM pool + 96 GiB native CPU tier |
| Drafting | DSpark, probabilistic, K=7 |
| Scheduling | paged KV, chunked prefill, prefix caching, up to 64 sequences |
| Kernel path | AITER and the repository's `gfx942` overlays |

When Caddy is configured, the public OpenAI-compatible endpoint is:

```text
https://<your-public-host>/v1
```

It is protected by the API key configured outside Git. Do not put that key in
this repository.

## Measurements from the local abliterated checkpoint

These are measurements from the exact checkpoint and recipe adaptation used in
this deployment, not copied from the reference project's official-model table:

| Probe | Result |
| --- | ---: |
| Target-only DSpark K=5 | ~32.4 tok/s on 600-token generation |
| Recipe DSpark K=7 cold | 200.6 tok/s on 600 tokens |
| Recipe DSpark K=7 warm | 280.6 / 280.3 tok/s on 600 tokens |
| Warm tool calls | ~161–171 tok/s end-to-end |
| EOS 1–500 | 20/20 reached 500; 0 premature EOS |
| Required tool calls | 20/20 parsed correctly |
| No-tool DSML leakage | 0/20 |
| HBM high-water | ~197–200 GB of 205.8 GB |

The model loads about 156.47 GiB of HBM for weights. The observed long-probe
high-water was about 199.9 GB of 205.8 GB, so the cache and graph settings are
intentionally conservative. A model-declared 1M position limit is not treated
as a validated 1M narrative service.

The active recipe uses probabilistic DSpark K=7; K=5 and K=6 remain useful
A/B controls, while disabling DSpark gives a measured warm decode rate of
68.38 tokens/s from the earlier control. Forced single responses beyond roughly
5K completion tokens
can enter a repetition loop; continue long chapters through explicit stateful
requests instead of using `ignore_eos`.

The full checkpoint-specific evidence is in
[`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md), including the
rejected MTP restoration and Markov-head calibration experiments.

## Why this repository is different

The central experiment here is a weight-edited checkpoint, so the important
question is not whether DeepSeek V4 can run on MI300X in general. It is whether
the edit preserves target/draft agreement, output correctness, and useful
long-form behavior. The repository therefore keeps three things separate:

1. **Serving machinery** — ROCm overlays, AITER settings, custom `gfx942`
   kernels, paged KV, CPU-tier fencing, and scheduler behavior.
2. **Checkpoint evidence** — measurements made after swapping in the
   abliterated weights, with the same prompts and runtime knobs.
3. **Rejected paths** — sidecars and sampler variants remain reproducible but
   are not silently promoted to the live profile.

The result is intentionally narrower than a general DeepSeek deployment
guide. It is an auditable answer for this checkpoint on this card.

## Repository map

```text
compose.yaml              vLLM service and pinned runtime mounts
vllm-entrypoint.sh        runtime setup and optional DSpark disable switch
configs/                   frozen production environment profile
scripts/download_model.sh resumable, revision-pinned model download
scripts/check_production_profile.sh health/configuration gate
patches/                   read-only vLLM/AITER overlays with provenance
                          plus Hermes XML-to-DSML compatibility parser
kernel-dev/hip-a8w4/       JIT-built MI300X kernels
tuning/                    measured AITER tuning tables
docs/ABLITERATED_FINDINGS.md
                          checkpoint-specific results and open questions
docs/MEASUREMENTS-LOCAL-ABLITERATED.md
                          measured 200–280 tok/s profile and correctness gate
docs/PRODUCTION_PROFILE.md
                          start, rollback, and generation policy
```

The large checkpoint, runtime cache, raw traces, and local secrets are ignored
by Git. `MODEL_LICENSES.md` describes model provenance; the Apache-2.0 license
in this repository applies to original repository code and configuration only.

## Reproduce the deployment

### 1. Prepare the host

Use Docker with `/dev/kfd` and `/dev/dri` available, a working ROCm driver,
and enough disk for the roughly 156 GiB checkpoint. One MI300X with about
235 GiB host RAM is the tested machine shape.

### 2. Download the exact weights

```bash
cd deepseek-v4-flash-ablit-mi300x
export MODEL_ID=windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated
export MODEL_REVISION=6de83db0be050e0338ae2f8376440642203ad90d
export MODEL_DIR=/mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated
export HF_TOKEN=hf_your_token_here
./scripts/download_model.sh
```

The script uses the pinned revision and a resumable Hugging Face download. It
does not place weights in Git.

### 3. Prepare and start

```bash
./prepare-artifacts.sh
cp .env.example .env
# Set MODEL_DIR, VLLM_API_KEY, and any local paths in .env.
docker compose up -d inference
docker compose logs -f inference
```

The first start builds or loads the MI300X kernels and captures the configured
graphs; a cold start can take several minutes. The tracked defaults include:

```text
MAX_MODEL_LEN=393216
MAX_NUM_SEQS=64
DS_NUM_SPECULATIVE_TOKENS=7
DISABLE_DSPARK=0
```

The host's existing Caddy service owns ports 80/443 and forwards the
authenticated public hostname to the loopback-only vLLM port. Do not start the
Compose `proxy` profile on that host unless the host-level proxy is moved.

### 4. Check the service

```bash
./scripts/check_production_profile.sh
curl -sS https://<your-public-host>/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

For a minimal chat request:

```bash
curl -sS https://<your-public-host>/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated",
       "messages":[{"role":"user","content":"Write a short scene."}],
       "temperature":0.8,"max_tokens":256}'
```

For Pi or another client, place stable story material first (system rules,
world state, character state, then recent context and the changing request) so
automatic prefix caching can reuse it.

For a secret-free request/stream diagnostic (plain response, native tool call,
truncation rejection, and three concurrent requests), run:

```bash
BASE_URL=https://<your-public-host>/v1 API_KEY="$VLLM_API_KEY" \
  ./scripts/test_stream_integrity.py
```

The server logs one `request_complete` record per POST. It includes request ID,
finish/stop metadata, generated-token count when usage is available, stream
chunk/terminal status, and disconnect reason; prompts, arguments, and API keys
are never logged.

## Runtime details worth preserving

- The `fp8_ds_mla` cache is the DeepSeek-specific block-scaled format; it is
  not interchangeable with a generic unscaled FP8 cache.
- The CPU tier is an opportunistic cache, not extra scheduler capacity. Keep
  the load-order fencing overlay mounted when native KV offload is enabled.
- The custom overlays are mounted read-only over the digest-pinned image. Use
  `patches/README.md` and `SHA256SUMS` to audit or regenerate them.
- `DISABLE_DSPARK=1` is the correctness/replay control. It is slower but useful
  when diagnosing draft acceptance or seed behavior.
- Native V4 tool calling uses `--tool-call-parser deepseek_v4` together with
  `--enable-auto-tool-choice` and `--reasoning-parser deepseek_v4`. The mounted
  `patches/deepseek_v4_hermes_fallback.py` additionally converts legacy Hermes
  `<execute_code>` and `<write_file>`/`<write-files>` wrappers into standard
  OpenAI `message.tool_calls` when those tools are present in the request. A
  truncated DSML/XML envelope is rejected rather than returned as an executable
  tool call.
- `patches/request_diagnostics.py` is mounted as vLLM middleware. It does not
  enable vLLM prompt logging; it records lifecycle metadata and fails closed
  with an explicit `error` finish before `[DONE]` if an upstream stream ends
  without a terminal finish event.
- The public key is intentionally absent from `.env.example`, Git history, and
  all benchmark artifacts.

## What is not claimed

This repository does not claim that the current windowsxp811203 checkpoint matches
the historical checkpoint's DSpark acceptance rate, that 1M context is
production-ready, or that long-form prose quality is unchanged after
abliteration. It also does not claim that a single tok/s number transfers
between synthetic and normal chat workloads. Those questions need matched
quality fixtures and remain open; the current operational configuration is a
configured 900K profile, not a claim of full 900K narrative validation. Validate memory, recall, latency, and stability before production.

## Further reading

- [`docs/ABLITERATED_FINDINGS.md`](docs/ABLITERATED_FINDINGS.md) — measurements,
  phase results, and rejected experiments.
- [`docs/PRODUCTION_PROFILE.md`](docs/PRODUCTION_PROFILE.md) — the promoted K=7
  profile and rollback procedure.
- [`docs/STREAM_DIAGNOSTICS.md`](docs/STREAM_DIAGNOSTICS.md) — termination
  diagnosis, stream safeguards, and regression metadata.
- [`MODEL_LICENSES.md`](MODEL_LICENSES.md) — checkpoint and upstream licenses.
- [`patches/README.md`](patches/README.md) — overlay lineage and regeneration.
- [DeepSeek V4 Flash DSpark model card](https://huggingface.co/windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated)
  — checkpoint-specific usage and provenance.
