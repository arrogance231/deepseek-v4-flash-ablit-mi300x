# Promoted production profile

This profile serves the exact windowsxp811203 DSpark abliterated checkpoint with the
existing MI300X overlays. The rejected Markov sidecar is not part of this
profile.

```text
model: /mnt/model-storage/DeepSeek-V4-Flash-0731-Abliterated
revision: 6de83db0be050e0338ae2f8376440642203ad90d
max model length: 524,288 tokens
weights: original checkpoint (FP8/MXFP4 as downloaded)
DSpark: enabled, probabilistic, K=7
KV: fp8_ds_mla, 16 GB GPU pool + 96 GiB native CPU tier
AITER: enabled
prefix caching: enabled
chunked prefill: enabled
MTP/draft sidecar: original checkpoint tensors
max concurrent sequences: 64
OpenAI tools: native `deepseek_v4` parser plus Hermes XML compatibility fallback
```

The profile is encoded in [`configs/production-k7.env`](../configs/production-k7.env).
Start or recreate it with:

```bash
cd deepseek-v4-flash-ablit-mi300x
set -a; source configs/production-k7.env; set +a
docker compose up -d inference
./scripts/check_production_profile.sh
```

The first start after a recreate performs model loading, kernel warm-up, and
graph capture. A healthy start can take several minutes. The verification
script checks the mounted checkpoint, DSpark K, `DISABLE_DSPARK`, API health,
and model discovery. Tool clients must use `/v1/chat/completions` and include
the OpenAI `tools` array. Run `scripts/test_stream_integrity.py` after a
recreate to verify finish events, `[DONE]`, native tool JSON, truncation
rejection, and a three-request concurrency probe.

## Generation policy

Phase 4 passed ordinary 2K prose and forced 5K output without an obvious
loop. A forced 10K response collapsed into a severe semantic repetition loop,
so callers should cap ordinary single responses at approximately **5,000
completion tokens** and continue a chapter through explicit state/continuation
requests. Do not set `ignore_eos` for production prose.

The default probabilistic K=7 draft path is throughput-oriented and does not
guarantee byte-identical output for repeated seeded requests. For exact replay
or debugging, restart with `DISABLE_DSPARK=1`; the measured warm decode rate
was measured with the earlier K=5 profile; benchmark K=7 separately before
using it for throughput comparisons. The server's secret-free
`request_complete` logs distinguish model EOS, length limits, stream loss, and
client disconnects.

## Rollback

The promotion is reversible. Keep `configs/production-k7.env` pinned to the
checkpoint above. The previous checkpoint remains isolated at
`/mnt/model-storage/DeepSeek-V4-Flash-0731-Ablit-MarkovCalibrated` and should
only be selected explicitly for A/B work. To disable speculative decoding for
a control run:

```bash
set -a; source configs/production-k7.env; set +a
export DISABLE_DSPARK=1
docker compose up -d inference
```

Restore DSpark by re-sourcing the profile and recreating the container.
