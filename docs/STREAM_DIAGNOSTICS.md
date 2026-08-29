# DeepSeek V4 stream diagnostics

## Diagnosis

The endpoint is running vLLM V1, not SGLang. Its effective generation settings
are:

```text
model config eos_token_id: 1 (<｜end▁of▁sentence｜>)
request default stop_token_ids: EOS is added by SamplingParams
request default stop strings: none
request default min_tokens: 0
request default ignore_eos: false
server generation config: vllm (no model generation_config stop override)
max_model_len: 524288
max_num_seqs: 16
```

A controlled request with `return_token_ids=true` ended with token ID `1` and
`finish_reason=stop`; the prompt ended with `</think>`, not EOS. This proves that
ordinary short `stop` responses are model-generated EOS, not a Caddy or chat
-template-injected stop. `ignore_eos=true` changed the same request to
`finish_reason=length` at the requested cap, while `min_tokens=128` delayed EOS
to token 130. Neither option is enabled globally: forcing a minimum on every
request would turn legitimate short answers into repetition.

The model's tokenizer and encoding tests pass all four checkpoint cases. The
vLLM tokenizer renders the expected V4 DSML tool instructions and assistant
boundary. Native one- and two-tool requests return complete OpenAI
`message.tool_calls`; no ordinary-text tool parser is used.

The DSpark A/B probe produced the same tool-call metadata and complete streams
with `DISABLE_DSPARK=0` and `1`. The three-request probe completed without OOM,
engine errors, or KV exhaustion. The server reports 1,549,376 GPU KV tokens and
2.96x theoretical concurrency at a full 524K request; `MAX_NUM_SEQS=64` is a
scheduler ceiling, not 16 simultaneous full-context allocations. Short probes
used little KV capacity.

Caddy uses `flush_interval -1`, has no configured stream read/write timeout,
and forwards vLLM's `text/event-stream` response. HTTP/1.1 and HTTP/2 probes
both delivered the terminal finish chunk and `[DONE]`. Historical Caddy
`aborting with incomplete response ... context canceled` records were client
cancellations (large requests canceled within about one second), not proxy
idle timeouts. Access logging is enabled without request headers containing the
API key in the log payload (Caddy redacts authorization).

## Fixes

1. `patches/deepseek_v4_hermes_fallback.py` now rejects incomplete native DSML
   and Hermes XML envelopes in non-streaming responses. It also refuses the
   parser's end-of-stream flush for an incomplete streaming envelope. A
   length-truncated request previously returned a tool call such as
   `name="ex", arguments={}`; it now returns `finish_reason="length"` with no
   executable tool call.
2. `patches/request_diagnostics.py` is mounted as vLLM ASGI middleware. It logs
   request ID, model, requested limits, stop metadata, token count, stream
   chunk count, finish/terminal status, and disconnect reason. It never logs
   prompts, tool arguments, or credentials. If the application ends an SSE
   response without a finish event, the middleware emits an `error` finish with
   `stop_reason="incomplete_stream"` before `[DONE]`, allowing the client to
   retry instead of treating a partial answer as a normal stop.
3. `scripts/test_stream_integrity.py` covers plain streaming, native tool JSON,
   truncation rejection, and three concurrent requests. It is safe to run
   against either localhost or the authenticated public `/v1` endpoint.

## Hermes custom-endpoint verification

The deployed public endpoint is the endpoint reached by the Hermes-compatible
OpenAI client: Caddy access records show `POST /v1/chat/completions` requests
with the OpenAI Python client headers, and the corresponding vLLM records show
`model=deepseek-v4-flash`, the configured tool-parser middleware, and
`finish_event=true terminal_sse=true disconnected=None` for streamed calls.
The mounted middleware and parser patches were loaded at the current container
startup and are active in those requests.

`custom` and `vllm-story` are client-side provider labels; neither is included
in the OpenAI-compatible request, so the server cannot distinguish those labels
from wire data alone. Correlate the client's `X-Request-ID` with
`request_complete` when attributing a particular session. The observed short
parent-like streamed responses ended with `finish_reason=stop` and
`eos_source=likely_model_eos`, not an incomplete SSE response. Therefore the
provider/runtime fix is active, but it correctly does not turn a legitimate
model EOS into a continuation; no generic continuation heuristic is warranted.

## Regression metadata

```text
plain stream, max_tokens=256: finish=length, completion_tokens=256, [DONE]=yes
native execute_code, max_tokens=128: finish=stop, completion_tokens=46, 1 tool call
truncated DSML, max_tokens=16: finish=length, tool_calls=none
HTTP/2 public stream: HTTP 200, text/event-stream, finish chunk=yes, [DONE]=yes
```

The endpoint now also normalizes DeepSeek V4 requests that omit
`chat_template_kwargs`: it selects the reasoning template by default and
respects explicit `thinking=false` / `reasoning_effort=none`. This is a
model-specific provider compatibility fix, not a continuation heuristic.
The normalizer is secret-free and records whether it injected the template
selection in `request_complete` diagnostics.

The remaining short `finish_reason=stop` cases after this normalization are
still model EOS by design. For a caller that explicitly requires a
minimum-length prose continuation, use a request-scoped `min_tokens` (or a
continuation request); do not set `ignore_eos` or a global minimum for tool
calls.
