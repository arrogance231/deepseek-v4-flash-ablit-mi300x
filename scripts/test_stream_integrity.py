#!/usr/bin/env python3
"""Small OpenAI-compatible regression probe for vLLM stream/tool integrity.

Usage:
  BASE_URL=http://127.0.0.1:8000/v1 API_KEY=... ./scripts/test_stream_integrity.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
API_KEY = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY", "")


def post(payload: dict, stream: bool = False) -> tuple[dict, list[dict], bool]:
    headers = {"Content-Type": "application/json", "X-Request-ID": str(uuid.uuid4())}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    req = Request(BASE_URL + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    with urlopen(req, timeout=300) as response:
        if not stream:
            return json.load(response), [], True
        events: list[dict] = []
        pending = b""
        done = False
        while True:
            chunk = response.read(16384)
            if not chunk:
                break
            pending += chunk
            while b"\n\n" in pending:
                event, pending = pending.split(b"\n\n", 1)
                for line in event.splitlines():
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                    else:
                        events.append(json.loads(data))
        if pending.strip():
            raise AssertionError("stream ended with a partial SSE event")
        return {}, events, done


def assert_stream(name: str, payload: dict) -> None:
    _, events, done = post({**payload, "stream": True, "stream_options": {"include_usage": True}}, True)
    choices = [c for event in events for c in event.get("choices", [])]
    terminal = [c for c in choices if c.get("finish_reason") is not None]
    assert terminal, f"{name}: missing terminal finish_reason"
    assert done, f"{name}: missing [DONE]"
    print(f"PASS {name}: chunks={len(events)} finish={terminal[-1]['finish_reason']}")


def main() -> int:
    base = {"model": "windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated", "temperature": 0.0}
    assert_stream("plain", {**base, "messages": [{"role": "user", "content": "Explain TCP retransmission in detail."}], "max_tokens": 256})

    tool = {"type": "function", "function": {"name": "execute_code", "description": "run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
    response, _, _ = post({**base, "messages": [{"role": "user", "content": "Run print(123) with execute_code."}], "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "execute_code"}}, "max_tokens": 128})
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "execute_code"
    json.loads(calls[0]["function"]["arguments"])
    print("PASS native tool: complete JSON arguments")

    # Regression: ParserEngine.finish() must not turn a length-truncated DSML
    # envelope into an executable tool call.
    response, _, _ = post({**base, "messages": [{"role": "user", "content": "Run print(123) with execute_code."}], "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "execute_code"}}, "max_tokens": 16})
    message = response["choices"][0]["message"]
    assert not message.get("tool_calls"), "truncated tool call was accepted"
    assert response["choices"][0]["finish_reason"] == "length"
    print("PASS truncated tool: rejected safely")

    concurrent = {**base, "messages": [{"role": "user", "content": "Write a detailed explanation of process scheduling."}], "max_tokens": 256}
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: post(concurrent)[0], range(3)))
    print("PASS concurrency: 3 requests completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
