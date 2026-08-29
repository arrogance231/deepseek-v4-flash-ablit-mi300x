"""Secret-free request/response diagnostics for the OpenAI API.

This is deliberately an ASGI middleware rather than ``--enable-log-requests``:
vLLM's request logger can print prompts and tool schemas.  The middleware logs
only request shape and response lifecycle metadata.  It also converts an
unexpected stream EOF into an explicit error finish event before ``[DONE]`` so
clients never mistake a truncated stream for a successful completion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("vllm.request_diagnostics")
_EOS_TOKEN_ID = 1
_MAX_CAPTURE = 2 * 1024 * 1024


def _header(scope: dict[str, Any], name: str) -> str | None:
    wanted = name.lower().encode()
    for key, value in scope.get("headers", []):
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


def _request_summary(body: bytes) -> dict[str, Any]:
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(request, dict):
        return {}

    stop = request.get("stop")
    tools = request.get("tools")
    return {
        "model": request.get("model"),
        "stream": bool(request.get("stream", False)),
        "max_tokens": request.get("max_tokens", request.get("max_completion_tokens")),
        "min_tokens": request.get("min_tokens"),
        "stop_token_ids_count": len(request.get("stop_token_ids", []) or [])
        if isinstance(request.get("stop_token_ids", []), list)
        else None,
        "stop_strings_count": len(stop) if isinstance(stop, list) else (1 if isinstance(stop, str) else 0),
        "tools_count": len(tools) if isinstance(tools, list) else 0,
        "tool_choice": request.get("tool_choice") if isinstance(request.get("tool_choice"), str) else ("named" if isinstance(request.get("tool_choice"), dict) else None),
        "parallel_tool_calls": request.get("parallel_tool_calls"),
        "return_token_ids": bool(request.get("return_token_ids", False)),
    }


def _tool_requirements(body: bytes) -> dict[str, set[str]]:
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    result: dict[str, set[str]] = {}
    for tool in request.get("tools", []) if isinstance(request, dict) else []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        params = function.get("parameters", {})
        if isinstance(name, str) and isinstance(params, dict):
            required = params.get("required", [])
            result[name] = set(required) if isinstance(required, list) else set()
    return result


def _sse_json(event: bytes) -> dict[str, Any] | None:
    for line in event.splitlines():
        line = line.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return {"_done": True}
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return value if isinstance(value, dict) else None
    return None


def _error_finish(response_id: str, model: str | None) -> bytes:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "error",
            "stop_reason": "incomplete_stream",
        }],
    }
    return ("data: " + json.dumps(payload, separators=(",", ":")) + "\n\n").encode()


class DeepSeekDiagnosticsMiddleware:
    """Log lifecycle metadata and fail closed on an incomplete SSE stream."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        request_id = _header(scope, "x-request-id") or uuid.uuid4().hex
        started = time.monotonic()
        captured: list[bytes] = []
        captured_size = 0
        request_info: dict[str, Any] = {}
        request_tools: dict[str, set[str]] = {}
        response_status: int | None = None
        response_headers: list[tuple[bytes, bytes]] = []
        is_stream = False
        response_id = request_id
        response_model: str | None = None
        body_parts: list[bytes] = []
        sse_pending = b""
        stream_chunks = 0
        finish_reason: Any = None
        stop_reason: Any = None
        stop_token_id: Any = None
        generated_tokens: int | None = None
        observed_token_count = 0
        terminal_sse = False
        finish_event = False
        disconnected: str | None = None
        tool_args: dict[int, str] = {}
        tool_names: dict[int, str] = {}

        async def receive_wrapper():
            nonlocal captured_size, request_info, request_tools
            message = await receive()
            if message.get("type") == "http.request":
                data = message.get("body", b"") or b""
                if captured_size < _MAX_CAPTURE:
                    keep = min(len(data), _MAX_CAPTURE - captured_size)
                    captured.append(data[:keep])
                    captured_size += keep
                if not message.get("more_body", False) and not request_info:
                    raw = b"".join(captured)
                    request_info = _request_summary(raw)
                    request_tools = _tool_requirements(raw)
            return message

        def observe(value: dict[str, Any]) -> None:
            nonlocal response_id, response_model, finish_reason, stop_reason
            nonlocal stop_token_id, generated_tokens, observed_token_count
            nonlocal finish_event, terminal_sse
            if value.get("_done"):
                terminal_sse = True
                return
            response_id = value.get("id") or response_id
            response_model = value.get("model") or response_model
            usage = value.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                generated_tokens = usage["completion_tokens"]
            for choice in value.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    finish_event = True
                    finish_reason = choice.get("finish_reason")
                    stop_reason = choice.get("stop_reason")
                ids = choice.get("token_ids")
                if isinstance(ids, list) and ids:
                    observed_token_count += len(ids)
                    stop_token_id = ids[-1]
                    if generated_tokens is None:
                        generated_tokens = observed_token_count
                delta = choice.get("delta") or {}
                for call in delta.get("tool_calls", []) or []:
                    if not isinstance(call, dict):
                        continue
                    index = call.get("index", 0)
                    function = call.get("function") or {}
                    if function.get("name"):
                        tool_names[index] = function["name"]
                    arguments = function.get("arguments")
                    if arguments:
                        tool_args[index] = tool_args.get(index, "") + arguments

        def transform_event(event: bytes) -> bytes:
            value = _sse_json(event)
            if value is not None:
                observe(value)
            return event

        async def send_wrapper(message: dict[str, Any]):
            nonlocal response_status, response_headers, is_stream, sse_pending
            nonlocal stream_chunks, disconnected, terminal_sse
            if message.get("type") == "http.response.start":
                response_status = message.get("status")
                response_headers = list(message.get("headers", []))
                content_type = dict(response_headers).get(b"content-type", b"").lower()
                is_stream = b"text/event-stream" in content_type
                if is_stream:
                    await send(message)
                return

            if message.get("type") != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"") or b""
            if not is_stream:
                body_parts.append(body)
                if message.get("more_body", False):
                    return
                raw = b"".join(body_parts)
                try:
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        observe(value)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                await send({"type": "http.response.start", "status": response_status or 200, "headers": response_headers})
                await send({"type": "http.response.body", "body": raw, "more_body": False})
                return

            sse_pending += body
            output: list[bytes] = []
            while b"\n\n" in sse_pending:
                event, sse_pending = sse_pending.split(b"\n\n", 1)
                if _sse_json(event) == {"_done": True}:
                    terminal_sse = True
                    if not finish_event:
                        output.append(_error_finish(response_id, response_model))
                    output.append(event + b"\n\n")
                else:
                    output.append(transform_event(event) + b"\n\n")
                stream_chunks += 1
            if message.get("more_body", False):
                if output:
                    await send({"type": "http.response.body", "body": b"".join(output), "more_body": True})
                return

            if sse_pending.strip():
                output.append(transform_event(sse_pending) + b"\n\n")
                stream_chunks += 1
                sse_pending = b""
            if not terminal_sse:
                if not finish_event:
                    output.append(_error_finish(response_id, response_model))
                output.append(b"data: [DONE]\n\n")
            await send({"type": "http.response.body", "body": b"".join(output), "more_body": False})

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except BaseException as exc:
            disconnected = f"{type(exc).__name__}: {exc}"[:160]
            if is_stream and response_status is not None and not terminal_sse:
                try:
                    await send({
                        "type": "http.response.body",
                        "body": _error_finish(response_id, response_model) + b"data: [DONE]\n\n",
                        "more_body": False,
                    })
                    terminal_sse = True
                except BaseException:
                    # A real client disconnect cannot receive a repair event.
                    pass
            raise
        finally:
            if not request_info:
                request_info = _request_summary(b"".join(captured))
                request_tools = _tool_requirements(b"".join(captured))
            if generated_tokens is None and body_parts and not is_stream:
                generated_tokens = None
            eos_model = stop_token_id == _EOS_TOKEN_ID if stop_token_id is not None else None
            if eos_model is True:
                eos_source = "model_eos"
            elif finish_reason == "length":
                eos_source = "server_length"
            elif request_info.get("stop_strings_count", 0):
                eos_source = "requested_stop"
            elif finish_reason == "stop" and generated_tokens is not None and (
                request_info.get("max_tokens") is None
                or generated_tokens < request_info["max_tokens"]
            ):
                eos_source = "likely_model_eos"
            else:
                eos_source = "unknown"
            server_forced = finish_reason == "length" or stop_reason not in (None, _EOS_TOKEN_ID)
            logger.info(
                "request_complete request_id=%s path=%s http_status=%s model=%s "
                "requested_max_tokens=%s min_tokens=%s stop_token_ids_count=%s "
                "stop_strings_count=%s finish_reason=%s stop_reason=%s "
                "stop_token_id=%s generated_token_count=%s eos_model_generated=%s "
                "eos_source=%s server_forced=%s stream=%s stream_chunk_count=%s "
                "finish_event=%s terminal_sse=%s disconnected=%s duration_ms=%d "
                "tools_count=%s tool_digest=%s",
                request_id,
                scope.get("path"),
                response_status,
                request_info.get("model"),
                request_info.get("max_tokens"),
                request_info.get("min_tokens"),
                request_info.get("stop_token_ids_count"),
                request_info.get("stop_strings_count"),
                finish_reason,
                stop_reason,
                stop_token_id,
                generated_tokens,
                eos_model,
                eos_source,
                server_forced,
                is_stream,
                stream_chunks,
                finish_event,
                terminal_sse,
                disconnected,
                int((time.monotonic() - started) * 1000),
                request_info.get("tools_count"),
                hashlib.sha256(json.dumps(sorted(request_tools), separators=(",", ":")).encode()).hexdigest()[:12] if request_tools else None,
            )
