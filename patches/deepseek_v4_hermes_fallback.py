# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING, Sequence

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

# Some Hermes/local-model prompts make the model emit tool-specific XML
# wrappers instead of the V4 DSML block. Keep native DSML as the primary path,
# but normalize the observed Hermes wrappers when the corresponding tool was
# supplied in the OpenAI request. This is schema-gated: ordinary assistant
# text is never promoted to a tool call unless that tool was offered.
_HERMES_FALLBACK_OPENERS = (
    ("<execute_code>", "execute_code"),
    ("<write_file>", "write_file"),
    ("<write-files>", "write_file"),
)
_HERMES_FALLBACK_CLOSERS = {
    "execute_code": ("</execute_code>",),
    "write_file": ("</write_file>", "</write-files>"),
}


def _dsml_string_parameter(name: str, value: str) -> str:
    return (
        f'<{_DSML}parameter name="{name}" string="true">{value}'
        f"</{_DSML}parameter>"
    )


def _write_file_wrapper_to_dsml(raw_body: str) -> str | None:
    """Convert Hermes' path/content sequence into one or more V4 calls."""
    pairs = re.findall(
        r"<path>\s*(.*?)\s*</path>\s*<content>(.*?)</content>",
        raw_body,
        flags=re.DOTALL,
    )
    if not pairs:
        return None

    calls = []
    for path, content in pairs:
        calls.append(
            f'<{_DSML}invoke name="write_file">\n'
            f"{_dsml_string_parameter('path', path)}\n"
            f"{_dsml_string_parameter('content', content)}\n"
            f"</{_DSML}invoke>"
        )
    return f"<{_DSML}tool_calls>\n" + "\n".join(calls) + f"\n</{_DSML}tool_calls>"


def _execute_code_wrapper_to_dsml(code: str) -> str:
    return (
        f"<{_DSML}tool_calls>\n"
        f'<{_DSML}invoke name="execute_code">\n'
        f"{_dsml_string_parameter('code', code)}\n"
        f"</{_DSML}invoke>\n"
        f"</{_DSML}tool_calls>"
    )

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

_ESCAPED_DSML = re.escape(_DSML)
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML}parameter>",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        self._hermes_tag_mode = "search"
        self._hermes_tag_buffer = ""
        self._hermes_flush_pending = False
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = (
            bool(chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking"))
            and chat_kwargs.get("reasoning_effort") != "none"
        )
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._hermes_tag_mode = "search"
        self._hermes_tag_buffer = ""
        self._hermes_tag_start = ""
        self._hermes_tag_tool = ""
        self._hermes_flush_pending = False

    def _has_hermes_fallback_tool(self, name: str) -> bool:
        return bool(
            self._tools and find_tool_properties(self._tools, name) is not None
        )

    def _fallback_openers(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (start, name)
            for start, name in _HERMES_FALLBACK_OPENERS
            if self._has_hermes_fallback_tool(name)
        )

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        """Normalize Hermes XML wrappers to native V4 DSML.

        Native DSML remains the normal path. This compatibility path buffers
        only enough text to recognize tags split across streaming deltas and
        is enabled only for tools present in the OpenAI request.
        """
        if self._suppress_tool_calls:
            return delta_text, delta_token_ids

        openers = self._fallback_openers()
        if not openers:
            return delta_text, delta_token_ids

        self._hermes_tag_buffer += delta_text
        output: list[str] = []
        max_start_len = max(len(start) for start, _ in openers)

        while True:
            if self._hermes_tag_mode == "search":
                matches = [
                    (self._hermes_tag_buffer.find(start), start, name)
                    for start, name in openers
                    if start in self._hermes_tag_buffer
                ]
                matches = [m for m in matches if m[0] >= 0]
                if not matches:
                    keep = max_start_len - 1
                    if len(self._hermes_tag_buffer) > keep:
                        output.append(self._hermes_tag_buffer[:-keep])
                        self._hermes_tag_buffer = self._hermes_tag_buffer[-keep:]
                    break
                pos, start, name = min(matches, key=lambda item: item[0])
                output.append(self._hermes_tag_buffer[:pos])
                self._hermes_tag_buffer = self._hermes_tag_buffer[pos + len(start) :]
                self._hermes_tag_mode = name
                self._hermes_tag_start = start
                self._hermes_tag_tool = name
                continue

            closers = _HERMES_FALLBACK_CLOSERS[self._hermes_tag_tool]
            matches = [
                (self._hermes_tag_buffer.find(close), close)
                for close in closers
                if close in self._hermes_tag_buffer
            ]
            matches = [m for m in matches if m[0] >= 0]
            if not matches:
                keep = max(len(close) for close in closers) - 1
                if len(self._hermes_tag_buffer) > keep:
                    output.append(self._hermes_tag_buffer[:-keep])
                    self._hermes_tag_buffer = self._hermes_tag_buffer[-keep:]
                break

            pos, close = min(matches, key=lambda item: item[0])
            body = self._hermes_tag_buffer[:pos]
            self._hermes_tag_buffer = self._hermes_tag_buffer[pos + len(close) :]
            if self._hermes_tag_tool == "execute_code":
                output.append(_execute_code_wrapper_to_dsml(body))
            else:
                converted = _write_file_wrapper_to_dsml(body)
                if converted is None:
                    output.append(self._hermes_tag_start + body + close)
                else:
                    output.append(converted)
            self._hermes_tag_mode = "search"
            self._hermes_tag_start = ""
            self._hermes_tag_tool = ""

        if self._hermes_flush_pending and self._hermes_tag_buffer:
            output.append(self._hermes_tag_buffer)
            self._hermes_tag_buffer = ""

        # Synthesized DSML was not present in the model token stream, so text
        # matching must be used for this feed rather than the original IDs.
        return "".join(output), []

    def parse_delta(self, *args, **kwargs):
        self._hermes_flush_pending = bool(kwargs.get("finished", False))
        try:
            return super().parse_delta(*args, **kwargs)
        finally:
            self._hermes_flush_pending = False

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
