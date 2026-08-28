#!/usr/bin/env python3
"""Run the Phase-4 quality/stability gate against the live OpenAI endpoint.

This is intentionally a lightweight engineering gate, not an automated claim
that a judge model has proved prose quality.  It exercises the production
checkpoint with long outputs, fixed-seed determinism, JSON structured output,
and a forced tool call, while recording transparent repetition and leakage
heuristics.  Generated text is written below ``results/raw`` (ignored by Git).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


LONG_PROMPT = (
    "Write an original long-form literary scene about Mara and Ilya, two friends "
    "repairing an abandoned observatory after a promise they both regret. Preserve "
    "close third-person viewpoint, past tense, concrete sensory detail, natural "
    "dialogue, subtext, and a slowly changing relationship. Do not summarize, do "
    "not discuss writing technique, do not add headings, and do not resolve every "
    "conflict. Continue the scene as prose until the token limit."
)


def request_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def words(text: str) -> list[str]:
    return re.findall(r"[\w’'-]+|[^\w\s]", text.lower(), re.UNICODE)


def ngram_excess(tokens: list[str], width: int) -> tuple[float, int]:
    if len(tokens) < width:
        return 0.0, 0
    counts = Counter(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(tokens) - width + 1), max(counts.values(), default=0)


def repetition_metrics(text: str) -> dict:
    tokens = words(text)
    eight_ratio, eight_max = ngram_excess(tokens, 8)
    twelve_ratio, twelve_max = ngram_excess(tokens, 12)
    paragraphs = [" ".join(words(part)) for part in re.split(r"\n\s*\n", text) if part.strip()]
    paragraph_counts = Counter(paragraphs)
    duplicate_paragraphs = sum(count - 1 for count in paragraph_counts.values() if count > 1)
    # Long repeated chunks are a stronger loop signal than common-word counts.
    chunk_width = 32
    chunk_counts = Counter(
        tuple(tokens[index : index + chunk_width])
        for index in range(0, max(0, len(tokens) - chunk_width + 1), chunk_width)
    )
    repeated_chunks = sum(count - 1 for count in chunk_counts.values() if count > 1)
    leakage = sorted(set(re.findall(r"(?i)(?:<\|[^>]+\|>|\b(?:system|assistant|user)\s*:|as an ai language model)", text)))
    return {
        "token_estimate": len(tokens),
        "paragraphs": len(paragraphs),
        "unique_word_ratio": len(set(token for token in tokens if token.isalnum())) / max(1, len([token for token in tokens if token.isalnum()])),
        "exact_8gram_excess_ratio": eight_ratio,
        "exact_8gram_max_count": eight_max,
        "exact_12gram_excess_ratio": twelve_ratio,
        "exact_12gram_max_count": twelve_max,
        "duplicate_paragraphs": duplicate_paragraphs,
        "repeated_32token_chunks": repeated_chunks,
        "prompt_format_leakage": leakage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "lovesenko/DeepSeek-V4-Flash-0731-Abliterated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/phase4-quality"))
    parser.add_argument("--lengths", default="2000,5000,10000", help="comma-separated max_tokens values")
    parser.add_argument("--seed", type=int, default=44017)
    parser.add_argument("--force-length", action="store_true", help="set ignore_eos so 5K/10K endurance runs reach the requested length")
    return parser.parse_args()


def long_generation(base: str, model: str, output_dir: Path, limit: int, seed: int, force_length: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": seed,
        "max_tokens": limit,
        "stream": False,
    }
    if force_length:
        payload["ignore_eos"] = True
    started = time.perf_counter()
    result = request_json(base + "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    choice = result["choices"][0]
    text = choice["message"].get("content", "")
    usage = result.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    path = output_dir / f"long_{limit}.txt"
    path.write_text(text, encoding="utf-8")
    metrics = repetition_metrics(text)
    row = {
        "max_tokens": limit,
        "completion_tokens": completion_tokens,
        "seconds": elapsed,
        "tok_s": completion_tokens / elapsed if elapsed else 0.0,
        "finish_reason": choice.get("finish_reason"),
        "repetition": metrics,
        "quality_gate": {
            "no_format_leakage": not metrics["prompt_format_leakage"],
            "no_obvious_loop": metrics["exact_12gram_excess_ratio"] < 0.08 and metrics["repeated_32token_chunks"] == 0,
        },
        "output_file": str(path),
    }
    print(f"long {limit}: {completion_tokens} tokens, {elapsed:.2f}s, {row['tok_s']:.2f} tok/s, finish={row['finish_reason']}", flush=True)
    return row


def deterministic_test(base: str, model: str, seed: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "In exactly three sentences, describe a locked room and one clue left beside the window."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": 128,
        "stream": False,
    }
    outputs = [request_json(base + "/v1/chat/completions", payload)["choices"][0]["message"].get("content", "") for _ in range(2)]
    return {"exact_match": outputs[0] == outputs[1], "prefix_a": outputs[0][:220], "prefix_b": outputs[1][:220]}


def structured_test(base: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return a JSON object with keys title (string), mood (string), and beats (array of exactly three strings) for a quiet reconciliation scene."}],
        "temperature": 0.2,
        "max_tokens": 180,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    try:
        result = request_json(base + "/v1/chat/completions", payload)
        content = result["choices"][0]["message"].get("content", "")
        parsed = json.loads(content)
        valid = isinstance(parsed, dict) and isinstance(parsed.get("title"), str) and isinstance(parsed.get("mood"), str) and isinstance(parsed.get("beats"), list) and len(parsed["beats"]) == 3 and all(isinstance(item, str) for item in parsed["beats"])
        return {"supported": True, "valid_json": valid, "content": content[:1000]}
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError) as error:
        return {"supported": False, "valid_json": False, "error": str(error)}


def tool_test(base: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Call lookup_weather for Kyoto. Do not answer in prose."}],
        "tools": [{"type": "function", "function": {"name": "lookup_weather", "description": "Look up current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }
    try:
        message = request_json(base + "/v1/chat/completions", payload)["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return {"supported": True, "valid_tool_call": False, "error": "no tool_calls in response", "content": message.get("content", "")[:500]}
        call = calls[0]
        arguments = json.loads(call["function"]["arguments"])
        valid = call["function"]["name"] == "lookup_weather" and isinstance(arguments.get("city"), str)
        return {"supported": True, "valid_tool_call": valid, "name": call["function"]["name"], "arguments": arguments}
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError) as error:
        return {"supported": False, "valid_tool_call": False, "error": str(error)}


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "force_length": args.force_length,
        "long_generations": [long_generation(base, args.model, args.output_dir, limit, args.seed + index, args.force_length) for index, limit in enumerate(lengths)],
        "fixed_seed": deterministic_test(base, args.model, args.seed),
        "structured_output": structured_test(base, args.model),
        "tool_call": tool_test(base, args.model),
    }
    long_rows = report["long_generations"]
    report["summary"] = {
        "long_generation_leakage_free": all(row["quality_gate"]["no_format_leakage"] for row in long_rows),
        "long_generation_loop_free": all(row["quality_gate"]["no_obvious_loop"] for row in long_rows),
        "fixed_seed_exact_match": report["fixed_seed"]["exact_match"],
        "structured_json_valid": report["structured_output"]["valid_json"],
        "tool_call_valid": report["tool_call"]["valid_tool_call"],
        "long_mean_tok_s": statistics.mean(row["tok_s"] for row in long_rows) if long_rows else 0.0,
    }
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
