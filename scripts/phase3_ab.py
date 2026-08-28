#!/usr/bin/env python3
"""Run the fixed Phase-3 live A/B probe against an OpenAI endpoint.

The probe intentionally reports client wall-clock decode throughput and the
DSpark counters exposed by vLLM.  It is small enough to run after every
sidecar restart, while keeping the prompt, seed, sampler, and output length
identical between candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path


PROMPT = "Write a coherent short scene with two friends disagreeing about a promise, using natural dialogue and a clear ending."


def get_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def metric(url: str, name: str) -> float:
    text = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    match = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "lovesenko/DeepSeek-V4-Flash-0731-Abliterated"))
    p.add_argument("--label", required=True)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    before_draft = metric(base, "vllm:spec_decode_num_draft_tokens_total")
    before_accepted = metric(base, "vllm:spec_decode_num_accepted_tokens_total")
    rows = []
    for index in range(args.runs):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.8, "top_p": 0.95, "seed": args.seed,
            "max_tokens": args.max_tokens, "stream": False,
        }
        started = time.perf_counter()
        result = get_json(base + "/v1/chat/completions", payload)
        elapsed = time.perf_counter() - started
        tokens = int(result.get("usage", {}).get("completion_tokens", 0))
        content = result["choices"][0]["message"].get("content", "")
        rows.append({"run": index, "tokens": tokens, "seconds": elapsed, "tok_s": tokens / elapsed, "prefix": content[:220]})
        print(f"{args.label} run={index} tokens={tokens} seconds={elapsed:.3f} tok_s={tokens / elapsed:.2f}", flush=True)
    after_draft = metric(base, "vllm:spec_decode_num_draft_tokens_total")
    after_accepted = metric(base, "vllm:spec_decode_num_accepted_tokens_total")
    summary = {
        "label": args.label,
        "model": args.model,
        "runs": rows,
        "warmup_excluded_mean_tok_s": statistics.mean(row["tok_s"] for row in rows[1:]) if len(rows) > 1 else rows[0]["tok_s"],
        "warmup_excluded_median_tok_s": statistics.median(row["tok_s"] for row in rows[1:]) if len(rows) > 1 else rows[0]["tok_s"],
        "draft_tokens_delta": after_draft - before_draft,
        "accepted_tokens_delta": after_accepted - before_accepted,
        "accepted_per_draft": (after_accepted - before_accepted) / (after_draft - before_draft) if after_draft > before_draft else None,
        "dspark_metrics": {"draft_before": before_draft, "draft_after": after_draft, "accepted_before": before_accepted, "accepted_after": after_accepted},
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
