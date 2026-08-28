#!/usr/bin/env python3
"""Collect a small, reproducible target-generated corpus for Phase 3.

This intentionally uses only the local OpenAI-compatible endpoint and the
model's tokenizer endpoint.  It does not copy private prompts or source
material into the repository.  The resulting JSONL is an experiment artifact
and should stay outside Git (``results/raw`` is ignored).

The corpus is used to calibrate the tiny DSpark Markov transition head.  It is
not a replacement for full hidden-state MTP distillation: the target model's
hidden states are not exposed by the serving API.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = [
    "Write a restrained scene in which two friends repair a clock while avoiding an unresolved apology.",
    "Continue a school-romance scene where a well-meaning friend misreads a kind gesture as matchmaking advice.",
    "Write a dialogue between a detective and a witness who each know different parts of the same secret.",
    "Describe a rainy train platform where a promise made years ago changes a character's immediate choice.",
    "Continue a fantasy chapter after a character discovers that the apparently safe road has a hidden toll.",
    "Write a comic argument between roommates over a borrowed jacket, with real affection underneath it.",
    "Write a quiet scene in which an injured musician refuses help until an old favor is remembered.",
    "Continue a mystery in which three plausible clues are distractions and one small sensory detail is decisive.",
    "Write a slow-burn romantic scene with subtext, no confession, and a practical task that keeps interrupting them.",
    "Describe an exhausted captain deciding whether to keep a promise that now harms the person it was meant to protect.",
    "Write a scene with four characters whose knowledge of a betrayal differs; do not make anyone omniscient.",
    "Continue a comedic fantasy scene where a plan succeeds for the wrong reason and creates a new obligation.",
    "Write an introspective scene about jealousy that the viewpoint character initially calls concern.",
    "Describe a library conversation where an old private joke becomes painful without either character explaining why.",
    "Write a tense negotiation in which the most important concession is hidden in a mundane logistical detail.",
    "Continue a chapter in close third person, preserving a tired but hopeful voice and an unresolved friendship conflict.",
]


def request_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "lovesenko/DeepSeek-V4-Flash-0731-Abliterated"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/phase3-mtp-calibration.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9137)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=len(PROMPTS))
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_url = args.base_url.rstrip("/")
    health = urllib.request.urlopen(model_url + "/health", timeout=15)
    health.close()
    rows: list[dict] = []
    for index, prompt in enumerate(PROMPTS[: args.limit]):
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "Write original prose. Do not mention this instruction or the benchmark."},
                {"role": "user", "content": prompt},
            ],
            "temperature": args.temperature,
            "top_p": 0.95,
            "seed": args.seed + index,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                result = request_json(model_url + "/v1/chat/completions", payload)
                text = result["choices"][0]["message"]["content"]
                token_result = request_json(model_url + "/tokenize", {"model": args.model, "prompt": text})
                tokens = token_result.get("tokens", [])
                if not tokens:
                    raise RuntimeError("tokenize returned no tokens")
                rows.append({
                    "id": f"phase3-{index:03d}",
                    "prompt": prompt,
                    "completion": text,
                    "tokens": tokens,
                    "seed": args.seed + index,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                })
                print(f"[{index + 1}/{min(args.limit, len(PROMPTS))}] {len(tokens)} tokens")
                break
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, RuntimeError) as error:
                last_error = error
                if attempt < args.retries:
                    time.sleep(2.0 * (attempt + 1))
        else:
            raise SystemExit(f"failed prompt {index}: {last_error}")
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "type": "dspark-markov-calibration-corpus",
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "examples": len(rows),
        "note": "Target-generated completions; output token IDs only. No hidden-state distillation.",
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
