#!/usr/bin/env python3
"""Calibrate the small DSpark Markov head from target-generated token streams.

This is deliberately a narrow Phase-3A pilot.  It does not touch the 43-layer
target model or the three MTP decoder blocks.  ``DSparkMarkovHead`` is a
low-rank token-transition bias: ``Embedding(V, r)`` followed by a
``Linear(r, V)``.  The script learns a conservative update from continuations
sampled from the frozen abliterated target, with parameter-delta regularization
so an accidentally overfit draft head can be rejected before serving.

Run inside the pinned vLLM image (it supplies torch and safetensors):

    python scripts/train_markov_calibrator.py \
      --checkpoint /models/deepseek \
      --data results/raw/phase3-mtp-calibration.jsonl \
      --output results/raw/phase3-markov-calibrated.safetensors

The output contains exactly two tensors and is consumed by
``export_markov_variant.py``.  A full hidden-state MTP distillation remains a
separate, substantially more expensive research task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


W1_NAME = "mtp.2.markov_head.markov_w1.weight"
W2_NAME = "mtp.2.markov_head.markov_w2.weight"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--delta-reg", type=float, default=2e-2)
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7301)
    return p.parse_args()


def find_shard(checkpoint: Path, tensor_name: str) -> Path:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    try:
        return checkpoint / index["weight_map"][tensor_name]
    except KeyError as exc:
        raise SystemExit(f"checkpoint index does not contain {tensor_name}") from exc


def load_markov(checkpoint: Path) -> tuple[torch.Tensor, torch.Tensor]:
    shard = find_shard(checkpoint, W1_NAME)
    shard2 = find_shard(checkpoint, W2_NAME)
    if shard != shard2:
        raise SystemExit(f"expected Markov tensors in one shard, got {shard} and {shard2}")
    with safe_open(str(shard), framework="pt", device="cpu") as reader:
        w1 = reader.get_tensor(W1_NAME).to(torch.float32)
        w2 = reader.get_tensor(W2_NAME).to(torch.float32)
    if w1.ndim != 2 or w2.ndim != 2 or w1.shape[1] != w2.shape[1]:
        raise SystemExit(f"unexpected Markov shapes: w1={tuple(w1.shape)} w2={tuple(w2.shape)}")
    if w1.shape[0] != w2.shape[0]:
        raise SystemExit("Markov source and destination vocabularies differ; pilot is full-vocab only")
    return w1, w2


def load_pairs(path: Path, seed: int, heldout_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tokens = [int(token) for token in row.get("tokens", [])]
            pairs.extend(zip(tokens[:-1], tokens[1:]))
    if len(pairs) < 128:
        raise SystemExit(f"need at least 128 token transitions, found {len(pairs)}")
    random.Random(seed).shuffle(pairs)
    split = max(1, min(len(pairs) - 1, int(len(pairs) * (1.0 - heldout_fraction))))
    train, heldout = pairs[:split], pairs[split:]
    tx, ty = zip(*train)
    vx, vy = zip(*heldout)
    return (
        torch.tensor(tx, dtype=torch.long),
        torch.tensor(ty, dtype=torch.long),
        torch.tensor(vx, dtype=torch.long),
        torch.tensor(vy, dtype=torch.long),
    )


def evaluate(w1: torch.Tensor, w2: torch.Tensor, source: torch.Tensor, target: torch.Tensor, batch_size: int) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, source.numel(), batch_size):
            ids = source[start : start + batch_size]
            labels = target[start : start + batch_size]
            logits = F.linear(w1[ids], w2)
            losses.append(float(F.cross_entropy(logits, labels).cpu()))
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    w1_base, w2_base = load_markov(args.checkpoint)
    train_x, train_y, held_x, held_y = load_pairs(args.data, args.seed, args.heldout_fraction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w1 = w1_base.to(device).requires_grad_(True)
    w2 = w2_base.to(device).requires_grad_(True)
    base1, base2 = w1.detach().clone(), w2.detach().clone()
    train_x, train_y = train_x.to(device), train_y.to(device)
    held_x, held_y = held_x.to(device), held_y.to(device)
    before = evaluate(w1, w2, held_x, held_y, args.batch_size)
    optimizer = torch.optim.AdamW([w1, w2], lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    best_loss = math.inf
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    for step in range(1, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % train_x.numel()
        indices = torch.arange(offset, offset + args.batch_size, device=device) % train_x.numel()
        ids, labels = train_x[indices], train_y[indices]
        logits = F.linear(w1[ids], w2)
        ce = F.cross_entropy(logits, labels)
        # Mean-square delta keeps the pilot close to the checkpoint.  The
        # coefficient is intentionally nonzero even for a tiny corpus.
        reg = args.delta_reg * (
            (w1 - base1).square().mean() + (w2 - base2).square().mean()
        )
        loss = ce + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w1, w2], 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            held_loss = evaluate(w1, w2, held_x, held_y, args.batch_size)
            print(f"step={step:04d} train_ce={ce.item():.4f} reg={reg.item():.6f} heldout_ce={held_loss:.4f}", flush=True)
            if held_loss < best_loss:
                best_loss = held_loss
                best = (w1.detach().clone(), w2.detach().clone())
    if best is None:
        best = (w1.detach(), w2.detach())
    after = evaluate(best[0], best[1], held_x, held_y, args.batch_size)
    delta1 = float((best[0] - base1).norm().cpu())
    delta2 = float((best[1] - base2).norm().cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({W1_NAME: best[0].to(torch.bfloat16).cpu(), W2_NAME: best[1].to(torch.bfloat16).cpu()}, str(args.output), metadata={
        "source_checkpoint": str(args.checkpoint),
        "calibration_data": str(args.data),
        "phase": "3A-markov-only",
        "heldout_ce_before": f"{before:.8f}",
        "heldout_ce_after": f"{after:.8f}",
        "delta_w1_l2": f"{delta1:.8f}",
        "delta_w2_l2": f"{delta2:.8f}",
    })
    print(json.dumps({
        "device": str(device), "transitions": int(train_x.numel() + held_x.numel()),
        "train_transitions": int(train_x.numel()), "heldout_transitions": int(held_x.numel()),
        "heldout_ce_before": before, "heldout_ce_after": after,
        "delta_w1_l2": delta1, "delta_w2_l2": delta2, "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
