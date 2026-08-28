#!/usr/bin/env python3
"""Verify that an MTP-only variant changed no decoder tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--abliterated-dir", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--variant-dir", type=Path, required=True)
    args = p.parse_args()

    with (args.abliterated_dir / "model.safetensors.index.json").open() as f:
        target_map = json.load(f)["weight_map"]
    with (args.base_dir / "model.safetensors.index.json").open() as f:
        base_map = json.load(f)["weight_map"]

    replacement = {
        key
        for key in target_map
        if key.startswith("mtp.")
        and key.endswith((".attn.wo_b.weight", ".attn.wo_b.scale"))
    }
    checked = replaced = 0
    for shard in sorted({target_map[key] for key in replacement}):
        keys = {key for key in replacement if target_map[key] == shard}
        with (
            safe_open(str(args.abliterated_dir / shard), framework="pt", device="cpu") as original,
            safe_open(str(args.variant_dir / shard), framework="pt", device="cpu") as variant,
        ):
            if set(original.keys()) != set(variant.keys()):
                raise RuntimeError(f"key set changed in {shard}")
            for key in original.keys():
                left = original.get_tensor(key)
                right = variant.get_tensor(key)
                if key in keys:
                    with safe_open(
                        str(args.base_dir / base_map[key]), framework="pt", device="cpu"
                    ) as base_reader:
                        expected = base_reader.get_tensor(key)
                    if not torch.equal(right, expected):
                        raise RuntimeError(f"replacement does not match base: {key}")
                    replaced += 1
                elif not torch.equal(left, right):
                    raise RuntimeError(f"unexpected change: {key}")
                checked += 1
    print(f"PASS: checked {checked} tensors; restored {replaced} MTP tensors")


if __name__ == "__main__":
    main()
