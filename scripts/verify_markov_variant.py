#!/usr/bin/env python3
"""Verify that a Phase-3A sidecar changes exactly the Markov head tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


EXPECTED = {
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--variant", type=Path, required=True)
    args = p.parse_args()
    index = json.loads((args.original / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    changed: list[str] = []
    checked = 0
    for shard in shards:
        left_path, right_path = args.original / shard, args.variant / shard
        with safe_open(str(left_path), framework="pt", device="cpu") as left, safe_open(str(right_path), framework="pt", device="cpu") as right:
            if set(left.keys()) != set(right.keys()):
                raise SystemExit(f"key set changed in {shard}")
            for name in left.keys():
                if not torch.equal(left.get_tensor(name), right.get_tensor(name)):
                    changed.append(name)
                checked += 1
    if set(changed) != EXPECTED or len(changed) != len(EXPECTED):
        raise SystemExit(f"unexpected changed tensors: {changed}")
    print(f"PASS: checked {checked} tensors; changed only {sorted(EXPECTED)}")


if __name__ == "__main__":
    main()
