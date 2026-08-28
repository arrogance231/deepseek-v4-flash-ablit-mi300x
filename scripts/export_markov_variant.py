#!/usr/bin/env python3
"""Create a reversible checkpoint sidecar with calibrated DSpark Markov weights.

Unchanged checkpoint files are hard-linked, so this does not duplicate the
~160-GiB model.  Only shard 48 is rewritten and only the two Markov tensors
are changed.  The original checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import errno
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


W1_NAME = "mtp.2.markov_head.markov_w1.weight"
W2_NAME = "mtp.2.markov_head.markov_w2.weight"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def copy_link_tree(source: Path, output: Path) -> None:
    """Clone a checkpoint without copying weights.

    Bind-mounted source/output paths can be different filesystems inside a
    container, where hard links fail with EXDEV.  Relative symlinks are the
    safe fallback; callers should mount both directories under a common parent
    so those links remain valid on the host.
    """
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = output / relative
        if path.is_dir():
            target.mkdir()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            target.symlink_to(os.path.relpath(path, target.parent))


def main() -> None:
    args = parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if source == output:
        raise SystemExit("--output must be distinct from --checkpoint")
    if not source.is_dir() or not args.calibration.is_file():
        raise SystemExit("checkpoint and calibration file must exist")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output} (use --force to replace)")
        shutil.rmtree(output)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    if mapping.get(W1_NAME) != mapping.get(W2_NAME):
        raise SystemExit("Markov tensors are not co-located in checkpoint")
    shard = mapping[W1_NAME]
    with safe_open(str(args.calibration), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != {W1_NAME, W2_NAME}:
            raise SystemExit(f"calibration must contain exactly {W1_NAME} and {W2_NAME}")
        replacement = {name: reader.get_tensor(name) for name in reader.keys()}
        metadata = dict(reader.metadata() or {})
    with safe_open(str(source / shard), framework="pt", device="cpu") as reader:
        tensors = {name: reader.get_tensor(name) for name in reader.keys()}
        shard_metadata = reader.metadata()
    for name, tensor in replacement.items():
        if name not in tensors or tuple(tensors[name].shape) != tuple(tensor.shape):
            raise SystemExit(f"shape mismatch for {name}")
        tensors[name] = tensor.to(tensors[name].dtype)
    copy_link_tree(source, output)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{Path(shard).name}.", suffix=".tmp", dir=output)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary), metadata=shard_metadata)
        os.replace(temporary, output / shard)
    finally:
        temporary.unlink(missing_ok=True)
    provenance = {
        "phase": "3A-markov-only",
        "source_checkpoint": str(source),
        "calibration_file": str(args.calibration.resolve()),
        "changed_tensors": [W1_NAME, W2_NAME],
        "calibration_metadata": metadata,
        "unchanged_decoder_and_mtp_blocks": True,
    }
    (output / "MARKOV_CALIBRATION_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    print(f"created {output}")


if __name__ == "__main__":
    main()
