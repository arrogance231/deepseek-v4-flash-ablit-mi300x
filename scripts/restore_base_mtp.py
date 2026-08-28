#!/usr/bin/env python3
"""Build a reversible A/B checkpoint with the original DSpark MTP tensors.

The output hard-links unchanged files from the abliterated checkpoint, then
rewrites only ``mtp.*.attn.wo_b.{weight,scale}`` from a compatible base model.
This keeps the 43 abliterated decoder edits intact and avoids a second full
model copy. Run this inside the pinned vLLM container, which provides
``safetensors``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_BASE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_BASE_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abliterated-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_index(directory: Path) -> dict:
    with (directory / "model.safetensors.index.json").open() as handle:
        return json.load(handle)


def copy_as_hardlink_tree(source: Path, destination: Path) -> None:
    """Copy the model directory without duplicating unchanged weight bytes."""
    shutil.copytree(source, destination, copy_function=os.link)


def replace_shard(
    output_shard: Path,
    base_shard: Path,
    replacement_keys: set[str],
) -> None:
    with safe_open(str(output_shard), framework="pt", device="cpu") as output_reader:
        output_keys = set(output_reader.keys())
        output_metadata = output_reader.metadata()
        missing_output = replacement_keys - output_keys
        if missing_output:
            raise RuntimeError(
                f"{output_shard.name} is missing expected output tensors: "
                f"{sorted(missing_output)}"
            )
        tensors = {key: output_reader.get_tensor(key) for key in output_keys}

    with safe_open(str(base_shard), framework="pt", device="cpu") as base_reader:
        base_keys = set(base_reader.keys())
        missing_base = replacement_keys - base_keys
        if missing_base:
            raise RuntimeError(
                f"{base_shard.name} is missing expected base tensors: "
                f"{sorted(missing_base)}"
            )
        for key in replacement_keys:
            tensors[key] = base_reader.get_tensor(key)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_shard.name}.", suffix=".tmp", dir=output_shard.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        save_file(tensors, str(temporary_path), metadata=output_metadata)
        os.replace(temporary_path, output_shard)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    ablit = args.abliterated_dir.resolve()
    base = args.base_dir.resolve()
    output = args.output_dir.resolve()

    if not ablit.is_dir() or not base.is_dir():
        raise SystemExit("--abliterated-dir and --base-dir must be directories")
    if output == ablit or output == base:
        raise SystemExit("--output-dir must be distinct from both input directories")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {output} (use --force to replace)")
        shutil.rmtree(output)

    target_index = load_index(ablit)
    base_index = load_index(base)
    target_map = target_index["weight_map"]
    base_map = base_index["weight_map"]

    replacement_keys = {
        key
        for key in target_map
        if key.startswith("mtp.")
        and key.endswith((".attn.wo_b.weight", ".attn.wo_b.scale"))
    }
    if not replacement_keys:
        raise SystemExit("no MTP wo_b tensors found in the abliterated index")
    for key in replacement_keys:
        if key not in base_map:
            raise SystemExit(f"base index is missing {key}")

    copy_as_hardlink_tree(ablit, output)
    shards = sorted({target_map[key] for key in replacement_keys})
    for shard in shards:
        shard_keys = {key for key in replacement_keys if target_map[key] == shard}
        base_shards = {base_map[key] for key in shard_keys}
        if len(base_shards) != 1:
            raise SystemExit(f"MTP tensors for {shard} span unexpected base shards: {base_shards}")
        replace_shard(output / shard, base / next(iter(base_shards)), shard_keys)

    provenance = {
        "target_checkpoint": str(ablit),
        "base_repository": args.base_repo,
        "base_revision": args.base_revision,
        "replaced_tensors": sorted(replacement_keys),
        "unchanged_decoder": True,
    }
    (output / "MTP_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))
    print(f"created {output}")


if __name__ == "__main__":
    main()
