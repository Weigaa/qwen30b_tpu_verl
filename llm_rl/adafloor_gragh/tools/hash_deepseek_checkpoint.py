#!/usr/bin/env python3
"""Hash the immutable actor state of a frozen DeepSeek checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_files(checkpoint: Path) -> list[Path]:
    checkpoint = checkpoint.resolve()
    actor = checkpoint / "actor"
    preserve = checkpoint / ".PRESERVE_COMMON_EPOCH0"
    if not actor.is_dir():
        raise ValueError(f"missing actor checkpoint directory: {actor}")
    if not preserve.is_file():
        raise ValueError(f"missing checkpoint preservation marker: {preserve}")
    paths = [preserve]
    for path in actor.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains a symbolic link: {path}")
        if path.is_file():
            paths.append(path)
    if not any(path.suffix == ".distcp" for path in paths):
        raise ValueError(f"checkpoint has no distcp shards: {checkpoint}")
    return sorted(paths, key=lambda path: path.relative_to(checkpoint).as_posix())


def digest(checkpoint: Path) -> tuple[str, int, int]:
    checkpoint = checkpoint.resolve()
    paths = checkpoint_files(checkpoint)
    hasher = hashlib.sha256(b"AdaFloor DeepSeek frozen checkpoint v1\0")
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(checkpoint).as_posix().encode("utf-8")
        size = path.stat().st_size
        total_bytes += size
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest(), len(paths), total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--show-details", action="store_true")
    args = parser.parse_args()
    try:
        value, count, total_bytes = digest(args.checkpoint)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"checkpoint hash failed: {exc}\n")
    if args.show_details:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "sha256": value,
                    "file_count": count,
                    "total_bytes": total_bytes,
                },
                sort_keys=True,
            )
        )
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
