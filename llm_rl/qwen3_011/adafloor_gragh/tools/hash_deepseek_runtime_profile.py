#!/usr/bin/env python3
"""Hash an ordered DeepSeek runtime-profile source closure."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(root: Path, profiles: list[Path]) -> str:
    root = root.resolve()
    if not profiles:
        raise ValueError("at least one runtime profile is required")
    hasher = hashlib.sha256()
    seen: set[Path] = set()
    for raw_path in profiles:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        path = path.resolve()
        if path in seen:
            raise ValueError(f"duplicate runtime profile: {path}")
        seen.add(path)
        if not path.is_file():
            raise ValueError(f"missing runtime profile: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"runtime profile is outside repository root: {path}") from exc
        name = relative.encode("utf-8")
        content = path.read_bytes()
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", required=True, action="append", type=Path)
    args = parser.parse_args()
    print(digest(args.root, args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
