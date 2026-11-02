#!/usr/bin/env python3
"""Create a deterministic archive for a DeepSeek execution-code digest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

from hash_deepseek_execution_code import digest, source_files


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_archive(root: Path, files: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                    for path in files:
                        relative = path.relative_to(root).as_posix()
                        content = path.read_bytes()
                        info = tarfile.TarInfo(relative)
                        info.size = len(content)
                        info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(content))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(destination: Path, payload: dict[str, object]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    execution_sha256, file_count = digest(root)
    if execution_sha256 != args.expected_sha256:
        raise SystemExit(
            "execution source digest mismatch: "
            f"expected {args.expected_sha256}, got {execution_sha256}"
        )

    files = source_files(root)
    if len(files) != file_count:
        raise SystemExit("execution source file count changed during archive creation")

    stem = f"deepseek_execution_source_{execution_sha256[:12]}"
    archive_path = args.output_dir.resolve() / f"{stem}.tar.gz"
    manifest_path = args.output_dir.resolve() / f"{stem}.manifest.json"
    _write_archive(root, files, archive_path)

    manifest_files = []
    for path in files:
        manifest_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "execution_code_sha256": execution_sha256,
        "file_count": file_count,
        "files": manifest_files,
        "root_at_archive_time": str(root),
        "schema_version": 1,
    }
    _write_json(manifest_path, payload)
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "archive_sha256": payload["archive_sha256"],
                "execution_code_sha256": execution_sha256,
                "file_count": file_count,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
