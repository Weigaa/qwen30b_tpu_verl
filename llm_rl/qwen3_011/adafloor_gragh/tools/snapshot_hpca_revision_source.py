#!/usr/bin/env python3
"""Create a code-only copy of the dirty paper-eager tree without changing it."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--extend-directory", choices=["patches", "r1_ascend"],
                        help="Copy a missing code dependency without overwriting revision edits")
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.destination.absolute()
    if args.extend_directory:
        marker = dest / "PAPER_SOURCE_SNAPSHOT.json"
        if not marker.is_file() or json.loads(marker.read_text())["source"] != str(source):
            raise ValueError("not the recorded revision source snapshot")
        name = args.extend_directory
        if (dest / name).exists():
            raise ValueError("refusing to overwrite dependency")
        shutil.copytree(source / name, dest / name, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        records = []
        for path in sorted((dest / name).rglob("*")):
            if path.is_file() and not path.is_symlink():
                original = source / path.relative_to(dest)
                digest = sha(path)
                if digest != sha(original):
                    raise ValueError("dependency copy mismatch")
                records.append({"source": str(original), "relative_path": str(path.relative_to(dest)), "sha256": digest})
        (dest / f"PAPER_SOURCE_SNAPSHOT_EXTRA_{name}.json").write_text(json.dumps(
            {"copied_regular_files": records, "tool_sha256": sha(Path(__file__))}, indent=2, sort_keys=True) + "\n")
        print(f"extended={name} verified_regular_files={len(records)}")
        return
    if dest.exists() or dest == source or source in dest.parents:
        raise ValueError("destination must be a new sibling tree, not the paper source")
    dest.mkdir()
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")
    for name in ("vllm", "vllm_ascend", "verl", "megatron", "mindspeed", "tools", "internal", "tests", "patches", "r1_ascend"):
        shutil.copytree(source / name, dest / name, symlinks=True, ignore=ignore)
    for path in source.iterdir():
        if path.is_file() and (path.suffix in (".sh", ".py") or path.name in
                              ("pyproject.toml", "setup.cfg", "setup.py")):
            shutil.copy2(path, dest / path.name)
    (dest / "analysis_eval").mkdir()
    for path in (source / "analysis_eval").glob("*.py"):
        shutil.copy2(path, dest / "analysis_eval" / path.name)
    records = []
    for path in sorted(dest.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(dest)
            original = source / relative
            digest = sha(path)
            if digest != sha(original):
                raise ValueError(f"copy differs: {relative}")
            records.append({"relative_path": str(relative), "source": str(original), "sha256": digest})
    (dest / "PAPER_SOURCE_SNAPSHOT.json").write_text(json.dumps(
        {"source": str(source), "destination": str(dest), "code_only": True,
         "copied_regular_files": records, "snapshot_tool_sha256": sha(Path(__file__)),
         "scope": "Working-tree source copy including uncommitted paper changes; not a native-library/environment snapshot."},
        indent=2, sort_keys=True) + "\n")
    print(f"snapshot={dest} verified_regular_files={len(records)}")


if __name__ == "__main__":
    main()
