#!/usr/bin/env python3
"""Build a deterministic, paired Eurus-2-RL code rollout workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SUPPORTED_CODE_SOURCES = {"apps", "codecontests", "codeforces", "taco"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_prompt(prompt: object) -> str:
    return json.dumps(prompt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def validate_test_cases(value: object) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if not isinstance(value, dict):
        return False
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    return (
        isinstance(inputs, list)
        and isinstance(outputs, list)
        and len(inputs) > 0
        and len(inputs) == len(outputs)
    )


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--selection-seed", type=int, default=20260730)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.train_samples <= 0 or args.test_samples <= 0:
        raise SystemExit("train and test sample counts must be positive")
    if not args.input.is_file():
        raise SystemExit(f"missing source parquet: {args.input}")
    if not args.model_path.is_dir():
        raise SystemExit(f"missing local tokenizer: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (train_path, test_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite an existing workload: "
            + ", ".join(str(path) for path in existing)
        )

    source_table = pq.read_table(args.input)
    required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing_columns = required_columns - set(source_table.column_names)
    if missing_columns:
        raise SystemExit(f"source parquet is missing columns: {sorted(missing_columns)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True
    )
    eligible: list[tuple[str, int, int, str]] = []
    rejection_counts: Counter[str] = Counter()
    seen_prompts: set[str] = set()

    for row_index, row in enumerate(source_table.to_pylist()):
        if str(row.get("ability", "")).lower() != "code":
            rejection_counts["not_code"] += 1
            continue
        data_source = str(row.get("data_source", "")).lower()
        if data_source not in SUPPORTED_CODE_SOURCES:
            rejection_counts["unsupported_data_source"] += 1
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            rejection_counts["invalid_prompt"] += 1
            continue
        prompt_key = canonical_prompt(prompt)
        if prompt_key in seen_prompts:
            rejection_counts["duplicate_prompt"] += 1
            continue
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict) or not validate_test_cases(
            reward_model.get("ground_truth")
        ):
            rejection_counts["invalid_test_cases"] += 1
            continue
        try:
            prompt_tokens = len(
                tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            )
        except Exception:
            rejection_counts["chat_template_failure"] += 1
            continue
        if prompt_tokens > args.max_prompt_length:
            rejection_counts["overlong_prompt"] += 1
            continue
        seen_prompts.add(prompt_key)
        selection_key = hashlib.sha256(
            f"{args.selection_seed}\0{data_source}\0{prompt_key}".encode("utf-8")
        ).hexdigest()
        eligible.append((selection_key, row_index, prompt_tokens, data_source))

    eligible.sort()
    required_rows = args.train_samples + args.test_samples
    if len(eligible) < required_rows:
        raise SystemExit(
            f"only {len(eligible)} eligible code rows remain, need {required_rows}"
        )

    train_records = eligible[: args.train_samples]
    test_records = eligible[args.train_samples : required_rows]
    train_indices = [record[1] for record in train_records]
    test_indices = [record[1] for record in test_records]
    train_table = source_table.take(pa.array(train_indices, type=pa.int64()))
    test_table = source_table.take(pa.array(test_indices, type=pa.int64()))
    pq.write_table(train_table, train_path, compression="zstd")
    pq.write_table(test_table, test_path, compression="zstd")

    prompt_lengths = [record[2] for record in train_records]
    manifest = {
        "schema_version": 1,
        "dataset": "PRIME-RL/Eurus-2-RL-Data",
        "source_split": args.source_split,
        "source_parquet": str(args.input.resolve()),
        "source_parquet_sha256": sha256_file(args.input),
        "selection_seed": args.selection_seed,
        "selection_method": "sha256_order_after_schema_and_prompt_length_filter",
        "model_tokenizer": str(args.model_path.resolve()),
        "max_prompt_length": args.max_prompt_length,
        "train_rows": len(train_records),
        "test_rows": len(test_records),
        "train_source_rows": train_indices,
        "test_source_rows": test_indices,
        "train_source_counts": dict(sorted(Counter(record[3] for record in train_records).items())),
        "test_source_counts": dict(sorted(Counter(record[3] for record in test_records).items())),
        "train_prompt_tokens": {
            "min": min(prompt_lengths),
            "median": median(prompt_lengths),
            "p95": quantile(prompt_lengths, 0.95),
            "max": max(prompt_lengths),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "train_parquet": str(train_path.resolve()),
        "train_parquet_sha256": sha256_file(train_path),
        "test_parquet": str(test_path.resolve()),
        "test_parquet_sha256": sha256_file(test_path),
        "supported_reward_sources": sorted(SUPPORTED_CODE_SOURCES),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)

    print(f"prepared={args.output_dir}")
    print(f"train_rows={len(train_records)} test_rows={len(test_records)}")
    print(f"train_source_counts={manifest['train_source_counts']}")
    print(f"train_prompt_tokens={manifest['train_prompt_tokens']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
