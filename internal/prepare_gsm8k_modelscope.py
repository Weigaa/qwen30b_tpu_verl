#!/usr/bin/env python3
"""Download GSM8K from ModelScope and convert it to verl parquet files."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_PROMPT_SUFFIX = 'Let\'s think step by step and output the final answer after "####".'
FINAL_ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")
NUMBER_RE = re.compile(r"-?[0-9.,]+")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", default="/data/gsm8k", help="Directory for raw jsonl and verl parquet files")
    parser.add_argument("--dataset-name", default="gsm8k", help="ModelScope dataset name")
    parser.add_argument("--namespace", default="modelscope", help="ModelScope namespace")
    parser.add_argument("--subset-name", default="main", help="ModelScope subset/config name")
    parser.add_argument("--train-split", default="train", help="Source training split name")
    parser.add_argument("--test-split", default="test", help="Source test split name")
    parser.add_argument("--data-source", default="openai/gsm8k", help="verl data_source field")
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX, help="Instruction appended to each question")
    parser.add_argument("--skip-repo-download", action="store_true", help="Do not mirror the dataset repo metadata")
    parser.add_argument("--repo-dir-name", default="modelscope_raw", help="Subdirectory for the ModelScope repo mirror")
    parser.add_argument("--write-short", action="store_true", help="Also write train_short/test_short parquet files")
    parser.add_argument("--train-short-size", type=int, default=1024, help="Rows for train_short.parquet")
    parser.add_argument("--test-short-size", type=int, default=256, help="Rows for test_short.parquet")
    return parser.parse_args()


def dataset_id(dataset_name: str, namespace: str) -> str:
    if "/" in dataset_name:
        return dataset_name
    return f"{namespace}/{dataset_name}"


def maybe_download_repo(args, output_dir: Path):
    if args.skip_repo_download:
        return
    repo_dir = output_dir / args.repo_dir_name
    if shutil.which("modelscope") is None:
        raise RuntimeError("modelscope CLI is not available; install modelscope or use --skip-repo-download")
    cmd = [
        "modelscope",
        "download",
        "--dataset",
        dataset_id(args.dataset_name, args.namespace),
        "--local_dir",
        str(repo_dir),
        "--max-workers",
        "4",
    ]
    subprocess.run(cmd, check=True)


def load_modelscope_split(args, split_name: str):
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as exc:
        raise RuntimeError(
            "Importing modelscope.msdatasets failed. Install dataset extras with "
            "`pip install 'modelscope[datasets]'` or at least `pip install addict simplejson sortedcontainers oss2`."
        ) from exc

    kwargs = {
        "subset_name": args.subset_name,
        "split": split_name,
        "trust_remote_code": True,
    }
    if "/" in args.dataset_name:
        return MsDataset.load(args.dataset_name, **kwargs)
    return MsDataset.load(args.dataset_name, namespace=args.namespace, **kwargs)


def to_plain_rows(dataset):
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return [dict(item) for item in dataset]


def extract_ground_truth(answer: str) -> str:
    answer = str(answer).strip()
    match = FINAL_ANSWER_RE.search(answer)
    if match:
        value = match.group(1)
    else:
        numbers = NUMBER_RE.findall(answer)
        if not numbers:
            raise ValueError(f"Could not extract GSM8K final answer from: {answer[:120]!r}")
        value = numbers[-1]
    return value.replace(",", "").replace("$", "").strip()


def build_prompt(question: str, suffix: str) -> str:
    question = str(question).strip()
    suffix = str(suffix).strip()
    if not suffix:
        return question
    return f"{question}\n{suffix}"


def convert_rows(rows, split_name: str, args):
    converted = []
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if question is None or answer is None:
            raise KeyError(f"Expected question/answer fields, got keys={list(row.keys())}")
        ground_truth = extract_ground_truth(answer)
        converted.append(
            {
                "data_source": args.data_source,
                "prompt": [{"role": "user", "content": build_prompt(question, args.prompt_suffix)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split_name,
                    "index": index,
                    "question": str(question).strip(),
                    "answer": str(answer).strip(),
                },
                "uid": f"gsm8k-{split_name}-{index:05d}",
            }
        )
    return converted


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_repo(args, output_dir)

    raw_train = to_plain_rows(load_modelscope_split(args, args.train_split))
    raw_test = to_plain_rows(load_modelscope_split(args, args.test_split))
    write_jsonl(output_dir / "raw" / "train.jsonl", raw_train)
    write_jsonl(output_dir / "raw" / "test.jsonl", raw_test)

    train_rows = convert_rows(raw_train, "train", args)
    test_rows = convert_rows(raw_test, "test", args)
    write_parquet(output_dir / "train.parquet", train_rows)
    write_parquet(output_dir / "test.parquet", test_rows)
    write_parquet(output_dir / "valid.parquet", test_rows)

    if args.write_short:
        write_parquet(output_dir / "train_short.parquet", train_rows[: args.train_short_size])
        write_parquet(output_dir / "test_short.parquet", test_rows[: args.test_short_size])

    summary = {
        "dataset": dataset_id(args.dataset_name, args.namespace),
        "subset": args.subset_name,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "output_dir": str(output_dir),
        "data_source": args.data_source,
        "prompt_suffix": args.prompt_suffix,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
