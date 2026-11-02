#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re


PAD_TOKEN_ID = 151643


def collapse(text: str, limit: int = 180) -> str:
    text = text.replace("\n", "\\n")
    text = re.sub(r"( &#){8,}", " &#...[repeat]...", text)
    text = re.sub(r"(!){16,}", "!!!!...[repeat]...", text)
    return text[:limit]


def summarize_file(path: pathlib.Path) -> None:
    outputs = []
    scores = []
    lengths = []
    heads = []
    first = None
    with path.open(errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            output = obj.get("output", "")
            response = obj.get("responses") or []
            outputs.append(output)
            scores.append(obj.get("score"))
            lengths.append(sum(1 for token in response if token != PAD_TOKEN_ID))
            heads.append(tuple(response[:16]))
            if first is None:
                first = (obj.get("score"), output, response[:24])

    if not outputs:
        print(f"{path.name}: empty")
        return

    output_counts = collections.Counter(outputs)
    head_counts = collections.Counter(heads)
    score_counts = collections.Counter(scores)
    bad_punct = sum(1 for output in outputs if output and set(output) <= {"!"})
    bad_entity = sum(1 for output in outputs if output.startswith(" &#"))
    print(f"{path.name}: n={len(outputs)} uniq_out={len(output_counts)} "
          f"score_counts={score_counts.most_common(6)} "
          f"len_min/mean/max=({min(lengths)},{sum(lengths)/len(lengths):.2f},{max(lengths)}) "
          f"bang_only={bad_punct} entity_prefix={bad_entity}")
    top_output, top_count = output_counts.most_common(1)[0]
    print(f"  top_count={top_count} top_output={collapse(top_output)}")
    print(f"  top_heads={head_counts.most_common(3)}")
    if first is not None:
        score, output, response_head = first
        print(f"  first score={score} output={collapse(output)} response_head={response_head}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_data", type=pathlib.Path)
    parser.add_argument("--max-files", type=int, default=10)
    args = parser.parse_args()
    files = sorted(args.rollout_data.glob("*.jsonl"))[:args.max_files]
    if not files:
        raise SystemExit(f"no jsonl files found under {args.rollout_data}")
    for path in files:
        summarize_file(path)


if __name__ == "__main__":
    main()
