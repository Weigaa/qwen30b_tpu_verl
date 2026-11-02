#!/usr/bin/env python3
"""Build explicitly censored history from instrumented rollout artifacts.

Old mask-only artifacts are deliberately rejected: finish reason and actual
generated token count cannot be reliably reconstructed from padded EOS masks.
This tool alone does not wire the corrected estimates into the live planner.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verl.utils.revision_censored_history import update_tail


def build(rows, previous=None, responses_per_prompt=16):
    previous = {} if previous is None else previous
    grouped = {}
    for row in rows:
        required = ("input", "rollout_generated_tokens", "rollout_finish_reason", "rollout_sampling_cap")
        if any(key not in row for key in required):
            raise ValueError("explicit generated count, finish reason and effective sampling cap required")
        count, cap, reason = row[required[1]], row[required[3]], row[required[2]]
        if (type(count) is not int or type(cap) is not int or not 0 <= count <= cap
                or cap <= 0 or reason not in ("stop", "length")):
            raise ValueError("invalid or aborted completion; not an eligible predictor observation")
        if reason == "length" and count != cap:
            raise ValueError("length finish must identify its effective generation limit, not a larger configured limit")
        grouped.setdefault(row["input"], []).append((count, reason == "length", cap, reason))
    result = []
    for prompt, observations in grouped.items():
        if len(observations) != responses_per_prompt:
            raise ValueError(f"expected {responses_per_prompt} aligned completions per prompt")
        lengths, censored, caps, reasons = map(list, zip(*observations))
        update = update_tail(previous.get(prompt), lengths, censored, caps)
        result.append(dict(input=prompt, lengths=lengths, censored=censored,
                           sampling_caps=caps, finish_reasons=reasons, **update))
    return {"schema_version": 2, "steps": len({row.get("step", 1) for row in rows}),
            "length_units": "actual_generated_tokens_excluding_padding",
            "censor_semantics": "length finish is a right-censored observation, not a natural tail",
            "responses_per_prompt": responses_per_prompt, "prompt_count": len(result),
            "records": result,
            "limitations": "Preserves prior tail under censoring; does not recover unseen continuations, prove unbiased GRPO or provide quantile coverage guarantees."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--previous-history", type=Path)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [*args.rollout, *([args.previous_history] if args.previous_history else [])]
    if args.output.exists() or args.output.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("refusing overwrite or source mutation")
    rows = [json.loads(line) for path in args.rollout for line in path.read_text().splitlines() if line.strip()]
    previous = {}
    if args.previous_history:
        for record in json.loads(args.previous_history.read_text())["records"]:
            previous[record["input"]] = record.get("tail_prediction", max(record["lengths"]))
    payload = build(rows, previous, args.responses_per_prompt)
    payload["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    payload["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__), Path(__file__).parents[1] / "verl/utils/revision_censored_history.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"prompt_count={payload['prompt_count']} explicit_censor_history={args.output}")


if __name__ == "__main__":
    main()
