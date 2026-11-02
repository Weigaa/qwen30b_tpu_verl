#!/usr/bin/env python3
"""Run a short semantic smoke test with direct DeepSeek HF weight loading."""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BASE_COMPLETION_PROMPT = (
    "An attention function can be described as mapping a query and a set of "
    "key-value pairs to an output, where the query, keys, values, and output "
    "are all vectors. The output is"
)
MATH_PROMPT = (
    "The operation $\\otimes$ is defined for all nonzero numbers by "
    "$a \\otimes b = \\frac{a^{2}}{b}$. Determine "
    "$[(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]$."
)
SECOND_MATH_PROMPT = (
    "If a rectangle has side lengths 7 and 9, what is its area? "
    "Give a short explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--requests-per-rank", type=int, default=2)
    parser.add_argument("--chat-only", action="store_true")
    parser.add_argument("--require-stop", action="store_true")
    parser.add_argument("--require-answer", action="store_true")
    parser.add_argument("--reject-dialogue-continuation", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def text_metrics(text: str) -> dict[str, float | int | bool]:
    visible = [char for char in text if not char.isspace()]
    alpha_numeric = sum(char.isalnum() for char in visible)
    punctuation = sum(char in string.punctuation for char in visible)
    visible_count = len(visible)
    alpha_numeric_ratio = alpha_numeric / visible_count if visible_count else 0.0
    punctuation_ratio = punctuation / visible_count if visible_count else 1.0
    return {
        "visible_chars": visible_count,
        "alpha_numeric_chars": alpha_numeric,
        "alpha_numeric_ratio": alpha_numeric_ratio,
        "punctuation_ratio": punctuation_ratio,
        "semantic_smoke_pass": visible_count > 0 and alpha_numeric_ratio >= 0.20,
    }


def answer_quality_pass(label: str, text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    if label == "math_chat_primary":
        return any(
            answer in compact
            for answer in (r"-\frac{2}{3}", r"-\dfrac{2}{3}", "-2/3", "-0.666")
        )
    if label == "math_chat_secondary":
        return re.search(r"(?<!\d)63(?!\d)", text) is not None
    return True


def main() -> int:
    args = parse_args()
    if (
        args.max_tokens <= 0
        or args.max_model_len <= args.max_tokens
        or args.expert_parallel_size <= 0
        or args.requests_per_rank <= 0
    ):
        raise ValueError("max token settings are invalid")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    math_chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
    )
    second_math_chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SECOND_MATH_PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
    )
    base_completion_prompt = tokenizer(
        BASE_COMPLETION_PROMPT, add_special_tokens=True
    ).input_ids
    if args.chat_only:
        prompt_templates = [math_chat_prompt, second_math_chat_prompt]
    else:
        prompt_templates = [base_completion_prompt, math_chat_prompt]
    if any(
        prompt_ids.count(tokenizer.bos_token_id) != 1
        for prompt_ids in prompt_templates
    ):
        raise ValueError("semantic smoke prompts must contain exactly one BOS token")
    request_count = args.expert_parallel_size * args.requests_per_rank
    prompts = [
        {"prompt_token_ids": prompt_templates[index % len(prompt_templates)]}
        for index in range(request_count)
    ]

    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        load_format="auto",
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        data_parallel_size=args.expert_parallel_size,
        data_parallel_backend="mp",
        enable_expert_parallel=args.expert_parallel_size > 1,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.requests_per_rank,
        max_num_batched_tokens=args.max_model_len,
    )
    load_seconds = time.perf_counter() - load_start

    generate_start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
        ),
    )
    generate_seconds = time.perf_counter() - generate_start

    if args.chat_only:
        labels = [
            "math_chat_primary" if index % 2 == 0 else "math_chat_secondary"
            for index in range(request_count)
        ]
    else:
        labels = [
            "base_completion" if index % 2 == 0 else "math_chat"
            for index in range(request_count)
        ]
    records = []
    for label, output in zip(labels, outputs, strict=True):
        candidate = output.outputs[0]
        dialogue_continuation = "User:" in candidate.text
        semantic_pass = text_metrics(candidate.text)["semantic_smoke_pass"]
        answer_pass = answer_quality_pass(label, candidate.text)
        prompt_bos_count = output.prompt_token_ids.count(tokenizer.bos_token_id)
        semantic_pass = semantic_pass and prompt_bos_count == 1
        if args.require_stop:
            semantic_pass = semantic_pass and candidate.finish_reason == "stop"
        if args.require_answer:
            semantic_pass = semantic_pass and answer_pass
        if args.reject_dialogue_continuation:
            semantic_pass = semantic_pass and not dialogue_continuation
        record = {
            "label": label,
            "prompt_token_ids": output.prompt_token_ids,
            "prompt_bos_count": prompt_bos_count,
            "output_token_ids": candidate.token_ids,
            "text": candidate.text,
            "finish_reason": candidate.finish_reason,
            "dialogue_continuation": dialogue_continuation,
            "answer_quality_pass": answer_pass,
            **text_metrics(candidate.text),
            "semantic_smoke_pass": semantic_pass,
        }
        records.append(record)

    result = {
        "model": args.model,
        "load_format": "auto",
        "expert_parallel_size": args.expert_parallel_size,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "max_tokens": args.max_tokens,
        "records": records,
        "passed": all(record["semantic_smoke_pass"] for record in records),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
