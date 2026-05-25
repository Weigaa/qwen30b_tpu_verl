# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# A small wrapper that adapts verl's GSM8K reward scorer to the custom reward
# function signature used by the local GRPO launch scripts.

import os
import re
from decimal import Decimal, InvalidOperation

from verl.utils.reward_score import gsm8k


_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")
_NUMBER_RE = re.compile(r"-?[0-9.,]+")


def _normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, (list, tuple)):
        if not ground_truth:
            return ""
        ground_truth = ground_truth[0]
    ground_truth = str(ground_truth).strip()

    final_match = _FINAL_ANSWER_RE.search(ground_truth)
    if final_match:
        ground_truth = final_match.group(1)
    else:
        numbers = _NUMBER_RE.findall(ground_truth)
        if numbers:
            ground_truth = numbers[-1]

    return ground_truth.replace(",", "").replace("$", "").strip()


def _normalize_number(value):
    value = str(value).replace(",", "").replace("$", "").strip()
    while value.endswith("."):
        value = value[:-1]
    return value


def _numbers_equal(lhs, rhs):
    lhs = _normalize_number(lhs)
    rhs = _normalize_number(rhs)
    try:
        return Decimal(lhs) == Decimal(rhs)
    except InvalidOperation:
        return lhs == rhs


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    method = os.getenv("GSM8K_REWARD_METHOD", "flexible").strip() or "flexible"
    if method not in {"strict", "flexible"}:
        raise ValueError(f"Unsupported GSM8K_REWARD_METHOD={method!r}; expected strict or flexible")
    answer = gsm8k.extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0.0
    return 1.0 if _numbers_equal(answer, _normalize_ground_truth(ground_truth)) else 0.0
