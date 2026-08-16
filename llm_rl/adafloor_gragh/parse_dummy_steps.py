#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

DUMMY_RE = re.compile(r"\brank\s+(\d+)\s+dummy_times:\s+(\d+)\b")
TOTAL_RE = re.compile(r"\brank\s+(\d+)\s+total_step_times:\s+(\d+)\b")
STEP_SPLIT_RE = re.compile(r"Dumped generations to\s+(.+)$")


def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE_RE.sub("", s)


@dataclass
class StepSnapshot:
    dummy_cum: Dict[int, int]
    total_cum: Dict[int, int]


def parse_log_to_steps(log_path: str) -> List[StepSnapshot]:
    """
    Split log into steps by 'Dumped generations to ...' boundary.
    Each step records the LAST seen cumulative dummy_times/total_step_times per rank.
    """
    steps: List[StepSnapshot] = []
    cur_dummy: Dict[int, int] = {}
    cur_total: Dict[int, int] = {}

    def finalize_step():
        if cur_dummy or cur_total:
            steps.append(StepSnapshot(dummy_cum=dict(cur_dummy), total_cum=dict(cur_total)))

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw).strip()

            # boundary => finalize current step
            if STEP_SPLIT_RE.search(line):
                finalize_step()
                cur_dummy.clear()
                cur_total.clear()
                continue

            m = DUMMY_RE.search(line)
            if m:
                rank = int(m.group(1))
                val = int(m.group(2))
                cur_dummy[rank] = val
                continue

            m = TOTAL_RE.search(line)
            if m:
                rank = int(m.group(1))
                val = int(m.group(2))
                cur_total[rank] = val
                continue

    finalize_step()
    return steps


def _delta_with_reset(cum_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cumulative counters to per-step delta.
    If delta < 0 (counter reset), treat delta as current cumulative value.
    """
    # difference along steps
    delta = cum_df.diff(axis=1)
    # step0 baseline: from 0
    delta.iloc[:, 0] = cum_df.iloc[:, 0]

    # handle reset (negative delta)
    reset_mask = delta < 0
    if reset_mask.any().any():
        delta = delta.mask(reset_mask, cum_df)

    # final: NaN -> 0
    delta = delta.fillna(0)
    return delta


def build_tables(steps: List[StepSnapshot]) -> Tuple[pd.DataFrame, List[int]]:
    if not steps:
        return pd.DataFrame(), []

    all_ranks = sorted(
        set().union(*[set(s.dummy_cum.keys()) | set(s.total_cum.keys()) for s in steps])
    )
    num_steps = len(steps)

    dummy_cum = pd.DataFrame(index=all_ranks, columns=range(num_steps), dtype="float64")
    total_cum = pd.DataFrame(index=all_ranks, columns=range(num_steps), dtype="float64")

    for i, snap in enumerate(steps):
        for r, v in snap.dummy_cum.items():
            dummy_cum.loc[r, i] = v
        for r, v in snap.total_cum.items():
            total_cum.loc[r, i] = v

    # ✅ 关键修复：先 ffill，再把 leading NaN 补 0（不然首次出现的 rank 会被吞）
    dummy_cum = dummy_cum.ffill(axis=1).fillna(0)
    total_cum = total_cum.ffill(axis=1).fillna(0)

    dummy_delta = _delta_with_reset(dummy_cum).astype("int64")
    total_delta = _delta_with_reset(total_cum).astype("int64")

    # 一维数组：每 step 的真实迭代次数
    # 用 total_delta 各 rank 的中位数（更抗缺失/个别异常）
    step_total_exec: List[int] = []
    for i in range(num_steps):
        vals = total_delta.iloc[:, i].to_numpy()
        vals = vals[vals >= 0]
        step_total_exec.append(int(np.median(vals)) if vals.size else 0)

    return dummy_delta, step_total_exec


def main():
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="path to your log file")
    ap.add_argument("--out_csv", default="dummy_runs_by_rank_step.csv")
    ap.add_argument("--out_json", default="step_total_exec.json")
    args = ap.parse_args()

    steps = parse_log_to_steps(args.log_path)
    df_dummy, step_total = build_tables(steps)

    print(f"Parsed steps: {len(steps)}")
    print(f"Ranks: {list(df_dummy.index)}")

    print("\n[2D] dummy_runs_by_rank_step (head):")
    print(df_dummy.head())

    print("\n[1D] step_total_exec:")
    print(step_total)

    df_dummy.to_csv(args.out_csv, index=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(step_total, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()