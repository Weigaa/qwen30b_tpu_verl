#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="./moe_stats")
    ap.add_argument("--pattern", type=str, default="moe_topk_ids_ep*.json")
    ap.add_argument("--num_experts", type=int, default=128)
    ap.add_argument("--group_size", type=int, default=8, help="8->16组; 16->8组")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--print_steps", action="store_true", help="是否打印每个 step 的极差")
    args = ap.parse_args()

    num_groups = (args.num_experts + args.group_size - 1) // args.group_size

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"Not found: {os.path.join(args.dir, args.pattern)}")

    rank_data = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        dd = {int(k): v for k, v in d.items()}
        rank_data.append(dd)

    # per-step group counts (OLD/NEW)
    old_step_counts = np.zeros((args.max_steps, num_groups), dtype=np.int64)
    new_step_counts = np.zeros((args.max_steps, num_groups), dtype=np.int64)

    for step in range(args.max_steps):
        old_counts = np.zeros(num_groups, dtype=np.int64)
        new_counts = np.zeros(num_groups, dtype=np.int64)

        for d in rank_data:
            rec = d.get(step)
            if rec is None:
                continue
            _, old_ids, new_ids = rec[0], rec[1], rec[2]

            old_arr = np.asarray(old_ids, dtype=np.int64).reshape(-1)
            new_arr = np.asarray(new_ids, dtype=np.int64).reshape(-1)

            # 过滤非法 expert id（保险）
            old_arr = old_arr[(old_arr >= 0) & (old_arr < args.num_experts)]
            new_arr = new_arr[(new_arr >= 0) & (new_arr < args.num_experts)]

            old_g = (old_arr // args.group_size).astype(np.int64)
            new_g = (new_arr // args.group_size).astype(np.int64)

            old_counts += np.bincount(old_g, minlength=num_groups)
            new_counts += np.bincount(new_g, minlength=num_groups)

        old_step_counts[step] = old_counts
        new_step_counts[step] = new_counts

    # 算法1：每step极差再平均
    old_ranges = old_step_counts.max(axis=1) - old_step_counts.min(axis=1)
    new_ranges = new_step_counts.max(axis=1) - new_step_counts.min(axis=1)
    old_avg_range_alg1 = float(old_ranges.mean())
    new_avg_range_alg1 = float(new_ranges.mean())

    # 算法2：先对组计数按step平均，再算极差
    old_mean_counts = old_step_counts.mean(axis=0)
    new_mean_counts = new_step_counts.mean(axis=0)
    old_avg_range_alg2 = float(old_mean_counts.max() - old_mean_counts.min())
    new_avg_range_alg2 = float(new_mean_counts.max() - new_mean_counts.min())

    if args.print_steps:
        for step in range(args.max_steps):
            print(f"step={step:04d}  range(OLD)={int(old_ranges[step])}  range(NEW)={int(new_ranges[step])}")

    # 额外给点分布信息（方便你看波动）
    def stats(x: np.ndarray):
        return {
            "mean": float(x.mean()),
            "min": int(x.min()),
            "max": int(x.max()),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
        }

    print("\n# Summary")
    print(f"num_files={len(files)}  num_experts={args.num_experts}  group_size={args.group_size}  num_groups={num_groups}  steps={args.max_steps}")

    print("\n[Algorithm 1] mean over steps of (max-min):")
    print("OLD:", stats(old_ranges))
    print("NEW:", stats(new_ranges))
    print(f"OLD_avg_range_alg1={old_avg_range_alg1:.6f}")
    print(f"NEW_avg_range_alg1={new_avg_range_alg1:.6f}")

    print("\n[Algorithm 2] range of (mean counts over steps):")
    print(f"OLD_avg_range_alg2={old_avg_range_alg2:.6f}")
    print(f"NEW_avg_range_alg2={new_avg_range_alg2:.6f}")

    # 如果你还想看“平均后每组的token数”，取消注释：
    # print("\nold_mean_counts:", old_mean_counts.tolist())
    # print("new_mean_counts:", new_mean_counts.tolist())


if __name__ == "__main__":
    main()
