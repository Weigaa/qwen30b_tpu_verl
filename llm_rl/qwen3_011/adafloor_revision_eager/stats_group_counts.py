#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./moe_stats", help="存放 moe_topk_ids_ep*.json 的目录")
    parser.add_argument("--pattern", type=str, default="moe_topk_ids_ep*.json")
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=8, help="每组包含多少个专家；8->16组，16->8组")
    parser.add_argument("--max_steps", type=int, default=1000, help="只打印前多少个 step（默认 1000）")
    args = parser.parse_args()

    num_groups = (args.num_experts + args.group_size - 1) // args.group_size

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"在 {args.dir} 下找不到 {args.pattern}")

    # 读入所有 rank 的 json：rank_data[i][step] = [layer_id, old_ids, new_ids]
    rank_data = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        # key 可能是字符串 step
        dd = {}
        for k, v in d.items():
            try:
                dd[int(k)] = v
            except Exception:
                continue
        rank_data.append((os.path.basename(fp), dd))

    # 打印 header
    group_ranges = [
        f"G{g}({g*args.group_size:03d}-{min((g+1)*args.group_size-1, args.num_experts-1):03d})"
        for g in range(num_groups)
    ]
    print(f"# files={len(files)}  num_experts={args.num_experts}  group_size={args.group_size}  num_groups={num_groups}")
    print("# group order:", " ".join(group_ranges))
    print("# Each count is token-expert edges (flatten(topk_ids)), summed over all ranks in that step.\n")

    for step in range(args.max_steps):
        old_counts = np.zeros(num_groups, dtype=np.int64)
        new_counts = np.zeros(num_groups, dtype=np.int64)

        layer_ids_seen = []

        for fname, d in rank_data:
            rec = d.get(step, None)
            if rec is None:
                continue

            layer_id, old_ids, new_ids = rec[0], rec[1], rec[2]
            layer_ids_seen.append(layer_id)

            # old_ids/new_ids: 可能是 [num_tokens, topk] 的 list
            old_arr = np.asarray(old_ids, dtype=np.int64).reshape(-1)
            new_arr = np.asarray(new_ids, dtype=np.int64).reshape(-1)

            # 过滤非法 id（可选）
            old_arr = old_arr[(old_arr >= 0) & (old_arr < args.num_experts)]
            new_arr = new_arr[(new_arr >= 0) & (new_arr < args.num_experts)]

            old_g = (old_arr // args.group_size).astype(np.int64)
            new_g = (new_arr // args.group_size).astype(np.int64)

            old_counts += np.bincount(old_g, minlength=num_groups)
            new_counts += np.bincount(new_g, minlength=num_groups)

        # layer_id 可能所有 rank 一致；不一致就只打印集合
        layer_info = layer_ids_seen[0] if layer_ids_seen and all(x == layer_ids_seen[0] for x in layer_ids_seen) else sorted(set(layer_ids_seen))

        print(f"step={step:04d}  layer={layer_info}  OLD:", " ".join(map(str, old_counts.tolist())))
        print(f"step={step:04d}  layer={layer_info}  NEW:", " ".join(map(str, new_counts.tolist())))


if __name__ == "__main__":
    main()
