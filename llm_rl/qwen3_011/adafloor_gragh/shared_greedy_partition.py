#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单一分组（所有层复用）——全局贪心均衡：
- 先汇总“所有层 + 所有 seq”的专家计数，得到 per-expert 总量 total_load[E]；
- 以 total_load 做带容量约束的贪心分组（n 组），使各组累积总量尽量均衡；
- 该分组对所有层复用（即每层用同一组划分）。

用法：
  python shared_greedy_partition.py input.json n
输入 JSON（顶层 list，取下标 0）：
[
  {
    "seq1": {"0":[E], "1":[E], ...},
    "seq2": {"0":[E], "1":[E], ...}
  }
]
"""

import json
import sys
from typing import Any, Dict, List, Tuple

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_vec(vec: Any) -> None:
    if not isinstance(vec, list) or len(vec) <= 0:
        raise ValueError(f"期望为非空 list，实际为 {type(vec)}。")

def make_capacities(total_items: int, n: int) -> List[int]:
    """把 total_items 尽量均匀拆成 n 份（前 remainder 份多 1）。"""
    base = total_items // n
    rem = total_items % n
    caps = [base + 1 if i < rem else base for i in range(n)]
    assert sum(caps) == total_items
    return caps

def greedy_partition_with_caps(weights: List[float], n: int) -> Tuple[List[List[int]], List[float]]:
    """
    对权重 weights[E] 做 n 组贪心分区，带人数容量约束：
      - 排序：按 weights 降序（tie 用专家编号）
      - 目标：各组总权重尽量均衡，同时人数不超过 caps
      - key = (sum_i/target_i, size_i/cap_i, 组索引)
    返回：groups（每组专家编号列表），sums（每组总权重）
    """
    E = len(weights)
    if n < 1:
        raise ValueError("n 必须 >= 1")
    caps = make_capacities(E, n)
    total = float(sum(weights))
    denom = total if total > 0 else 1.0
    target = [denom * (cap / float(E if E > 0 else 1)) for cap in caps]

    order = sorted(range(E), key=lambda i: (-float(weights[i]), i))
    groups = [[] for _ in range(n)]
    sums = [0.0 for _ in range(n)]
    sizes = [0 for _ in range(n)]

    for e in order:
        w = float(weights[e])
        best_i, best_key = None, None
        for i in range(n):
            if sizes[i] >= caps[i]:
                continue
            rel = sums[i] / (target[i] if target[i] > 0 else 1.0)
            use = (sizes[i] / caps[i]) if caps[i] > 0 else 1.0
            key = (rel, use, i)
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        if best_i is None:
            continue  # 可能 E<n 导致所有可分配组都满
        groups[best_i].append(e)
        sums[best_i] += w
        sizes[best_i] += 1
    return groups, sums

def main():
    if len(sys.argv) < 3:
        print("用法: python shared_greedy_partition.py input.json n")
        sys.exit(1)

    data_path = sys.argv[1]
    try:
        n_groups = int(sys.argv[2])
    except Exception:
        raise ValueError("n 必须为整数。")
    if n_groups < 1:
        raise ValueError("n 必须 >= 1。")

    data = load_json(data_path)

    # 顶层 list，取第 1 个元素（索引 0）
    if not isinstance(data, list) or len(data) < 1:
        raise ValueError("顶层应为长度>=1的 list。")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("第 1 个元素应为字典：{seq_id -> {layer_id -> [E个数]}}。")

    # 汇总每层（跨 seq）得到 sums_by_layer[lid][i]
    sums_by_layer: Dict[str, List[float]] = {}
    layer_keys = set()
    E_ref = None

    for seq_id, layer_dict in first.items():
        if not isinstance(layer_dict, dict):
            raise ValueError(f"seq_id={seq_id} 的值应为字典。")
        for lid, vec in layer_dict.items():
            ensure_vec(vec)
            E = len(vec)
            if E_ref is None:
                E_ref = E
            elif E != E_ref:
                raise ValueError(f"所有层需拥有一致的专家数以复用同一分组：检测到 {E_ref} vs {E} 于层 {lid}")
            layer_keys.add(lid)
            if lid not in sums_by_layer:
                sums_by_layer[lid] = [0.0] * E
            acc = sums_by_layer[lid]
            for i in range(E):
                acc[i] += float(vec[i])

    if E_ref is None:
        raise ValueError("未解析到任何层数据。")

    # 计算“全局 per-expert 权重” = 所有层累计
    total_load = [0.0] * E_ref
    for lid in sums_by_layer:
        layer_vec = sums_by_layer[lid]
        for i in range(E_ref):
            total_load[i] += layer_vec[i]

    # 基于 total_load 做一次分组，所有层复用
    groups, sums = greedy_partition_with_caps(total_load, n_groups)

    # 统计均衡度
    mn, mx = min(sums) if sums else 0.0, max(sums) if sums else 0.0
    gap = mx - mn
    ratio = (mx / mn) if mn > 0 else float("inf") if mx > 0 else 1.0

    print("# 共享分组（所有层复用）")
    for gi, g in enumerate(groups):
        print(f"group{gi} =", g)

    print("\n# 全层累积负载均衡统计")
    print(f"min_sum={mn:.6f}  max_sum={mx:.6f}  gap={gap:.6f}  ratio={ratio:.6f}")

if __name__ == "__main__":
    main()