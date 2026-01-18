#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
贪心分组（输出 n 个 list），兼容任意专家数 E（由每层向量长度决定）。
新增：比较“贪心分配 vs 线性分配”的不均衡情况（每层的 min / max / gap）。

用法：
  python layerwise_greedy_groups_anyE_n.py input.json n
"""

import json
import sys
from typing import Any, Dict, List, Tuple

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_vec(vec: Any) -> None:
    if not isinstance(vec, list) or len(vec) <= 0:
        raise ValueError(f"期望为非空 list，实际为 {type(vec)} 且长度 {len(vec) if isinstance(vec, list) else 'N/A'}。")

def make_capacities(total_items: int, n: int) -> List[int]:
    """把 total_items 尽量均匀拆成 n 份（前 remainder 份多 1）。允许 n > total_items（会产生若干 0 容量组）。"""
    base = total_items // n
    rem = total_items % n
    caps = [base + 1 if i < rem else base for i in range(n)]
    assert sum(caps) == total_items
    return caps

def greedy_assign_n_groups(vec: List[float], n: int) -> Tuple[List[List[int]], List[float]]:
    """
    贪心分配：
      - 对 vec（长度 E）按值降序排序；
      - 每次把专家放到“相对目标和最小、且未满容量”的组；
        key = (sum_i / target_sum_i, size_i / caps[i], 组索引)
    返回 groups（索引列表）和 sums（各组总和）。
    """
    E = len(vec)
    if n < 1:
        raise ValueError("组数 n 必须 >= 1。")

    caps = make_capacities(E, n)
    total = float(sum(vec))
    denom = total if total > 0 else 1.0
    target_sums = [denom * (cap / float(E if E > 0 else 1)) for cap in caps]

    # 按 vec 值降序（tie: expert id）
    ranking = sorted(range(E), key=lambda i: (-float(vec[i]), i))

    groups = [[] for _ in range(n)]
    sums = [0.0 for _ in range(n)]
    sizes = [0 for _ in range(n)]

    for e in ranking:
        w = float(vec[e])
        best_idx, best_key = None, None
        for i in range(n):
            if sizes[i] >= caps[i]:
                continue
            rel_load = sums[i] / (target_sums[i] if target_sums[i] > 0 else 1.0)
            usage = (sizes[i] / caps[i]) if caps[i] > 0 else 1.0
            key = (rel_load, usage, i)
            if best_key is None or key < best_key:
                best_key, best_idx = key, i

        if best_idx is None:
            # 可能出现 E < n，一些组容量为 0；当所有可分配组已满则跳过
            continue

        groups[best_idx].append(e)
        sums[best_idx] += w
        sizes[best_idx] += 1

    return groups, sums

def linear_assign_n_groups(vec: List[float], n: int) -> Tuple[List[List[int]], List[float]]:
    """
    线性分配：
      - 按专家编号升序把 E 个专家切成 n 段（容量由 make_capacities 决定）；
      - group0: 连续前 cap[0] 个专家；group1: 下一个 cap[1] 个；以此类推。
    返回 groups（索引列表）和 sums（各组总和）。
    """
    E = len(vec)
    caps = make_capacities(E, n)
    groups = [[] for _ in range(n)]
    sums = [0.0 for _ in range(n)]

    start = 0
    for i, cap in enumerate(caps):
        end = start + cap
        if cap > 0:
            groups[i] = list(range(start, end))
            sums[i] = float(sum(vec[j] for j in range(start, end)))
        start = end
    return groups, sums

def imbalance_stats(sums: List[float]) -> Tuple[float, float, float]:
    """返回 (min, max, gap=max-min)"""
    if not sums:
        return 0.0, 0.0, 0.0
    mn = min(sums)
    mx = max(sums)
    return mn, mx, (mx - mn)

def main():
    if len(sys.argv) < 3:
        print("用法: python layerwise_greedy_groups_anyE_n.py input.json n")
        sys.exit(1)

    data_path = sys.argv[1]
    try:
        n_groups = int(sys.argv[2])
    except Exception:
        raise ValueError("第二个参数 n 必须为整数。")
    if n_groups < 1:
        raise ValueError("n 必须 >= 1。")

    data = load_json(data_path)

    # 顶层 list
    if not isinstance(data, list) or len(data) < 1:
        raise ValueError("顶层应为长度>=1的 list。")

    # 汇总每层（跨 seq）得到 vec(E)；允许不同层有不同 E，但同一层必须一致
    sums_by_layer: Dict[str, List[float]] = {}
    layer_len: Dict[str, int] = {}
    layer_keys_set = set()

    # 汇总每层（跨顶层 list 的 16 个元素）得到 vec(E)
    for item in data:  # item: depth2 dict, has 2 keys
        if not isinstance(item, dict):
            continue
        for seq_id, layer_dict in item.items():  # seq_id 这里其实是 depth2 的 key（2个分支）
            if not isinstance(layer_dict, dict):
                continue
            for lid, vec in layer_dict.items():  # lid: 48 layers
                ensure_vec(vec)
                E = len(vec)
                layer_keys_set.add(lid)
                if lid not in sums_by_layer:
                    sums_by_layer[lid] = [0.0] * E
                    layer_len[lid] = E
                else:
                    if len(sums_by_layer[lid]) != E:
                        raise ValueError(f"层 {lid} 的专家数不一致：之前为 {layer_len[lid]}，现在是 {E}。")
                acc = sums_by_layer[lid]
                for i in range(E):
                    acc[i] += float(vec[i])

    # 层顺序（尽量按数字层号）
    try:
        layer_keys_sorted = sorted(layer_keys_set, key=lambda x: int(x))
    except Exception:
        layer_keys_sorted = sorted(layer_keys_set)

    # 准备 n 个“按层收集”的列表（贪心分配的结果）
    groups_by_layer_lists: List[List[List[int]]] = [[] for _ in range(n_groups)]

    # 统计用
    greedy_mins, greedy_maxs, greedy_gaps = [], [], []
    linear_mins, linear_maxs, linear_gaps = [], [], []

    print("# layer  贪心[min  max  gap]    线性[min  max  gap]")
    for lid in layer_keys_sorted:
        vec = sums_by_layer[lid]

        # 贪心
        g_groups, g_sums = greedy_assign_n_groups(vec, n_groups)
        g_min, g_max, g_gap = imbalance_stats(g_sums)
        greedy_mins.append(g_min); greedy_maxs.append(g_max); greedy_gaps.append(g_gap)

        # 线性
        l_groups, l_sums = linear_assign_n_groups(vec, n_groups)
        l_min, l_max, l_gap = imbalance_stats(l_sums)
        linear_mins.append(l_min); linear_maxs.append(l_max); linear_gaps.append(l_gap)

        # 收集贪心的分组（按层）输出为 n 个 list
        for gi in range(n_groups):
            groups_by_layer_lists[gi].append(g_groups[gi] if gi < len(g_groups) else [])

        print(f"{lid:>6}  "
              f"{g_min:.6f} {g_max:.6f} {g_gap:.6f}    "
              f"{l_min:.6f} {l_max:.6f} {l_gap:.6f}")

    # 跨层平均
    def avg(x: List[float]) -> float:
        return sum(x) / len(x) if x else 0.0

    print("\n# 跨层平均（Arithmetic Mean）")
    print(f"贪心  min_avg={avg(greedy_mins):.6f}  max_avg={avg(greedy_maxs):.6f}  gap_avg={avg(greedy_gaps):.6f}")
    print(f"线性  min_avg={avg(linear_mins):.6f}  max_avg={avg(linear_maxs):.6f}  gap_avg={avg(linear_gaps):.6f}")

    # === 最终输出（n 个 list；贪心分配） ===
    for gi, lst in enumerate(groups_by_layer_lists):
        print(f"group{gi}_by_layer =", lst)

if __name__ == "__main__":
    main()
