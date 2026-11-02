#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
热到冷顺序装箱分组（输出 n 个 list），兼容任意专家数 E（由每层向量长度决定）。

目标：
  - 每层把专家按权重从热到冷排序；
  - 按组容量 make_capacities(E, n)：
      先填满 group0，再填 group1 ...；
  - 保证 group0 在每一层永远是“最热”的组（拿到top最热的一段专家）。

用法：
  python layerwise_hotfill_groups_anyE_n.py input.json n
"""

import json
import sys
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_vec(vec: Any) -> None:
    if not isinstance(vec, list) or len(vec) <= 0:
        raise ValueError(
            f"期望为非空 list，实际为 {type(vec)} 且长度 {len(vec) if isinstance(vec, list) else 'N/A'}。"
        )


def make_capacities(total_items: int, n: int) -> List[int]:
    """把 total_items 尽量均匀拆成 n 份（前 remainder 份多 1）。允许 n > total_items（会产生若干 0 容量组）。"""
    base = total_items // n
    rem = total_items % n
    caps = [base + 1 if i < rem else base for i in range(n)]
    assert sum(caps) == total_items
    return caps


def hot_to_cold_fill_assign_n_groups(vec: List[float], n: int) -> Tuple[List[List[int]], List[float]]:
    """
    热到冷顺序装箱：
      - 对 vec（长度 E）按值降序排序，得到 ranking（热->冷）；
      - 按 caps 依次填充：group0 先拿 cap[0] 个最热专家，再 group1 拿 cap[1] 个...。
    返回 groups（索引列表）和 sums（各组总和）。
    """
    E = len(vec)
    if n < 1:
        raise ValueError("组数 n 必须 >= 1。")

    caps = make_capacities(E, n)

    ranking = sorted(range(E), key=lambda i: (-float(vec[i]), i))  # 热->冷

    groups = [[] for _ in range(n)]
    sums = [0.0 for _ in range(n)]

    pos = 0
    for gi, cap in enumerate(caps):
        if cap <= 0:
            continue
        seg = ranking[pos:pos + cap]
        groups[gi] = seg
        sums[gi] = float(sum(vec[e] for e in seg))
        pos += cap

    return groups, sums


def linear_assign_n_groups(vec: List[float], n: int) -> Tuple[List[List[int]], List[float]]:
    """线性分配：按专家编号切段（对照用）"""
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
        print("用法: python layerwise_hotfill_groups_anyE_n.py input.json n")
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

    for item in data:  # item: depth2 dict, has 2 keys
        if not isinstance(item, dict):
            continue
        for _, layer_dict in item.items():
            if not isinstance(layer_dict, dict):
                continue
            for lid, vec in layer_dict.items():
                ensure_vec(vec)
                E = len(vec)
                layer_keys_set.add(lid)
                if lid not in sums_by_layer:
                    sums_by_layer[lid] = [0.0] * E
                    layer_len[lid] = E
                else:
                    if len(sums_by_layer[lid]) != E:
                        raise ValueError(
                            f"层 {lid} 的专家数不一致：之前为 {layer_len[lid]}，现在是 {E}。"
                        )
                acc = sums_by_layer[lid]
                for i in range(E):
                    acc[i] += float(vec[i])

    # 层顺序（尽量按数字层号）
    try:
        layer_keys_sorted = sorted(layer_keys_set, key=lambda x: int(x))
    except Exception:
        layer_keys_sorted = sorted(layer_keys_set)

    # 准备 n 个“按层收集”的列表（热到冷装箱的结果）
    groups_by_layer_lists: List[List[List[int]]] = [[] for _ in range(n_groups)]

    hotfill_mins, hotfill_maxs, hotfill_gaps = [], [], []
    linear_mins, linear_maxs, linear_gaps = [], [], []

    print("# layer  热到冷装箱[min  max  gap]    线性[min  max  gap]")
    for lid in layer_keys_sorted:
        vec = sums_by_layer[lid]

        # 热到冷装箱（你要的方案）
        h_groups, h_sums = hot_to_cold_fill_assign_n_groups(vec, n_groups)
        h_min, h_max, h_gap = imbalance_stats(h_sums)
        hotfill_mins.append(h_min); hotfill_maxs.append(h_max); hotfill_gaps.append(h_gap)

        # 线性（对照）
        l_groups, l_sums = linear_assign_n_groups(vec, n_groups)
        l_min, l_max, l_gap = imbalance_stats(l_sums)
        linear_mins.append(l_min); linear_maxs.append(l_max); linear_gaps.append(l_gap)

        # 收集热到冷装箱的分组（按层）输出为 n 个 list
        for gi in range(n_groups):
            groups_by_layer_lists[gi].append(h_groups[gi] if gi < len(h_groups) else [])

        print(f"{lid:>6}  "
              f"{h_min:.6f} {h_max:.6f} {h_gap:.6f}    "
              f"{l_min:.6f} {l_max:.6f} {l_gap:.6f}")

    def avg(x: List[float]) -> float:
        return sum(x) / len(x) if x else 0.0

    print("\n# 跨层平均（Arithmetic Mean）")
    print(f"热到冷装箱  min_avg={avg(hotfill_mins):.6f}  max_avg={avg(hotfill_maxs):.6f}  gap_avg={avg(hotfill_gaps):.6f}")
    print(f"线性        min_avg={avg(linear_mins):.6f}  max_avg={avg(linear_maxs):.6f}  gap_avg={avg(linear_gaps):.6f}")

    # === 最终输出（n 个 list；热到冷装箱） ===
    for gi, lst in enumerate(groups_by_layer_lists):
        print(f"rank{gi} =", lst)


if __name__ == "__main__":
    main()
