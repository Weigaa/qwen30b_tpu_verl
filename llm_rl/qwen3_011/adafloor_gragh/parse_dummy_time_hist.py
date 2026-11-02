#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import math
from collections import defaultdict

# 1) 兼容 ANSI 颜色码
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# 2) 匹配 dummy batch time / dummy run time（你也可以按需改关键词）
#    示例: "rank 5 dummy batch time: 0.04438042640686035"
PATTERN = re.compile(
    r"\brank\s+(?P<rank>\d+)\s+dummy\s+(?:batch|run)\s+time:\s+(?P<t>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def compute_bins(min_v: float, max_v: float, n_bins: int = 10):
    """
    返回 bin_edges 长度 n_bins+1
    区间定义： [e0,e1), [e1,e2), ... [e9,e10]（最后一个右闭）
    """
    if min_v == max_v:
        # 退化情况：全都一样，给一个“宽度=1”的虚拟区间，全部落在最后一个bin也行
        return [min_v + (i / n_bins) for i in range(n_bins + 1)]
    width = (max_v - min_v) / n_bins
    return [min_v + i * width for i in range(n_bins + 1)]

def bin_index(v: float, edges):
    # edges 长度 n+1
    n = len(edges) - 1
    if n <= 0:
        return 0
    if v <= edges[0]:
        return 0
    if v >= edges[-1]:
        return n - 1
    # [e0,e1), ... [e(n-1), e(n))  (最后一个实际是 [e(n-1), e(n)] )
    width = (edges[-1] - edges[0]) / n if edges[-1] != edges[0] else 0.0
    if width == 0.0:
        return n - 1
    idx = int((v - edges[0]) / width)
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx

def main():
    if len(sys.argv) < 2:
        print("用法: python parse_dummy_time_hist.py /path/to/your.log", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]

    # -------- Pass 1: 统计每个 rank 的 min/max & count --------
    min_t = {}
    max_t = {}
    cnt = defaultdict(int)

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            line = strip_ansi(line)
            m = PATTERN.search(line)
            if not m:
                continue
            r = int(m.group("rank"))
            t = safe_float(m.group("t"))
            if t is None or math.isnan(t) or math.isinf(t):
                continue
            cnt[r] += 1
            if r not in min_t or t < min_t[r]:
                min_t[r] = t
            if r not in max_t or t > max_t[r]:
                max_t[r] = t

    ranks = sorted(cnt.keys())
    if not ranks:
        print("没有匹配到任何 dummy time 记录。你可能需要调整 PATTERN 里的关键词。", file=sys.stderr)
        sys.exit(2)

    # 为每个 rank 生成 10 等分边界
    edges_by_rank = {r: compute_bins(min_t[r], max_t[r], 10) for r in ranks}
    hist_by_rank = {r: [0] * 10 for r in ranks}

    # -------- Pass 2: 统计每条记录落在哪个 bin --------
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            line = strip_ansi(line)
            m = PATTERN.search(line)
            if not m:
                continue
            r = int(m.group("rank"))
            t = safe_float(m.group("t"))
            if r not in edges_by_rank or t is None or math.isnan(t) or math.isinf(t):
                continue
            idx = bin_index(t, edges_by_rank[r])
            hist_by_rank[r][idx] += 1

    # -------- 输出（人类可读）--------
    print("\n=== Per-rank dummy time summary (10 bins) ===")
    for r in ranks:
        edges = edges_by_rank[r]
        hist = hist_by_rank[r]
        print(f"\n[rank {r}] count={cnt[r]}  min={min_t[r]:.10g}  max={max_t[r]:.10g}")
        # 打印 10 个区间
        for i in range(10):
            left = edges[i]
            right = edges[i + 1]
            # 最后一个右闭
            bracket = ")" if i < 9 else "]"
            print(f"  bin{i:02d}: [{left:.10g}, {right:.10g}{bracket}  -> {hist[i]}")

    # -------- 同时输出 CSV（方便你直接导入表格/画图）--------
    # CSV: rank,bin_idx,left,right,count
    print("\n=== CSV ===")
    print("rank,bin_idx,left,right,count")
    for r in ranks:
        edges = edges_by_rank[r]
        hist = hist_by_rank[r]
        for i in range(10):
            print(f"{r},{i},{edges[i]},{edges[i+1]},{hist[i]}")

if __name__ == "__main__":
    main()
