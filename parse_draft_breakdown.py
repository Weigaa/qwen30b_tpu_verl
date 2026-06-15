#!/usr/bin/env python3
import argparse
import re
import statistics
from typing import Dict, List


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = int((len(xs) - 1) * p)
    return xs[idx]


def _fmt(v: float) -> str:
    if v != v:
        return "nan"
    return f"{v:.3f}"


def parse_rows(path: str, include_warmup: bool) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            if "breakdown" not in line:
                continue
            if "[DraftTrainer] profile-only breakdown" not in line and "[DraftTrainer] breakdown" not in line:
                continue
            kv = {k: v for k, v in KV_RE.findall(line)}
            tag = kv.get("tag", "")
            if not include_warmup and tag and not tag.startswith("profile-"):
                continue

            row: Dict[str, float] = {}
            for key, value in kv.items():
                if key in {"tag"}:
                    continue
                row[key] = _to_float(value)
            row["tag"] = tag  # type: ignore[assignment]

            # Merged "real stage" times (stage + local sync waits).
            row["real_fwd_ms"] = (
                row.get("sync_fwd_before_ms", 0.0)
                + row.get("fwd_ms", 0.0)
                + row.get("sync_fwd_after_ms", 0.0)
            )
            row["real_bwd_ms"] = (
                row.get("sync_bwd_before_ms", 0.0)
                + row.get("bwd_ms", 0.0)
                + row.get("sync_bwd_after_ms", 0.0)
            )
            row["real_opt_ms"] = (
                row.get("sync_opt_before_ms", 0.0)
                + row.get("opt_ms", 0.0)
                + row.get("sync_opt_after_ms", 0.0)
            )
            rows.append(row)
    return rows


def summarize(rows: List[Dict[str, float]], keys: List[str]) -> None:
    print(f"rows={len(rows)}")
    if not rows:
        return
    for key in keys:
        vals = [r.get(key, float("nan")) for r in rows]
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        print(
            f"{key}: mean={_fmt(statistics.mean(vals))} "
            f"med={_fmt(statistics.median(vals))} "
            f"p90={_fmt(_pct(vals, 0.90))} "
            f"max={_fmt(max(vals))}"
        )

    med_total = statistics.median([r["total_ms"] for r in rows if "total_ms" in r])
    if med_total == med_total and med_total > 0:
        med_real_fwd = statistics.median([r["real_fwd_ms"] for r in rows])
        med_real_bwd = statistics.median([r["real_bwd_ms"] for r in rows])
        med_real_opt = statistics.median([r["real_opt_ms"] for r in rows])
        med_residual = statistics.median([r.get("residual_ms", 0.0) for r in rows])
        print(
            "shares@median: "
            f"real_fwd={100.0 * med_real_fwd / med_total:.1f}% "
            f"real_bwd={100.0 * med_real_bwd / med_total:.1f}% "
            f"real_opt={100.0 * med_real_opt / med_total:.1f}% "
            f"residual={100.0 * med_residual / med_total:.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse DraftTrainer breakdown logs and emit merged real stage timings."
    )
    ap.add_argument("logfile", help="path to training/profile log file")
    ap.add_argument(
        "--include-warmup",
        action="store_true",
        help="include warmup-* tags (default: only profile-* tags)",
    )
    args = ap.parse_args()

    rows = parse_rows(args.logfile, include_warmup=args.include_warmup)
    keys = [
        "real_fwd_ms",
        "real_bwd_ms",
        "real_opt_ms",
        "fwd_ms",
        "bwd_ms",
        "opt_ms",
        "sync_total_ms",
        "residual_ms",
        "total_ms",
    ]
    summarize(rows, keys)


if __name__ == "__main__":
    main()
