#!/usr/bin/env python3
"""Legacy rank-peak proxy, retained for historical reproducibility only.

Do not use this proxy for paper evidence: retained peaks are response-only,
local_tail is not response-level, and shared caps can misrepresent Planned-F4.
Use analyze_hpca_prediction_profiles.py with its explicit manifest instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FLOORS = (2, 4, 8, 16)
DEFAULT_FACTORS = (1.0, 1.10, 1.16, 1.25, 1.40)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_caps(value: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in value.split(","):
        floor_text, cap_text = item.split(":", 1)
        floor, cap = int(floor_text), float(cap_text)
        if floor not in FLOORS or cap <= 0:
            raise ValueError(f"invalid cap entry {item!r}")
        result[floor] = cap
    if set(result) != set(FLOORS):
        raise ValueError(f"caps must contain floors {FLOORS}, got {sorted(result)}")
    return result


def load_steps(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path}: every step must be an object")
    return payload


def rank_peaks(step: dict[str, Any]) -> tuple[dict[int, float], dict[int, float]]:
    adjusted = step.get("rank_adjusted_peak_loads")
    response = step.get("rank_peak_loads")
    if not isinstance(adjusted, dict) or not isinstance(response, dict):
        raise ValueError("step is missing rank_adjusted_peak_loads or rank_peak_loads")
    adjusted_out = {int(key): float(value) for key, value in adjusted.items()}
    response_out = {int(key): float(value) for key, value in response.items()}
    if sorted(adjusted_out) != sorted(response_out) or not adjusted_out:
        raise ValueError("rank peak fields have inconsistent rank sets")
    return adjusted_out, response_out


def scenario_peak(
    step: dict[str, Any], factor: float, mode: str
) -> dict[int, float]:
    adjusted, response = rank_peaks(step)
    ranks = sorted(adjusted)
    hottest = max(ranks, key=lambda rank: adjusted[rank])
    result: dict[int, float] = {}
    for rank in ranks:
        scale = factor if mode == "global" else 1.0
        if mode == "hottest_rank" and rank == hottest:
            scale = factor
        if mode == "local_tail" and rank == hottest:
            # The plan only stores rank peaks, not every sampled continuation.
            # Applying the factor to the response peak is a conservative local
            # tail proxy and is reported as such in the output.
            scale = factor
        result[rank] = adjusted[rank] + (scale - 1.0) * response[rank]
    return result


def evaluate_step(
    step: dict[str, Any], caps: dict[int, float], factor: float, mode: str
) -> dict[str, Any]:
    peaks = scenario_peak(step, factor, mode)
    current = int(step["selected_floor"])
    feasible = [floor for floor in FLOORS if floor >= current and max(peaks.values()) <= caps[floor]]
    if feasible:
        selected = min(feasible)
        if selected == current:
            disposition = "safe_current_floor"
        elif selected == 16:
            disposition = "full16_fallback" if current != 16 else "safe_current_floor"
        else:
            disposition = "safe_floor_escalation"
    else:
        selected = None
        disposition = "unsafe_workload_rejection"
    return {
        "step": int(step["step"]),
        "nominal_floor": current,
        "selected_floor_under_stress": selected,
        "factor": factor,
        "scenario": mode,
        "max_adjusted_rank_peak_tokens": max(peaks.values()),
        "floor_caps": {str(floor): caps[floor] for floor in FLOORS},
        "disposition": disposition,
        "release_area_rank_token_proxy": float(step.get("release_area", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", action="append", nargs=3, required=True,
        metavar=("LABEL", "PLAN_JSON", "HISTORY_JSON"),
    )
    parser.add_argument("--caps", default="2:131072,4:280576,8:315648,16:380800")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factors", default=",".join(str(x) for x in DEFAULT_FACTORS))
    parser.add_argument(
        "--allow-missing-history", action="store_true",
        help="Allow legacy paper plans that did not retain compact history. "
        "The rank-peak fields remain the stress source and the omission is recorded.",
    )
    args = parser.parse_args()

    caps = parse_caps(args.caps)
    factors = tuple(float(item) for item in args.factors.split(","))
    if not factors or any(not math.isfinite(x) or x < 1.0 for x in factors):
        raise SystemExit("factors must be finite values >= 1")

    cases: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for raw_case in args.case:
        label, plan_text, history_text = raw_case
        plan_path, history_path = Path(plan_text), Path(history_text)
        if not plan_path.is_file() or (str(history_path) != "-" and not history_path.is_file()):
            raise SystemExit(f"missing explicit input for case {label}: {plan_path} {history_path}")
        plan = load_steps(plan_path)
        history_records: int | None
        history_digest: str | None
        if str(history_path) == "-":
            if not args.allow_missing_history:
                raise SystemExit(f"{label}: history is omitted; pass --allow-missing-history explicitly")
            history_records = None
            history_digest = None
        else:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, dict) or not isinstance(history.get("records"), list):
                raise SystemExit(f"{history_path}: invalid compact planning history")
            history_records = len(history["records"])
            history_digest = sha256(history_path)
        cases.append({
            "label": label,
            "plan": str(plan_path),
            "history": None if str(history_path) == "-" else str(history_path),
            "plan_sha256": sha256(plan_path),
            "history_sha256": history_digest,
            "history_records": history_records,
        })
        for factor in factors:
            for mode in ("global", "hottest_rank", "local_tail"):
                for step in plan:
                    records.append({
                        "case": label,
                        **evaluate_step(step, caps, factor, mode),
                    })

    counts: dict[str, int] = {}
    for record in records:
        key = record["disposition"]
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "schema_version": 1,
        "status": "LEGACY_PROXY_NOT_PAPER_EVIDENCE",
        "tool": "analyze_hpca_prediction_robustness",
        "model": "rank_peak_conservative_response_inflation",
        "caps": {str(floor): value for floor, value in caps.items()},
        "factors": list(factors),
        "cases": cases,
        "disposition_counts": counts,
        "records": records,
        "claim_boundary": (
            "Deprecated response-only rank proxy, not paper-qualified evidence. "
            "It omits prompt KV and does not implement response-level local tails. "
            "It does not generate tokens, "
            "measure runtime latency, or claim that a stochastic response follows "
            "the inflated trajectory. Runtime safety is supported separately by "
            "transition, recovery, and sidecar lifecycle artifacts."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hpca_prediction_robustness.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HPCA prediction robustness stress",
        "",
        "Deprecated response-only rank proxy; excludes prompt KV and is not paper-qualified evidence. Use the response-profile analyzer instead.",
        "",
        "| Disposition | Count |",
        "|---|---:|",
    ]
    for key in sorted(counts):
        lines.append(f"| {key} | {counts[key]} |")
    lines.extend([
        "",
        "| Case | Scenario | Factor | Step | Nominal floor | Stress floor | Max adjusted peak | Disposition |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in records:
        lines.append(
            f"| {row['case']} | {row['scenario']} | {row['factor']:.2f} | {row['step']} | "
            f"{row['nominal_floor']} | {row['selected_floor_under_stress'] or '-'} | "
            f"{row['max_adjusted_rank_peak_tokens']:.1f} | {row['disposition']} |"
        )
    lines += ["", "Input provenance:", ""]
    lines += [f"- `{case['label']}` plan SHA256 `{case['plan_sha256']}`, history SHA256 `{case['history_sha256']}`" for case in cases]
    (args.output_dir / "hpca_prediction_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status=LEGACY_PROXY_NOT_PAPER_EVIDENCE output={json_path}")


if __name__ == "__main__":
    main()
