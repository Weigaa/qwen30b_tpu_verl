#!/usr/bin/env python3
"""Build paired TailBatch plans from one immutable rollout history.

The builder keeps step membership identical across the Full16 and AdaFloor
variants.  Only the within-step rank mapping and selected shrink floor differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_mode1_length_sorted_e2e_plan import (
    _read_baseline_history,
    _read_ema_baseline_stats,
    _solve_group,
    _write_outputs,
)
from build_mode1_optimized_rank_plan import (
    PromptStats,
    _map_stats_to_dataset,
    _prompt_to_input,
)


PLAN_FILENAMES = (
    "length_sorted_train.parquet",
    "length_sorted_rank_plan.json",
    "length_sorted_rank_plan_summary.json",
    "length_sorted_length_oracle.json",
)
DEFAULT_FLOOR_KV_CAPS = {
    2: 131072.0,
    4: 280576.0,
    8: 315648.0,
    16: 380800.0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_tailbatch_groups(
    stats: list[PromptStats],
    *,
    pool_count: int = 4,
    pool_size: int = 40,
    short_round_size: int = 32,
) -> tuple[list[list[PromptStats]], list[dict[str, Any]]]:
    expected = pool_count * pool_size
    if len(stats) != expected:
        raise ValueError(f"expected {expected} prompt stats, got {len(stats)}")
    if not 0 < short_round_size < pool_size:
        raise ValueError(
            f"short_round_size must be in (0, {pool_size}), got {short_round_size}"
        )

    canonical = sorted(stats, key=lambda item: int(item.source_idx))
    source_indices = [int(item.source_idx) for item in canonical]
    if len(set(source_indices)) != expected:
        raise ValueError("TailBatch source indices must be unique")

    short_rounds: list[list[PromptStats]] = []
    long_queue: list[PromptStats] = []
    pools: list[dict[str, Any]] = []
    for pool_index in range(pool_count):
        pool = canonical[pool_index * pool_size:(pool_index + 1) * pool_size]
        ranked = sorted(pool, key=lambda item: (float(item.load), int(item.source_idx)))
        short_round = ranked[:short_round_size]
        deferred = ranked[short_round_size:]
        short_rounds.append(short_round)
        long_queue.extend(deferred)
        pools.append(
            {
                "pool": pool_index + 1,
                "canonical_source_indices": [int(item.source_idx) for item in pool],
                "short_source_indices": [
                    int(item.source_idx) for item in short_round
                ],
                "deferred_source_indices": [
                    int(item.source_idx) for item in deferred
                ],
                "short_predicted_tail_range": [
                    float(min(item.load for item in short_round)),
                    float(max(item.load for item in short_round)),
                ],
                "deferred_predicted_tail_range": [
                    float(min(item.load for item in deferred)),
                    float(max(item.load for item in deferred)),
                ],
            }
        )

    if len(long_queue) != short_round_size:
        raise ValueError(
            "the accumulated deferred queue must form one complete final round, "
            f"got {len(long_queue)} prompts"
        )
    groups = [*short_rounds, long_queue]
    flattened = [int(item.source_idx) for group in groups for item in group]
    if len(flattened) != expected or set(flattened) != set(source_indices):
        raise ValueError("TailBatch groups do not partition the canonical prompts")
    return groups, pools


def _solve_plans(
    groups: list[list[PromptStats]],
    *,
    variant: str,
    floor_kv_caps: dict[int, float],
    active_peak_safety_factor: float,
    max_response_len: float,
) -> tuple[list[Any], str]:
    if variant == "full16":
        matching = "contiguous"
        plans = [
            _solve_group(
                group,
                max_rank_peak_tokens=floor_kv_caps[16],
                floor_kv_caps={16: floor_kv_caps[16]},
                adaptive_floor=True,
                force_selected_floor=16,
                rank_matching_policy=matching,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
            )
            for group in groups
        ]
    elif variant == "natural_f2":
        matching = "release_area"
        plans = [
            _solve_group(
                group,
                max_rank_peak_tokens=floor_kv_caps[16],
                floor_kv_caps=floor_kv_caps,
                adaptive_floor=True,
                min_adaptive_floor=2,
                rank_matching_policy=matching,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
            )
            for group in groups
        ]
    else:
        raise ValueError(f"unsupported TailBatch variant {variant!r}")
    return plans, matching


def _write_variant(
    output_dir: Path,
    full_df: Any,
    plans: list[Any],
    *,
    history_dir: Path,
    matching: str,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = [output_dir / name for name in PLAN_FILENAMES]
    _write_outputs(
        full_df,
        plans,
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        max_rank_peak_tokens=max_rank_peak_tokens,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
        baseline_dirs=[history_dir],
        length_ema_decay=0.3,
        tail_guard_ratio=999.0,
        tail_guard_ratio_quantile=1.0,
        tail_guard_ratio_window=1,
        tail_guard_sample_count=0,
        tail_guard_min_cap=int(max_response_len),
        tail_guard_round_to=512,
        rank_matching_policy=matching,
        kv_safe_fixed_floor=None,
    )


def _step_membership_from_plan(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    memberships: list[list[int]] = []
    for step in payload:
        indices = [
            int(source_idx)
            for rank in range(16)
            for source_idx in step["rank_to_source_idx"][str(rank)]
        ]
        memberships.append(sorted(indices))
    return memberships


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--active-peak-safety-factor", type=float, default=1.16)
    parser.add_argument("--max-response-len", type=float, default=16384.0)
    parser.add_argument("--floor2-kv-cap", type=float, default=131072.0)
    parser.add_argument("--floor4-kv-cap", type=float, default=280576.0)
    parser.add_argument("--floor8-kv-cap", type=float, default=315648.0)
    parser.add_argument("--floor16-kv-cap", type=float, default=380800.0)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"refusing existing output root: {args.output_root}")
    history_file = args.history_dir / "offline_planning_history.json"
    if not history_file.is_file():
        raise SystemExit(f"missing compact rollout history: {history_file}")
    if not args.train_file.is_file():
        raise SystemExit(f"missing train parquet: {args.train_file}")

    history = _read_baseline_history(
        [args.history_dir],
        steps=5,
        responses_per_prompt=args.responses_per_prompt,
        require_compact_history=True,
    )
    stats_by_input = _read_ema_baseline_stats(history, ema_decay=0.3)
    full_df, stats = _map_stats_to_dataset(
        args.train_file,
        stats_by_input,
        dataset_fraction=args.dataset_fraction,
        max_samples=-1,
        min_samples=160,
    )
    groups, pools = build_tailbatch_groups(stats)
    floor_kv_caps = {
        2: float(args.floor2_kv_cap),
        4: float(args.floor4_kv_cap),
        8: float(args.floor8_kv_cap),
        16: float(args.floor16_kv_cap),
    }

    args.output_root.mkdir(parents=True)
    variant_plans: dict[str, list[Any]] = {}
    for variant in ("full16", "natural_f2"):
        plans, matching = _solve_plans(
            groups,
            variant=variant,
            floor_kv_caps=floor_kv_caps,
            active_peak_safety_factor=args.active_peak_safety_factor,
            max_response_len=args.max_response_len,
        )
        _write_variant(
            args.output_root / variant,
            full_df,
            plans,
            history_dir=args.history_dir,
            matching=matching,
            max_rank_peak_tokens=floor_kv_caps[16],
            active_peak_safety_factor=args.active_peak_safety_factor,
            max_response_len=args.max_response_len,
        )
        variant_plans[variant] = plans

    memberships = {
        variant: _step_membership_from_plan(
            args.output_root / variant / "length_sorted_rank_plan.json"
        )
        for variant in variant_plans
    }
    expected_membership = [
        sorted(int(item.source_idx) for item in group) for group in groups
    ]
    if any(value != expected_membership for value in memberships.values()):
        raise SystemExit("materialized plans changed TailBatch step membership")

    prompt_hashes = {
        int(item.source_idx): hashlib.sha256(
            _prompt_to_input(full_df.iloc[int(item.source_idx)]["prompt"]).encode()
        ).hexdigest()
        for item in stats
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "RollPacker-style causal prompt-level TailBatch replay",
        "scope_note": (
            "This implements only fixed prompt-level short-round and long-round "
            "construction. It is not a complete RollPacker port."
        ),
        "history_dir": str(args.history_dir.resolve()),
        "history_sha256": _sha256_file(history_file),
        "train_file": str(args.train_file.resolve()),
        "train_file_sha256": _sha256_file(args.train_file),
        "responses_per_prompt": int(args.responses_per_prompt),
        "score": "maximum response length over rollout.n historical responses",
        "pool_count": 4,
        "pool_size": 40,
        "short_prompts_per_pool": 32,
        "deferred_prompts_per_pool": 8,
        "steps": 5,
        "prompts_per_step": 32,
        "tail_guard_enabled": False,
        "active_peak_safety_factor": float(args.active_peak_safety_factor),
        "floor_kv_caps": {
            str(floor): cap for floor, cap in floor_kv_caps.items()
        },
        "pools": pools,
        "step_source_indices": expected_membership,
        "step_prompt_hashes": [
            sorted(prompt_hashes[source_idx] for source_idx in step)
            for step in expected_membership
        ],
        "step_prompt_hash_scheme": "sha256_utf8_rendered_input",
        "step_membership_digest": _digest_json(expected_membership),
        "step_prompt_hash_digest": _digest_json(
            [
                sorted(prompt_hashes[source_idx] for source_idx in step)
                for step in expected_membership
            ]
        ),
        "variants": {
            variant: {
                "plan_dir": str((args.output_root / variant).resolve()),
                "rank_matching_policy": (
                    "contiguous" if variant == "full16" else "release_area"
                ),
                "selected_floors": [
                    int(plan.selected_floor) for plan in variant_plans[variant]
                ],
                "feasible": [
                    bool(plan.feasible) for plan in variant_plans[variant]
                ],
                "predicted_release_area": [
                    float(plan.release_area) for plan in variant_plans[variant]
                ],
                "max_adjusted_rank_peak_tokens": [
                    float(max(plan.rank_adjusted_peak_loads.values()))
                    for plan in variant_plans[variant]
                ],
                "file_sha256": {
                    name: _sha256_file(args.output_root / variant / name)
                    for name in PLAN_FILENAMES
                },
            }
            for variant in variant_plans
        },
    }
    manifest_path = args.output_root / "layout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "TAILBATCH_PLAN_COMPLETE.txt").write_text(
        f"step_membership_digest={manifest['step_membership_digest']}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["variants"], indent=2, sort_keys=True))
    print(f"[TailBatch plan] manifest={manifest_path}")


if __name__ == "__main__":
    main()
