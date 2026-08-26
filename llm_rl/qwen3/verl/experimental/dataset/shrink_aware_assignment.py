from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from vllm_ascend.shrink_aware import (
    PromptAssignment,
    PromptAssignmentPlan,
    assign_prompts_to_ranks,
    build_reorder_indices,
    parse_rank_list,
    parse_rank_topology,
    plan_survivor_ranks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShrinkAwareScheduleResult:
    enabled: bool
    reorder_indices: list[int]
    restore_indices: list[int]
    assignment_plan: Optional[PromptAssignmentPlan]
    role_plan: Any
    fallback_reason: Optional[str] = None


def maybe_apply_shrink_aware_schedule(
    gen_batch: DataProto,
    rollout_config: Any,
    sampler: Any = None,
    *,
    world_size: Optional[int] = None,
) -> ShrinkAwareScheduleResult:
    shrink_cfg = _get_shrink_aware_config(rollout_config)
    if not _cfg_bool(shrink_cfg, "enable_shrink_aware_scheduling", False):
        return ShrinkAwareScheduleResult(False, [], [], None, None, "disabled")
    mode = str(_cfg_get(shrink_cfg, "shrink_aware_mode", "off")).lower()
    if mode == "off":
        return ShrinkAwareScheduleResult(False, [], [], None, None, "mode_off")

    batch_size = len(gen_batch)
    if batch_size <= 0:
        return ShrinkAwareScheduleResult(False, [], [], None, None, "empty_batch")

    world_size = int(world_size or _infer_world_size(rollout_config))
    plan_entry = _maybe_load_current_optimized_plan_entry(gen_batch, shrink_cfg)
    worker_plans = (
        plan_entry.get("worker_plans")
        if isinstance(plan_entry, dict) else None
    )
    if isinstance(worker_plans, list) and worker_plans:
        planned_global_ranks = sorted({
            int(rank)
            for worker_plan in worker_plans
            if isinstance(worker_plan, dict)
            for rank in worker_plan.get("global_ranks", [])
        })
        if planned_global_ranks != list(range(len(planned_global_ranks))):
            raise ValueError(
                "hierarchical optimized plan must cover contiguous global "
                f"ranks starting at zero, got {planned_global_ranks}")
        world_size = len(planned_global_ranks)
    shrink_stages = _cfg_get(shrink_cfg, "shrink_stages", [8, 4])
    stage_survivor_ranks = None
    intermediate_ranks = parse_rank_list(
        _cfg_get(shrink_cfg, "intermediate_survivor_ranks", None))
    final_ranks = parse_rank_list(
        _cfg_get(shrink_cfg, "final_survivor_ranks", None))
    if isinstance(worker_plans, list) and worker_plans:
        # The global role plan is used only to reorder the 32-prompt batch into
        # rollout-rank order. Each rollout process selects its EP-local role
        # plan from worker_plans before entering vLLM.
        shrink_stages = [world_size]
        stage_survivor_ranks = [list(range(world_size))]
        intermediate_ranks = list(range(world_size))
        final_ranks = list(range(world_size))
    elif plan_entry is not None:
        entry_stages = plan_entry.get("shrink_stages")
        if isinstance(entry_stages, list) and len(entry_stages) >= 1:
            shrink_stages = [int(stage) for stage in entry_stages]
        entry_stage_ranks = plan_entry.get("stage_survivor_ranks")
        if isinstance(entry_stage_ranks, list):
            stage_survivor_ranks = [
                [int(rank) for rank in ranks]
                for ranks in entry_stage_ranks
                if isinstance(ranks, list)
            ]
        entry_intermediate = plan_entry.get("intermediate_survivor_ranks")
        if isinstance(entry_intermediate, list):
            intermediate_ranks = [int(rank) for rank in entry_intermediate]
        entry_final = plan_entry.get("final_survivor_ranks")
        if isinstance(entry_final, list):
            final_ranks = [int(rank) for rank in entry_final]

    role_plan = plan_survivor_ranks(
        world_size=world_size,
        shrink_stages=shrink_stages,
        package_topology=parse_rank_topology(
            _cfg_get(shrink_cfg, "package_topology", None),
            world_size=world_size),
        policy=str(_cfg_get(
            shrink_cfg, "survivor_selection_policy", "topology_aware")),
        intermediate_survivor_ranks=intermediate_ranks,
        final_survivor_ranks=final_ranks,
        stage_survivor_ranks=stage_survivor_ranks,
    )
    predicted = predict_lengths(
        gen_batch,
        shrink_cfg,
        sampler=sampler,
        default_length=float(_cfg_get(
            shrink_cfg, "default_length",
            _cfg_get(rollout_config, "response_length", 1))),
    )
    assignment_plan = _maybe_assign_optimized_rank_plan(
        gen_batch, predicted, role_plan, shrink_cfg, plan_entry=plan_entry)
    if assignment_plan is None:
        assignment_plan = _maybe_assign_manual_5_2_1(
            gen_batch, predicted, role_plan, shrink_cfg)
    if assignment_plan is None:
        assignment_plan = assign_prompts_to_ranks(predicted, role_plan)
    full_rank_order = _rank_order_from_role_plan(role_plan)
    reorder_indices, restore_indices = build_reorder_indices(
        assignment_plan.assignments, full_rank_order)

    dry_run = _cfg_bool(shrink_cfg, "dry_run_shrink_aware_schedule", False)
    if not dry_run and reorder_indices != list(range(batch_size)):
        gen_batch.reorder(torch.tensor(reorder_indices, dtype=torch.long))

    gen_batch.meta_info["shrink_aware_role_plan"] = {
        "donor_ranks": role_plan.donor_ranks,
        "wave2_ranks": role_plan.wave2_ranks,
        "intermediate_survivor_ranks": role_plan.intermediate_survivor_ranks,
        "final_survivor_ranks": role_plan.final_survivor_ranks,
        "stage_survivor_ranks": role_plan.stage_survivor_ranks,
        "package_locality_score": role_plan.package_locality_score,
        "fallback_reason": role_plan.fallback_reason,
    }
    gen_batch.meta_info["shrink_aware_rank_assignment"] = [
        assignment.rank for assignment in assignment_plan.assignments
    ]
    gen_batch.meta_info["shrink_aware_rank_assignment_reordered"] = [
        assignment_plan.assignments[idx].rank for idx in reorder_indices
    ]
    gen_batch.meta_info["shrink_aware_predicted_load"] = list(map(float, predicted))
    gen_batch.meta_info["shrink_aware_dry_run"] = dry_run
    kv_plan: dict[str, Any] = {}
    if isinstance(plan_entry, dict):
        for key in (
                "selected_floor",
                "theoretical_floor",
                "kv_cap",
                "max_adjusted_rank_peak_tokens",
                "max_rank_peak_tokens",
                "tail_guard_response_cap",
                "tail_guard_enabled",
                "tail_guard_ratio",
                "tail_guard_ratio_quantile",
                "tail_guard_ratio_sample_count",
                "tail_guard_predicted_step_exit",
                "tail_guard_raw_cap",
                "tail_guard_min_cap",
                "tail_guard_round_to",
        ):
            if key in plan_entry:
                kv_plan[key] = plan_entry[key]
    if kv_plan:
        gen_batch.meta_info["shrink_aware_kv_plan"] = kv_plan
    gen_batch.meta_info["shrink_aware_runtime"] = {
        "mode": mode,
        "target_policy": str(_cfg_get(
            shrink_cfg, "target_policy", "natural")).lower(),
        "shrink_stages": [
            len(stage_ranks) for stage_ranks in role_plan.stage_survivor_ranks
        ],
        "survivor_selection_policy": "manual",
        "max_rollout_overhead_ratio": float(_cfg_get(
            shrink_cfg, "max_rollout_overhead_ratio", 1.10)),
        "min_shrink_window_seconds": float(_cfg_get(
            shrink_cfg, "min_shrink_window_seconds", 1.0)),
        "enable_shrink_aware_logging": _cfg_bool(
            shrink_cfg, "enable_shrink_aware_logging", False),
        "dry_run_shrink_aware_schedule": dry_run,
    }
    if kv_plan:
        runtime = gen_batch.meta_info["shrink_aware_runtime"]
        if "selected_floor" in kv_plan:
            runtime["selected_floor"] = int(kv_plan["selected_floor"])
        if "theoretical_floor" in kv_plan:
            runtime["theoretical_floor"] = int(kv_plan["theoretical_floor"])
        if "kv_cap" in kv_plan:
            runtime["kv_cap"] = int(float(kv_plan["kv_cap"]))
        if "tail_guard_response_cap" in kv_plan:
            runtime["tail_guard_response_cap"] = int(float(
                kv_plan["tail_guard_response_cap"]))
        if "tail_guard_enabled" in kv_plan:
            runtime["tail_guard_enabled"] = bool(kv_plan["tail_guard_enabled"])
    if isinstance(worker_plans, list) and worker_plans:
        gen_batch.meta_info["shrink_aware_worker_plans"] = deepcopy(
            worker_plans)

    if _cfg_bool(shrink_cfg, "enable_shrink_aware_logging", False):
        logger.info(
            "Shrink-aware schedule: mode=%s dry_run=%s world_size=%s donor=%s wave2=%s final=%s "
            "counts=%s predicted_load=%s assignment_policy=%s fallback=%s",
            mode,
            dry_run,
            world_size,
            role_plan.donor_ranks,
            role_plan.wave2_ranks,
            role_plan.final_survivor_ranks,
            assignment_plan.per_rank_counts,
            assignment_plan.per_rank_predicted_load,
            _cfg_get(shrink_cfg, "assignment_policy", "default"),
            role_plan.fallback_reason,
        )

    return ShrinkAwareScheduleResult(
        enabled=not dry_run,
        reorder_indices=[] if dry_run else reorder_indices,
        restore_indices=[] if dry_run else restore_indices,
        assignment_plan=assignment_plan,
        role_plan=role_plan,
        fallback_reason="dry_run" if dry_run else role_plan.fallback_reason,
    )


def select_shrink_aware_worker_plan(
    meta_info: dict[str, Any],
    global_rank: int,
) -> dict[str, Any]:
    """Overlay the EP-local plan selected for one external-DP worker."""
    if not isinstance(meta_info, dict):
        return meta_info
    worker_plans = meta_info.get("shrink_aware_worker_plans")
    if not isinstance(worker_plans, list) or not worker_plans:
        return meta_info

    selected = None
    for worker_plan in worker_plans:
        if not isinstance(worker_plan, dict):
            continue
        global_ranks = [int(rank) for rank in worker_plan.get("global_ranks", [])]
        if int(global_rank) in global_ranks:
            selected = worker_plan
            break
    if selected is None:
        raise ValueError(
            f"no hierarchical shrink-aware worker plan contains global rank "
            f"{global_rank}")

    global_ranks = [int(rank) for rank in selected["global_ranks"]]
    local_rank = global_ranks.index(int(global_rank))
    role_plan = selected.get("role_plan")
    if not isinstance(role_plan, dict):
        raise ValueError(
            f"worker plan for global rank {global_rank} has no role_plan")

    runtime = dict(meta_info.get("shrink_aware_runtime", {}))
    kv_plan = dict(meta_info.get("shrink_aware_kv_plan", {}))
    for key in (
        "selected_floor",
        "theoretical_floor",
        "kv_cap",
        "tail_guard_response_cap",
        "tail_guard_enabled",
        "tail_guard_ratio",
        "tail_guard_predicted_step_exit",
    ):
        if key in selected:
            kv_plan[key] = selected[key]
    runtime.update({
        "selected_floor": int(selected["selected_floor"]),
        "theoretical_floor": int(selected.get(
            "theoretical_floor", selected["selected_floor"])),
        "kv_cap": int(float(selected["kv_cap"])),
        "shrink_stages": [
            int(stage) for stage in selected.get("shrink_stages", [])
        ],
    })
    if "tail_guard_response_cap" in selected:
        runtime["tail_guard_response_cap"] = int(float(
            selected["tail_guard_response_cap"]))
    if "tail_guard_enabled" in selected:
        runtime["tail_guard_enabled"] = bool(selected["tail_guard_enabled"])

    meta_info["shrink_aware_role_plan"] = deepcopy(role_plan)
    meta_info["shrink_aware_runtime"] = runtime
    meta_info["shrink_aware_kv_plan"] = kv_plan
    meta_info["shrink_aware_worker_context"] = {
        "worker_id": int(selected.get("worker_id", -1)),
        "global_ranks": global_ranks,
        "global_rank": int(global_rank),
        "local_rank": int(local_rank),
    }
    return meta_info


def _maybe_assign_manual_5_2_1(
    gen_batch: DataProto,
    predicted: list[float],
    role_plan: Any,
    shrink_cfg: Any,
) -> Optional[PromptAssignmentPlan]:
    policy = str(_cfg_get(shrink_cfg, "assignment_policy", "default")).lower()
    if policy not in ("manual_5_2_1", "fixed_5_2_1", "521"):
        return None

    if len(role_plan.donor_ranks) != 8 or len(role_plan.wave2_ranks) != 4 \
            or len(role_plan.final_survivor_ranks) != 4:
        raise ValueError(
            "manual_5_2_1 assignment requires 8 donor ranks, 4 wave2 ranks, "
            f"and 4 final ranks; got donor={role_plan.donor_ranks}, "
            f"wave2={role_plan.wave2_ranks}, final={role_plan.final_survivor_ranks}")

    sample_ids = _extract_sample_ids(gen_batch)
    if sample_ids is None or len(sample_ids) != len(predicted):
        raise ValueError(
            "manual_5_2_1 assignment requires dataset_item_idx/index for every "
            "repeated rollout row")

    grouped_positions = _group_repeated_prompts(sample_ids, predicted)
    if len(grouped_positions) != 32:
        raise ValueError(
            "manual_5_2_1 assignment requires 32 prompts per train batch; "
            f"got {len(grouped_positions)} unique prompts")
    prompt_groups: list[dict[str, Any]] = []
    for sample_id, positions in grouped_positions.items():
        prompt_groups.append({
            "sample_id": int(sample_id),
            "positions": positions,
            "first_pos": int(min(positions)),
            "length_sum": sum(float(predicted[pos]) for pos in positions),
        })

    for group in prompt_groups:
        group["mean_length"] = (
            group["length_sum"] / max(1, len(group["positions"])))

    sorted_groups = sorted(
        prompt_groups,
        key=lambda group: (float(group["mean_length"]), int(group["first_pos"])))
    short_bucket = sorted_groups[:20]
    medium_bucket = sorted_groups[20:28]
    long_bucket = sorted_groups[28:32]

    donor_ranks = list(map(int, role_plan.donor_ranks))
    wave2_ranks = list(map(int, role_plan.wave2_ranks))
    final_ranks = list(map(int, role_plan.final_survivor_ranks))

    prompt_rank: dict[int, int] = {}
    for group, rank in zip(short_bucket[:16],
                           [rank for rank in donor_ranks for _ in range(2)],
                           strict=True):
        prompt_rank[int(group["sample_id"])] = rank
    for group, rank in zip(short_bucket[16:20], final_ranks, strict=True):
        prompt_rank[int(group["sample_id"])] = rank
    for group, rank in zip(medium_bucket,
                           [rank for rank in wave2_ranks for _ in range(2)],
                           strict=True):
        prompt_rank[int(group["sample_id"])] = rank
    for group, rank in zip(long_bucket, final_ranks, strict=True):
        prompt_rank[int(group["sample_id"])] = rank

    role_by_rank = {
        **{int(rank): "donor" for rank in donor_ranks},
        **{int(rank): "wave2" for rank in wave2_ranks},
        **{int(rank): "survivor" for rank in final_ranks},
    }
    assignments: list[PromptAssignment] = []
    per_rank_counts = {rank: 0 for rank in role_by_rank}
    per_rank_load = {rank: 0.0 for rank in role_by_rank}
    for prompt_group in prompt_groups:
        rank = prompt_rank[int(prompt_group["sample_id"])]
        role = role_by_rank[rank]
        for pos in prompt_group["positions"]:
            load = float(predicted[pos])
            assignments.append(PromptAssignment(
                prompt_index=int(pos),
                rank=int(rank),
                role=role,
                predicted_load=load,
            ))
            per_rank_counts[rank] += 1
            per_rank_load[rank] += load

    assignments.sort(key=lambda item: item.prompt_index)
    return PromptAssignmentPlan(
        assignments=assignments,
        per_rank_counts=per_rank_counts,
        per_rank_predicted_load=per_rank_load,
        role_by_rank=role_by_rank,
    )


def _maybe_assign_optimized_rank_plan(
    gen_batch: DataProto,
    predicted: list[float],
    role_plan: Any,
    shrink_cfg: Any,
    *,
    plan_entry: Optional[dict[str, Any]] = None,
) -> Optional[PromptAssignmentPlan]:
    policy = str(_cfg_get(shrink_cfg, "assignment_policy", "default")).lower()
    if policy not in ("optimized_rank_plan", "rank_plan", "optimized"):
        return None

    plan_path = _cfg_get(shrink_cfg, "optimized_rank_plan_path", None)
    if not plan_path:
        raise ValueError(
            "optimized_rank_plan assignment requires "
            "shrink_aware.optimized_rank_plan_path")

    sample_ids = _extract_sample_ids(gen_batch)
    if sample_ids is None or len(sample_ids) != len(predicted):
        raise ValueError(
            "optimized_rank_plan assignment requires dataset_item_idx/index "
            "for every repeated rollout row")

    prompt_groups = _group_repeated_prompts(sample_ids, predicted)
    if plan_entry is None:
        plan_entry = _load_matching_rank_plan(plan_path, set(prompt_groups))
    raw_rank_map = plan_entry.get("rank_to_dataset_item_idx")
    if not isinstance(raw_rank_map, dict):
        raise ValueError(
            f"optimized rank plan entry missing rank_to_dataset_item_idx: "
            f"{plan_path}")

    prompt_rank: dict[int, int] = {}
    for rank_key, ids in raw_rank_map.items():
        rank = int(rank_key)
        if not isinstance(ids, list):
            raise ValueError(
                f"rank_to_dataset_item_idx[{rank_key!r}] must be a list")
        for sample_id in ids:
            sample_id = int(sample_id)
            if sample_id in prompt_rank:
                raise ValueError(
                    f"duplicate dataset_item_idx={sample_id} in optimized plan")
            prompt_rank[sample_id] = rank

    missing = sorted(set(prompt_groups) - set(prompt_rank))
    extra = sorted(set(prompt_rank) - set(prompt_groups))
    if missing or extra:
        raise ValueError(
            "optimized rank plan does not match current batch: "
            f"missing={missing[:8]} extra={extra[:8]} path={plan_path}")

    role_by_rank = _role_by_rank_from_role_plan(role_plan)
    assignments: list[PromptAssignment] = []
    per_rank_counts = {rank: 0 for rank in role_by_rank}
    per_rank_load = {rank: 0.0 for rank in role_by_rank}
    for sample_id, positions in prompt_groups.items():
        rank = int(prompt_rank[int(sample_id)])
        if rank not in role_by_rank:
            raise ValueError(
                f"optimized rank plan assigned dataset_item_idx={sample_id} "
                f"to rank={rank}, which is not in the shrink-aware role plan")
        role = role_by_rank[rank]
        for pos in positions:
            load = float(predicted[int(pos)])
            assignments.append(PromptAssignment(
                prompt_index=int(pos),
                rank=rank,
                role=role,
                predicted_load=load,
            ))
            per_rank_counts[rank] += 1
            per_rank_load[rank] += load

    assignments.sort(key=lambda item: item.prompt_index)
    return PromptAssignmentPlan(
        assignments=assignments,
        per_rank_counts=per_rank_counts,
        per_rank_predicted_load=per_rank_load,
        role_by_rank=role_by_rank,
    )


def _maybe_load_current_optimized_plan_entry(
    gen_batch: DataProto,
    shrink_cfg: Any,
) -> Optional[dict[str, Any]]:
    policy = str(_cfg_get(shrink_cfg, "assignment_policy", "default")).lower()
    if policy not in ("optimized_rank_plan", "rank_plan", "optimized"):
        return None
    plan_path = _cfg_get(shrink_cfg, "optimized_rank_plan_path", None)
    if not plan_path:
        return None
    sample_ids = _extract_sample_ids(gen_batch)
    if sample_ids is None:
        return None
    sample_id_set = set(int(item) for item in sample_ids)
    try:
        return _load_matching_rank_plan(plan_path, sample_id_set)
    except ValueError:
        return None


def _rank_order_from_role_plan(role_plan: Any) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for rank in role_plan.donor_ranks:
        rank = int(rank)
        if rank not in seen:
            ordered.append(rank)
            seen.add(rank)
    for ranks in getattr(role_plan, "stage_survivor_ranks", None) or [
        role_plan.intermediate_survivor_ranks,
        role_plan.final_survivor_ranks,
    ]:
        for rank in ranks:
            rank = int(rank)
            if rank not in seen:
                ordered.append(rank)
                seen.add(rank)
    return ordered


def _role_by_rank_from_role_plan(role_plan: Any) -> dict[int, str]:
    role_by_rank = {int(rank): "donor" for rank in role_plan.donor_ranks}
    stage_sets = getattr(role_plan, "stage_survivor_ranks", None) or [
        role_plan.intermediate_survivor_ranks,
        role_plan.final_survivor_ranks,
    ]
    for stage_idx, ranks in enumerate(stage_sets):
        current_stage = {int(rank) for rank in ranks}
        next_stage = (
            {int(rank) for rank in stage_sets[stage_idx + 1]}
            if stage_idx + 1 < len(stage_sets) else set()
        )
        role_ranks = sorted(current_stage - next_stage) if next_stage else sorted(current_stage)
        if stage_idx == len(stage_sets) - 1:
            role = "survivor"
        elif len(stage_sets) == 2 and stage_idx == 0:
            role = "wave2"
        else:
            role = f"stage{stage_idx + 1}"
        for rank in role_ranks:
            role_by_rank[int(rank)] = role
    for rank in role_plan.final_survivor_ranks:
        role_by_rank[int(rank)] = "survivor"
    return role_by_rank


def _group_repeated_prompts(sample_ids: np.ndarray,
                            predicted: list[float]) -> dict[int, list[int]]:
    if len(sample_ids) != len(predicted):
        raise ValueError(
            f"sample_ids/predicted length mismatch: {len(sample_ids)} vs {len(predicted)}")
    prompt_groups: dict[int, list[int]] = {}
    for pos, sample_id in enumerate(sample_ids):
        prompt_groups.setdefault(int(sample_id), []).append(int(pos))
    if not prompt_groups:
        raise ValueError(
            "optimized/manual shrink-aware assignment requires at least one prompt")
    copies_per_prompt = {len(positions) for positions in prompt_groups.values()}
    if len(copies_per_prompt) != 1:
        raise ValueError(
            "optimized/manual shrink-aware assignment requires a uniform number "
            "of rollout responses per prompt; "
            f"got copies_per_prompt={sorted(copies_per_prompt)}")
    return prompt_groups


_OPTIMIZED_RANK_PLAN_CACHE: dict[str, list[dict[str, Any]]] = {}


def _load_matching_rank_plan(path: str,
                             sample_ids: set[int]) -> dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(str(path)))
    payload = _OPTIMIZED_RANK_PLAN_CACHE.get(path)
    if payload is None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"optimized rank plan must be a list: {path}")
        _OPTIMIZED_RANK_PLAN_CACHE[path] = payload
    for entry in payload:
        raw_rank_map = entry.get("rank_to_dataset_item_idx")
        if not isinstance(raw_rank_map, dict):
            continue
        planned_ids: set[int] = set()
        for ids in raw_rank_map.values():
            if isinstance(ids, list):
                planned_ids.update(int(item) for item in ids)
        if planned_ids == sample_ids:
            return entry
    raise ValueError(
        f"no optimized rank plan entry matches current batch ids "
        f"{sorted(sample_ids)[:8]}... in {path}")


def restore_shrink_aware_order(data: DataProto,
                               result: ShrinkAwareScheduleResult) -> None:
    if not result.enabled or not result.restore_indices:
        return
    if result.restore_indices == list(range(len(result.restore_indices))):
        return
    data.reorder(torch.tensor(result.restore_indices, dtype=torch.long))


def predict_lengths(
    gen_batch: DataProto,
    shrink_cfg: Any,
    *,
    sampler: Any = None,
    default_length: float = 1.0,
) -> list[float]:
    source = str(_cfg_get(
        shrink_cfg, "length_prediction_source", "existing_regroup")).lower()
    batch_size = len(gen_batch)
    if source in ("existing_regroup", "history"):
        values = _lengths_from_sampler(gen_batch, sampler)
        if values is not None:
            return values
        if source == "history":
            return [float(default_length)] * batch_size
        source = "prompt_length"
    if source == "prompt_length":
        return _prompt_lengths(gen_batch)
    if source == "oracle_trace":
        values = _lengths_from_oracle(gen_batch, _cfg_get(
            shrink_cfg, "oracle_trace_path", None))
        if values is not None:
            return values
        return [float(default_length)] * batch_size
    return [float(default_length)] * batch_size


def _lengths_from_sampler(gen_batch: DataProto,
                          sampler: Any) -> Optional[list[float]]:
    length_est = getattr(sampler, "_length_estimate", None)
    if length_est is None:
        return None
    sample_ids = None
    for key in ("dataset_item_idx", "index"):
        if key in gen_batch.non_tensor_batch:
            sample_ids = np.asarray(gen_batch.non_tensor_batch[key], dtype=np.int64)
            break
    if sample_ids is None or sample_ids.size == 0:
        return None
    result: list[float] = []
    for sample_id in sample_ids:
        if 0 <= int(sample_id) < len(length_est):
            result.append(float(length_est[int(sample_id)]))
        else:
            result.append(float(np.mean(length_est)))
    return result


def _prompt_lengths(gen_batch: DataProto) -> list[float]:
    if gen_batch.batch is not None and "attention_mask" in gen_batch.batch:
        mask = gen_batch.batch["attention_mask"]
        if isinstance(mask, torch.Tensor):
            return [float(x) for x in mask.sum(dim=-1).detach().cpu().tolist()]
    raw_prompt_ids = gen_batch.non_tensor_batch.get("raw_prompt_ids")
    if raw_prompt_ids is not None:
        return [float(len(item)) for item in raw_prompt_ids]
    if gen_batch.batch is not None and "input_ids" in gen_batch.batch:
        return [float(gen_batch.batch["input_ids"].shape[-1])] * len(gen_batch)
    return [1.0] * len(gen_batch)


def _extract_sample_ids(batch: DataProto) -> Optional[np.ndarray]:
    for key in ("dataset_item_idx", "index"):
        if key in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch[key], dtype=np.int64)
    return None


def _lengths_from_oracle(gen_batch: DataProto,
                         path: Optional[str]) -> Optional[list[float]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError:
        logger.warning("Failed to read shrink-aware oracle trace: %s", path)
        return None

    values: list[float] = []
    for key in ("request_id", "uid", "dataset_item_idx", "index"):
        if key not in gen_batch.non_tensor_batch:
            continue
        for item in gen_batch.non_tensor_batch[key]:
            lookup = str(item)
            if lookup in payload:
                values.append(float(payload[lookup]))
            else:
                values.append(float(np.mean(list(payload.values()))))
        return values
    if isinstance(payload, list):
        return [float(item) for item in payload[:len(gen_batch)]]
    return None


def _get_shrink_aware_config(rollout_config: Any) -> Any:
    cfg = _cfg_get(rollout_config, "shrink_aware", None)
    if cfg is None:
        return {}
    return cfg


def _infer_world_size(rollout_config: Any) -> int:
    env_world_size = os.getenv("VLLM_DP_SIZE", "").strip()
    if env_world_size:
        try:
            return max(1, int(env_world_size))
        except ValueError:
            logger.warning("Ignoring invalid VLLM_DP_SIZE=%s", env_world_size)
    dp = int(_cfg_get(rollout_config, "data_parallel_size", 1) or 1)
    tp = int(_cfg_get(rollout_config, "tensor_model_parallel_size", 1) or 1)
    ep = int(_cfg_get(rollout_config, "expert_parallel_size", 1) or 1)
    return max(1, dp * max(1, ep // max(tp, 1)))


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)
