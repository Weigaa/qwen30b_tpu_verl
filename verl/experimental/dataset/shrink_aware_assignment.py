from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from vllm_ascend.shrink_aware import (
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
    role_plan = plan_survivor_ranks(
        world_size=world_size,
        shrink_stages=_cfg_get(shrink_cfg, "shrink_stages", [8, 4]),
        package_topology=parse_rank_topology(
            _cfg_get(shrink_cfg, "package_topology", None),
            world_size=world_size),
        policy=str(_cfg_get(
            shrink_cfg, "survivor_selection_policy", "topology_aware")),
        intermediate_survivor_ranks=parse_rank_list(
            _cfg_get(shrink_cfg, "intermediate_survivor_ranks", None)),
        final_survivor_ranks=parse_rank_list(
            _cfg_get(shrink_cfg, "final_survivor_ranks", None)),
    )
    predicted = predict_lengths(
        gen_batch,
        shrink_cfg,
        sampler=sampler,
        default_length=float(_cfg_get(
            shrink_cfg, "default_length",
            _cfg_get(rollout_config, "response_length", 1))),
    )
    assignment_plan = assign_prompts_to_ranks(predicted, role_plan)
    full_rank_order = (
        role_plan.donor_ranks +
        role_plan.wave2_ranks +
        role_plan.final_survivor_ranks)
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
    gen_batch.meta_info["shrink_aware_runtime"] = {
        "mode": mode,
        "shrink_stages": [
            len(role_plan.intermediate_survivor_ranks),
            len(role_plan.final_survivor_ranks),
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

    if _cfg_bool(shrink_cfg, "enable_shrink_aware_logging", False):
        logger.info(
            "Shrink-aware schedule: mode=%s dry_run=%s world_size=%s donor=%s wave2=%s final=%s "
            "counts=%s predicted_load=%s fallback=%s",
            mode,
            dry_run,
            world_size,
            role_plan.donor_ranks,
            role_plan.wave2_ranks,
            role_plan.final_survivor_ranks,
            assignment_plan.per_rank_counts,
            assignment_plan.per_rank_predicted_load,
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
