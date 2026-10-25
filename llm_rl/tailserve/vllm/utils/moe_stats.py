# vllm/utils/moe_stats.py
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _stable_hash_to_signed_int64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    out = int(digest, 16)
    if out >= (1 << 63):
        out -= (1 << 64)
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dense_rank_from_counts(counts: list[int]) -> dict[int, int]:
    order = sorted(range(len(counts)), key=lambda idx: (-int(counts[idx]), idx))
    return {expert_id: rank + 1 for rank, expert_id in enumerate(order)}


def _cosine_similarity_from_counts(lhs: list[int], rhs: list[int]) -> float:
    if len(lhs) != len(rhs) or not lhs:
        return 0.0
    dot = float(sum(int(a) * int(b) for a, b in zip(lhs, rhs)))
    lhs_norm = float(sum(int(a) * int(a) for a in lhs)) ** 0.5
    rhs_norm = float(sum(int(b) * int(b) for b in rhs)) ** 0.5
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _jaccard_selected_from_counts(lhs: list[int], rhs: list[int]) -> float:
    lhs_selected = {idx for idx, count in enumerate(lhs) if int(count) > 0}
    rhs_selected = {idx for idx, count in enumerate(rhs) if int(count) > 0}
    if not lhs_selected and not rhs_selected:
        return 1.0
    union = lhs_selected | rhs_selected
    if not union:
        return 0.0
    return float(len(lhs_selected & rhs_selected)) / float(len(union))


def _top_expert_id_from_counts(counts: list[int]) -> int:
    if not counts or sum(int(x) for x in counts) <= 0:
        return -1
    return max(range(len(counts)), key=lambda idx: (int(counts[idx]), -idx))


def _selected_expert_sets_from_counts(lhs: list[int],
                                      rhs: list[int]) -> tuple[set[int], set[int]]:
    lhs_selected = {idx for idx, count in enumerate(lhs) if int(count) > 0}
    rhs_selected = {idx for idx, count in enumerate(rhs) if int(count) > 0}
    return lhs_selected, rhs_selected


def _prompt_token_export_mode() -> str:
    return os.getenv("VLLM_MOE_PROMPT_TOKEN_EXPORT_MODE",
                     "compact").strip().lower()


def _sanitize_filename_fragment(value: str) -> str:
    sanitized = [
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(value)
    ]
    collapsed = "".join(sanitized).strip("_")
    return collapsed or "item"


def _new_timing_bucket() -> dict[str, float]:
    return {
        "total_s": 0.0,
        "calls": 0.0,
        "rows": 0.0,
        "max_s": 0.0,
    }


def _accumulate_timing(stats: dict[str, dict[str, float]],
                       name: str,
                       *,
                       total_s: float,
                       calls: int = 1,
                       rows: int = 0,
                       max_s: Optional[float] = None) -> None:
    bucket = stats.setdefault(name, _new_timing_bucket())
    bucket["total_s"] += float(total_s)
    bucket["calls"] += float(calls)
    bucket["rows"] += float(rows)
    bucket["max_s"] = max(float(bucket.get("max_s", 0.0)),
                          float(total_s if max_s is None else max_s))


def _timing_stats_to_rows(
        stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, bucket in sorted(stats.items(),
                               key=lambda item: (-float(item[1].get("total_s", 0.0)),
                                                 item[0])):
        total_s = float(bucket.get("total_s", 0.0))
        calls = int(bucket.get("calls", 0))
        rows_seen = int(bucket.get("rows", 0))
        max_s = float(bucket.get("max_s", 0.0))
        rows.append({
            "name": name,
            "total_s": total_s,
            "total_ms": total_s * 1000.0,
            "calls": calls,
            "avg_ms_per_call": (total_s * 1000.0 / calls) if calls > 0 else 0.0,
            "max_ms": max_s * 1000.0,
            "rows": rows_seen,
            "avg_us_per_row": (total_s * 1_000_000.0 / rows_seen)
            if rows_seen > 0 else 0.0,
        })
    return rows


class MoEStats:

    def __init__(self):
        self.pattern_stats_env = "VLLM_MOE_PATTERN_STATS"
        legacy_enabled = os.getenv("VLLM_MOE_STATS", "1")
        self.enabled = os.getenv(self.pattern_stats_env, legacy_enabled) == "1"
        self.decode_only = os.getenv("VLLM_MOE_STATS_DECODE_ONLY", "1") == "1"
        self.timing_enabled = (
            self.enabled
            and os.getenv("VLLM_MOE_STATS_TIMING", "1") == "1"
        )
        self.output_dir = os.getenv("VLLM_MOE_STATS_DIR", "./moe_stats")
        self.long_tail_stages = tuple(
            int(x) for x in os.getenv("VLLM_MOE_LONG_TAIL_STAGES", "8,4,2").split(",")
            if x.strip()
        )
        self._lock = threading.Lock()
        self.current_batch_seq_ids: Optional[torch.Tensor] = None
        self.current_batch_seq_hashes: list[str] = []
        self.current_batch_prompt_hashes: list[str] = []
        self.current_batch_token_positions: list[int] = []
        self.current_epoch: int = -1
        self.current_global_step: int = -1
        self.current_active_rank_count: int = -1
        self.schedule_count = 0
        self.top_k_count = 0
        self.step_layer_topk = None
        self.num_experts: Optional[int] = None
        self.total_expert_counts: Optional[torch.Tensor] = None
        logger.info(
            "MoE pattern stats %s: %s=%s dir=%s long_tail_stages=%s",
            "enabled" if self.enabled else "disabled",
            self.pattern_stats_env,
            "1" if self.enabled else "0",
            self.output_dir,
            self.long_tail_stages,
        )
        logger.info(
            "MoE pattern stats timing %s: VLLM_MOE_STATS_TIMING=%s",
            "enabled" if self.timing_enabled else "disabled",
            "1" if self.timing_enabled else "0",
        )
        self.reset_epoch()

    def is_pattern_stats_enabled(self) -> bool:
        return self.enabled

    def _make_zero_vec(self) -> torch.Tensor:
        assert self.num_experts is not None
        return torch.zeros(self.num_experts, dtype=torch.int64)

    def _ensure_num_experts(self, num_experts: int) -> None:
        num_experts = int(num_experts)
        if self.num_experts is None:
            self.num_experts = num_experts
            self.total_expert_counts = torch.zeros(num_experts, dtype=torch.int64)
            return
        if self.num_experts != num_experts:
            raise ValueError(
                f"MoEStats saw inconsistent num_experts: {self.num_experts} vs {num_experts}"
            )

    def set_generation_context(self, *, epoch: Optional[int] = None,
                               global_step: Optional[int] = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if global_step is not None:
            self.current_global_step = int(global_step)

    def set_step_context(self, *, active_rank_count: Optional[int] = None) -> None:
        if active_rank_count is not None:
            self.current_active_rank_count = int(active_rank_count)

    def _record_timing(self,
                       name: str,
                       elapsed_s: float,
                       *,
                       calls: int = 1,
                       rows: int = 0) -> None:
        if not self.timing_enabled:
            return
        _accumulate_timing(
            self.timing_stats,
            name,
            total_s=float(elapsed_s),
            calls=int(calls),
            rows=int(rows),
        )

    def get_current_batch_seq_ids(self, num_scheduled_tokens) -> None:
        start_total = time.perf_counter() if self.timing_enabled else 0.0
        out_ids = []
        seq_hashes: list[str] = []
        prompt_hashes: list[str] = []
        token_positions: list[int] = []
        for rid, payload in num_scheduled_tokens.items():
            seq_hash = str(rid)
            prompt_hash = seq_hash
            start_pos = 0
            explicit_positions: Optional[list[int]] = None
            if isinstance(payload, dict):
                prompt_hash = str(payload.get("prompt_hash", seq_hash))
                repeat = int(payload.get("count", 0))
                start_pos = int(payload.get("start_pos", 0))
                raw_positions = payload.get("token_positions")
                if raw_positions is not None:
                    explicit_positions = [int(x) for x in raw_positions]
            elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
                prompt_hash = str(payload[0])
                repeat = int(payload[1])
                if len(payload) >= 3:
                    start_pos = int(payload[2])
            else:
                repeat = int(payload)
            if repeat <= 0:
                continue
            if explicit_positions is None:
                expanded_positions = list(range(start_pos, start_pos + repeat))
            else:
                expanded_positions = explicit_positions[:repeat]
                if len(expanded_positions) < repeat:
                    next_pos = expanded_positions[-1] + 1 if expanded_positions else start_pos
                    while len(expanded_positions) < repeat:
                        expanded_positions.append(next_pos)
                        next_pos += 1
            signed_id = _stable_hash_to_signed_int64(seq_hash)
            out_ids.extend([signed_id] * repeat)
            seq_hashes.extend([seq_hash] * repeat)
            prompt_hashes.extend([prompt_hash] * repeat)
            token_positions.extend(expanded_positions)
        self.current_batch_seq_ids = torch.tensor(out_ids, dtype=torch.long)
        self.current_batch_seq_hashes = seq_hashes
        self.current_batch_prompt_hashes = prompt_hashes
        self.current_batch_token_positions = token_positions
        self.schedule_count += 1
        if self.timing_enabled:
            self._record_timing(
                "scheduler_expand_batch_context",
                time.perf_counter() - start_total,
                rows=len(out_ids),
            )

    def reset_epoch(self):
        with self._lock:
            self.per_prompt = {}
            self.total_expert_counts = (
                torch.zeros(self.num_experts, dtype=torch.int64)
                if self.num_experts is not None else None
            )
            self.epoch_expert_counts: dict[int, torch.Tensor] = {}
            self.epoch_stage_expert_counts: dict[int, dict[int, torch.Tensor]] = {}
            self.epoch_layer_expert_counts: dict[int, dict[int, torch.Tensor]] = {}
            self.epoch_stage_layer_expert_counts: dict[int, dict[int, dict[int, torch.Tensor]]] = {}
            self.epoch_prompt_expert_counts: dict[int, dict[str, torch.Tensor]] = {}
            self.epoch_prompt_stage_expert_counts: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
            self.epoch_prompt_layer_expert_counts: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
            self.epoch_prompt_stage_layer_expert_counts: dict[
                int, dict[int, dict[str, dict[int, torch.Tensor]]]
            ] = {}
            self.epoch_prompt_token_layer_expert_counts: dict[
                int, dict[str, dict[int, dict[int, torch.Tensor]]]
            ] = {}
            self.epoch_seq_expert_counts: dict[int, dict[str, torch.Tensor]] = {}
            self.epoch_seq_layer_expert_counts: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
            self.epoch_seq_stage_layer_expert_counts: dict[
                int, dict[int, dict[str, dict[int, torch.Tensor]]]
            ] = {}
            self.epoch_seq_parent_prompt: dict[int, dict[str, str]] = defaultdict(dict)
            self.epoch_layer_route_rows: dict[int, dict[int, int]] = defaultdict(dict)
            self.epoch_stage_layer_route_rows: dict[int, dict[int, dict[int, int]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            self.epoch_prompt_route_rows: dict[int, dict[str, int]] = defaultdict(dict)
            self.epoch_prompt_stage_route_rows: dict[int, dict[int, dict[str, int]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            self.epoch_prompt_layer_route_rows: dict[int, dict[str, dict[int, int]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            self.epoch_prompt_stage_layer_route_rows: dict[
                int, dict[int, dict[str, dict[int, int]]]
            ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            self.epoch_prompt_token_layer_route_rows: dict[
                int, dict[str, dict[int, dict[int, int]]]
            ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            self.epoch_seq_route_rows: dict[int, dict[str, int]] = defaultdict(dict)
            self.epoch_seq_layer_route_rows: dict[int, dict[str, dict[int, int]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            self.epoch_seq_stage_layer_route_rows: dict[
                int, dict[int, dict[str, dict[int, int]]]
            ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            self.current_batch_seq_ids = None
            self.current_batch_seq_hashes = []
            self.current_batch_prompt_hashes = []
            self.current_batch_token_positions = []
            self.current_epoch = -1
            self.current_global_step = -1
            self.current_active_rank_count = -1
            self.top_k_count = 0
            self.schedule_count = 0
            self.step_layer_topk = None
            self.timing_stats: dict[str, dict[str, float]] = {}
            print("successfully reset MoEStats for new epoch.")

    def _ensure_vec(self, prompt_id: int, layer_idx: int, num_experts: int):
        self._ensure_num_experts(num_experts)
        d = self.per_prompt.setdefault(int(prompt_id), {})
        if layer_idx not in d:
            d[layer_idx] = self._make_zero_vec().to(torch.float32)

    def _ensure_epoch_prompt_vec(self, epoch: int, prompt_hash: str) -> torch.Tensor:
        d = self.epoch_prompt_expert_counts.setdefault(epoch, {})
        if prompt_hash not in d:
            d[prompt_hash] = self._make_zero_vec()
        return d[prompt_hash]

    def _ensure_epoch_stage_prompt_vec(self, epoch: int, stage: int,
                                       prompt_hash: str) -> torch.Tensor:
        stage_dict = self.epoch_prompt_stage_expert_counts.setdefault(epoch, {}).setdefault(stage, {})
        if prompt_hash not in stage_dict:
            stage_dict[prompt_hash] = self._make_zero_vec()
        return stage_dict[prompt_hash]

    def _ensure_epoch_prompt_layer_vec(self, epoch: int, prompt_hash: str,
                                       layer_idx: int) -> torch.Tensor:
        prompt_dict = self.epoch_prompt_layer_expert_counts.setdefault(epoch, {}).setdefault(
            prompt_hash, {}
        )
        if layer_idx not in prompt_dict:
            prompt_dict[layer_idx] = self._make_zero_vec()
        return prompt_dict[layer_idx]

    def _ensure_epoch_stage_prompt_layer_vec(self, epoch: int, stage: int,
                                              prompt_hash: str,
                                              layer_idx: int) -> torch.Tensor:
        prompt_dict = self.epoch_prompt_stage_layer_expert_counts.setdefault(
            epoch, {}).setdefault(stage, {}).setdefault(prompt_hash, {})
        if layer_idx not in prompt_dict:
            prompt_dict[layer_idx] = self._make_zero_vec()
        return prompt_dict[layer_idx]

    def _ensure_epoch_prompt_token_layer_vec(
            self, epoch: int, prompt_hash: str, token_position: int,
            layer_idx: int) -> torch.Tensor:
        token_dict = self.epoch_prompt_token_layer_expert_counts.setdefault(
            epoch, {}).setdefault(prompt_hash, {}).setdefault(int(token_position),
                                                              {})
        if layer_idx not in token_dict:
            token_dict[layer_idx] = self._make_zero_vec()
        return token_dict[layer_idx]

    def _ensure_epoch_layer_vec(self, epoch: int, layer_idx: int) -> torch.Tensor:
        layer_dict = self.epoch_layer_expert_counts.setdefault(epoch, {})
        if layer_idx not in layer_dict:
            layer_dict[layer_idx] = self._make_zero_vec()
        return layer_dict[layer_idx]

    def _ensure_epoch_stage_layer_vec(self, epoch: int, stage: int,
                                      layer_idx: int) -> torch.Tensor:
        stage_dict = self.epoch_stage_layer_expert_counts.setdefault(epoch, {}).setdefault(stage, {})
        if layer_idx not in stage_dict:
            stage_dict[layer_idx] = self._make_zero_vec()
        return stage_dict[layer_idx]

    def _ensure_epoch_seq_vec(self, epoch: int, seq_hash: str) -> torch.Tensor:
        seq_dict = self.epoch_seq_expert_counts.setdefault(epoch, {})
        if seq_hash not in seq_dict:
            seq_dict[seq_hash] = self._make_zero_vec()
        return seq_dict[seq_hash]

    def _ensure_epoch_seq_layer_vec(self, epoch: int, seq_hash: str,
                                    layer_idx: int) -> torch.Tensor:
        seq_dict = self.epoch_seq_layer_expert_counts.setdefault(epoch, {}).setdefault(
            seq_hash, {}
        )
        if layer_idx not in seq_dict:
            seq_dict[layer_idx] = self._make_zero_vec()
        return seq_dict[layer_idx]

    def _ensure_epoch_stage_seq_layer_vec(self, epoch: int, stage: int,
                                          seq_hash: str,
                                          layer_idx: int) -> torch.Tensor:
        seq_dict = self.epoch_seq_stage_layer_expert_counts.setdefault(
            epoch, {}).setdefault(stage, {}).setdefault(seq_hash, {})
        if layer_idx not in seq_dict:
            seq_dict[layer_idx] = self._make_zero_vec()
        return seq_dict[layer_idx]

    @torch.no_grad()
    def record_layer_topk(self, layer_id: int, topk_ids: torch.Tensor):
        layer_id = int(layer_id)
        topk_ids_cpu = topk_ids.detach().to("cpu")
        if self.step_layer_topk is None:
            self.step_layer_topk = []
        if len(self.step_layer_topk) <= layer_id:
            self.step_layer_topk.extend([None] * (layer_id + 1 - len(self.step_layer_topk)))
        self.step_layer_topk[layer_id] = topk_ids_cpu

    @torch.no_grad()
    def record_pattern(self, layer_idx: int, topk_ids: torch.Tensor,
                       num_experts: int) -> None:
        start_total = time.perf_counter() if self.timing_enabled else 0.0
        if not self.enabled:
            return
        if (not self.current_batch_seq_hashes or not self.current_batch_prompt_hashes
                or not self.current_batch_token_positions):
            return

        self._ensure_num_experts(num_experts)
        assert self.total_expert_counts is not None

        start_phase = time.perf_counter() if self.timing_enabled else 0.0
        topk_cpu = topk_ids.detach().to(dtype=torch.int64, device="cpu")
        if self.timing_enabled:
            self._record_timing("record_pattern.topk_to_cpu",
                                time.perf_counter() - start_phase)
        if topk_cpu.dim() == 1:
            topk_cpu = topk_cpu.unsqueeze(-1)
        if topk_cpu.numel() == 0:
            return

        usable_rows = min(
            len(self.current_batch_seq_hashes),
            len(self.current_batch_prompt_hashes),
            len(self.current_batch_token_positions),
            int(topk_cpu.shape[0]),
        )
        if usable_rows <= 0:
            return
        if usable_rows != int(topk_cpu.shape[0]):
            topk_cpu = topk_cpu[:usable_rows]

        seq_hashes = self.current_batch_seq_hashes[:usable_rows]
        prompt_hashes = self.current_batch_prompt_hashes[:usable_rows]
        token_positions = self.current_batch_token_positions[:usable_rows]
        flat_ids = topk_cpu.reshape(-1)
        total_counts = torch.bincount(flat_ids, minlength=num_experts)
        self.total_expert_counts += total_counts

        epoch = int(self.current_epoch)
        stage = int(self.current_active_rank_count)
        layer_idx = int(layer_idx)

        epoch_counts = self.epoch_expert_counts.setdefault(epoch, self._make_zero_vec())
        epoch_counts += total_counts

        layer_counts = self._ensure_epoch_layer_vec(epoch, layer_idx)
        layer_counts += total_counts
        self.epoch_layer_route_rows[epoch][layer_idx] = (
            int(self.epoch_layer_route_rows[epoch].get(layer_idx, 0))
            + usable_rows
        )

        if stage in self.long_tail_stages:
            stage_counts = self.epoch_stage_expert_counts.setdefault(epoch, {}).setdefault(
                stage, self._make_zero_vec()
            )
            stage_counts += total_counts

        if stage >= 0:
            stage_layer_counts = self._ensure_epoch_stage_layer_vec(
                epoch, stage, layer_idx)
            stage_layer_counts += total_counts
            stage_layer_rows = self.epoch_stage_layer_route_rows[epoch][stage]
            stage_layer_rows[layer_idx] = int(stage_layer_rows.get(layer_idx, 0)) + usable_rows

        start_phase = time.perf_counter() if self.timing_enabled else 0.0
        grouped_prompt_indices: dict[str, list[int]] = defaultdict(list)
        grouped_seq_indices: dict[str, list[int]] = defaultdict(list)
        grouped_prompt_token_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
        for idx, (seq_hash, prompt_hash) in enumerate(zip(seq_hashes,
                                                          prompt_hashes)):
            grouped_prompt_indices[prompt_hash].append(idx)
            grouped_seq_indices[seq_hash].append(idx)
            grouped_prompt_token_indices[(prompt_hash,
                                          int(token_positions[idx]))].append(idx)
        if self.timing_enabled:
            self._record_timing("record_pattern.group_rows",
                                time.perf_counter() - start_phase,
                                rows=usable_rows)

        start_phase = time.perf_counter() if self.timing_enabled else 0.0
        for prompt_hash, indices in grouped_prompt_indices.items():
            row_index = torch.tensor(indices, dtype=torch.long)
            prompt_ids = topk_cpu.index_select(0, row_index).reshape(-1)
            prompt_counts = torch.bincount(prompt_ids, minlength=num_experts)

            prompt_vec = self._ensure_epoch_prompt_vec(epoch, prompt_hash)
            prompt_vec += prompt_counts
            self.epoch_prompt_route_rows[epoch][prompt_hash] = (
                int(self.epoch_prompt_route_rows[epoch].get(prompt_hash, 0))
                + len(indices)
            )

            prompt_layer_vec = self._ensure_epoch_prompt_layer_vec(
                epoch, prompt_hash, layer_idx)
            prompt_layer_vec += prompt_counts
            prompt_layer_rows = self.epoch_prompt_layer_route_rows[epoch][prompt_hash]
            prompt_layer_rows[layer_idx] = int(
                prompt_layer_rows.get(layer_idx, 0)) + len(indices)

            if stage in self.long_tail_stages:
                stage_prompt_vec = self._ensure_epoch_stage_prompt_vec(epoch, stage, prompt_hash)
                stage_prompt_vec += prompt_counts
                stage_route_rows = self.epoch_prompt_stage_route_rows[epoch][stage]
                stage_route_rows[prompt_hash] = int(stage_route_rows.get(prompt_hash, 0)) + len(indices)

            if stage >= 0:
                stage_prompt_layer_vec = self._ensure_epoch_stage_prompt_layer_vec(
                    epoch, stage, prompt_hash, layer_idx)
                stage_prompt_layer_vec += prompt_counts
                stage_prompt_layer_rows = (
                    self.epoch_prompt_stage_layer_route_rows[epoch][stage][prompt_hash]
                )
                stage_prompt_layer_rows[layer_idx] = int(
                    stage_prompt_layer_rows.get(layer_idx, 0)) + len(indices)
        if self.timing_enabled:
            self._record_timing("record_pattern.prompt_aggregate",
                                time.perf_counter() - start_phase,
                                rows=usable_rows)

        start_phase = time.perf_counter() if self.timing_enabled else 0.0
        for (prompt_hash, token_position), indices in grouped_prompt_token_indices.items():
            row_index = torch.tensor(indices, dtype=torch.long)
            token_ids = topk_cpu.index_select(0, row_index).reshape(-1)
            token_counts = torch.bincount(token_ids, minlength=num_experts)
            prompt_token_layer_vec = self._ensure_epoch_prompt_token_layer_vec(
                epoch, prompt_hash, int(token_position), layer_idx)
            prompt_token_layer_vec += token_counts
            prompt_token_rows = self.epoch_prompt_token_layer_route_rows[
                epoch][prompt_hash][int(token_position)]
            prompt_token_rows[layer_idx] = int(
                prompt_token_rows.get(layer_idx, 0)) + len(indices)
        if self.timing_enabled:
            self._record_timing("record_pattern.prompt_token_aggregate",
                                time.perf_counter() - start_phase,
                                rows=usable_rows)

        start_phase = time.perf_counter() if self.timing_enabled else 0.0
        for seq_hash, indices in grouped_seq_indices.items():
            row_index = torch.tensor(indices, dtype=torch.long)
            seq_ids = topk_cpu.index_select(0, row_index).reshape(-1)
            seq_counts = torch.bincount(seq_ids, minlength=num_experts)
            prompt_hash = prompt_hashes[indices[0]]

            self.epoch_seq_parent_prompt[epoch][seq_hash] = prompt_hash
            seq_vec = self._ensure_epoch_seq_vec(epoch, seq_hash)
            seq_vec += seq_counts
            self.epoch_seq_route_rows[epoch][seq_hash] = (
                int(self.epoch_seq_route_rows[epoch].get(seq_hash, 0))
                + len(indices)
            )

            seq_layer_vec = self._ensure_epoch_seq_layer_vec(epoch, seq_hash,
                                                             layer_idx)
            seq_layer_vec += seq_counts
            seq_layer_rows = self.epoch_seq_layer_route_rows[epoch][seq_hash]
            seq_layer_rows[layer_idx] = int(
                seq_layer_rows.get(layer_idx, 0)) + len(indices)

            if stage >= 0:
                stage_seq_layer_vec = self._ensure_epoch_stage_seq_layer_vec(
                    epoch, stage, seq_hash, layer_idx)
                stage_seq_layer_vec += seq_counts
                stage_seq_layer_rows = self.epoch_seq_stage_layer_route_rows[
                    epoch][stage][seq_hash]
                stage_seq_layer_rows[layer_idx] = int(
                    stage_seq_layer_rows.get(layer_idx, 0)) + len(indices)
        if self.timing_enabled:
            self._record_timing("record_pattern.seq_aggregate",
                                time.perf_counter() - start_phase,
                                rows=usable_rows)
            self._record_timing("record_pattern.total",
                                time.perf_counter() - start_total,
                                rows=usable_rows)

    @torch.no_grad()
    def record(self,
               layer_idx: int,
               topk_ids: torch.Tensor,
               num_experts: int,
               token_types: Optional[torch.Tensor] = None):
        if not self.enabled:
            return
        seq_ids = self.current_batch_seq_ids
        self.top_k_count += 1
        if seq_ids is None:
            return
        assert isinstance(seq_ids, torch.Tensor) and seq_ids.dim() == 1
        seq_ids = seq_ids.detach().to("cpu")
        topk_ids = topk_ids.detach().to("cpu")
        idxs = torch.arange(min(seq_ids.shape[0], topk_ids.shape[0]), device=seq_ids.device).tolist()
        for i in idxs:
            pid = int(seq_ids[i].item())
            self._ensure_vec(pid, layer_idx, num_experts)
            ids = topk_ids[i].tolist()
            for e in ids:
                self.per_prompt[pid][layer_idx][int(e)] += 1.0

    @torch.no_grad()
    def record_no_seqid(self,
                        layer_idx: int,
                        topk_ids: torch.Tensor,
                        num_experts: int,
                        token_types: Optional[torch.Tensor] = None):
        if not self.enabled:
            return
        seq_ids = self.current_batch_seq_ids
        self.top_k_count += 1
        if seq_ids is None:
            return
        topk_ids = topk_ids.detach().to("cpu")
        idxs = torch.arange(topk_ids.shape[0], device=seq_ids.device).tolist()
        seq_ids = 0
        for i in idxs:
            pid = int(seq_ids)
            self._ensure_vec(pid, layer_idx, num_experts)
            ids = topk_ids[i].tolist()
            for e in ids:
                self.per_prompt[pid][layer_idx][int(e)] += 1.0

    def snapshot(self):
        out = {}
        for pid, layers in self.per_prompt.items():
            out[str(pid)] = {str(l): v.tolist() for l, v in layers.items()}
        return out

    def snapshot_pattern(self) -> dict[str, Any]:
        start_total = time.perf_counter() if self.timing_enabled else 0.0
        with self._lock:
            if self.timing_enabled:
                self._record_timing("snapshot_pattern.total",
                                    time.perf_counter() - start_total)
            num_experts = int(self.num_experts or 0)
            return {
                "version": 1,
                "num_experts": num_experts,
                "long_tail_stages": list(self.long_tail_stages),
                "total_expert_counts": (
                    self.total_expert_counts.tolist() if self.total_expert_counts is not None else []
                ),
                "epoch_expert_counts": {
                    str(epoch): counts.tolist()
                    for epoch, counts in self.epoch_expert_counts.items()
                },
                "epoch_stage_expert_counts": {
                    str(epoch): {
                        str(stage): counts.tolist()
                        for stage, counts in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_stage_expert_counts.items()
                },
                "epoch_layer_expert_counts": {
                    str(epoch): {
                        str(layer_idx): counts.tolist()
                        for layer_idx, counts in layer_map.items()
                    }
                    for epoch, layer_map in self.epoch_layer_expert_counts.items()
                },
                "epoch_stage_layer_expert_counts": {
                    str(epoch): {
                        str(stage): {
                            str(layer_idx): counts.tolist()
                            for layer_idx, counts in layer_map.items()
                        }
                        for stage, layer_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_stage_layer_expert_counts.items()
                },
                "epoch_prompt_expert_counts": {
                    str(epoch): {
                        prompt_hash: counts.tolist()
                        for prompt_hash, counts in prompt_map.items()
                    }
                    for epoch, prompt_map in self.epoch_prompt_expert_counts.items()
                },
                "epoch_prompt_stage_expert_counts": {
                    str(epoch): {
                        str(stage): {
                            prompt_hash: counts.tolist()
                            for prompt_hash, counts in prompt_map.items()
                        }
                        for stage, prompt_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_prompt_stage_expert_counts.items()
                },
                "epoch_prompt_layer_expert_counts": {
                    str(epoch): {
                        prompt_hash: {
                            str(layer_idx): counts.tolist()
                            for layer_idx, counts in layer_map.items()
                        }
                        for prompt_hash, layer_map in prompt_map.items()
                    }
                    for epoch, prompt_map in self.epoch_prompt_layer_expert_counts.items()
                },
                "epoch_prompt_stage_layer_expert_counts": {
                    str(epoch): {
                        str(stage): {
                            prompt_hash: {
                                str(layer_idx): counts.tolist()
                                for layer_idx, counts in layer_map.items()
                            }
                            for prompt_hash, layer_map in prompt_map.items()
                        }
                        for stage, prompt_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_prompt_stage_layer_expert_counts.items()
                },
                "epoch_prompt_token_layer_expert_counts": {
                    str(epoch): {
                        prompt_hash: {
                            str(token_position): {
                                str(layer_idx): counts.tolist()
                                for layer_idx, counts in layer_map.items()
                            }
                            for token_position, layer_map in token_map.items()
                        }
                        for prompt_hash, token_map in prompt_map.items()
                    }
                    for epoch, prompt_map in self.epoch_prompt_token_layer_expert_counts.items()
                },
                "epoch_seq_expert_counts": {
                    str(epoch): {
                        seq_hash: counts.tolist()
                        for seq_hash, counts in seq_map.items()
                    }
                    for epoch, seq_map in self.epoch_seq_expert_counts.items()
                },
                "epoch_seq_layer_expert_counts": {
                    str(epoch): {
                        seq_hash: {
                            str(layer_idx): counts.tolist()
                            for layer_idx, counts in layer_map.items()
                        }
                        for seq_hash, layer_map in seq_map.items()
                    }
                    for epoch, seq_map in self.epoch_seq_layer_expert_counts.items()
                },
                "epoch_seq_stage_layer_expert_counts": {
                    str(epoch): {
                        str(stage): {
                            seq_hash: {
                                str(layer_idx): counts.tolist()
                                for layer_idx, counts in layer_map.items()
                            }
                            for seq_hash, layer_map in seq_map.items()
                        }
                        for stage, seq_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_seq_stage_layer_expert_counts.items()
                },
                "epoch_prompt_route_rows": {
                    str(epoch): {prompt_hash: int(count) for prompt_hash, count in prompt_map.items()}
                    for epoch, prompt_map in self.epoch_prompt_route_rows.items()
                },
                "epoch_prompt_layer_route_rows": {
                    str(epoch): {
                        prompt_hash: {
                            str(layer_idx): int(count)
                            for layer_idx, count in layer_map.items()
                        }
                        for prompt_hash, layer_map in prompt_map.items()
                    }
                    for epoch, prompt_map in self.epoch_prompt_layer_route_rows.items()
                },
                "epoch_layer_route_rows": {
                    str(epoch): {str(layer_idx): int(count) for layer_idx, count in layer_map.items()}
                    for epoch, layer_map in self.epoch_layer_route_rows.items()
                },
                "epoch_stage_layer_route_rows": {
                    str(epoch): {
                        str(stage): {str(layer_idx): int(count) for layer_idx, count in layer_map.items()}
                        for stage, layer_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_stage_layer_route_rows.items()
                },
                "epoch_prompt_stage_layer_route_rows": {
                    str(epoch): {
                        str(stage): {
                            prompt_hash: {
                                str(layer_idx): int(count)
                                for layer_idx, count in layer_map.items()
                            }
                            for prompt_hash, layer_map in prompt_map.items()
                        }
                        for stage, prompt_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_prompt_stage_layer_route_rows.items()
                },
                "epoch_prompt_token_layer_route_rows": {
                    str(epoch): {
                        prompt_hash: {
                            str(token_position): {
                                str(layer_idx): int(count)
                                for layer_idx, count in layer_map.items()
                            }
                            for token_position, layer_map in token_map.items()
                        }
                        for prompt_hash, token_map in prompt_map.items()
                    }
                    for epoch, prompt_map in self.epoch_prompt_token_layer_route_rows.items()
                },
                "epoch_prompt_stage_route_rows": {
                    str(epoch): {
                        str(stage): {prompt_hash: int(count) for prompt_hash, count in prompt_map.items()}
                        for stage, prompt_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_prompt_stage_route_rows.items()
                },
                "epoch_seq_route_rows": {
                    str(epoch): {seq_hash: int(count) for seq_hash, count in seq_map.items()}
                    for epoch, seq_map in self.epoch_seq_route_rows.items()
                },
                "epoch_seq_layer_route_rows": {
                    str(epoch): {
                        seq_hash: {
                            str(layer_idx): int(count)
                            for layer_idx, count in layer_map.items()
                        }
                        for seq_hash, layer_map in seq_map.items()
                    }
                    for epoch, seq_map in self.epoch_seq_layer_route_rows.items()
                },
                "epoch_seq_stage_layer_route_rows": {
                    str(epoch): {
                        str(stage): {
                            seq_hash: {
                                str(layer_idx): int(count)
                                for layer_idx, count in layer_map.items()
                            }
                            for seq_hash, layer_map in seq_map.items()
                        }
                        for stage, seq_map in stage_map.items()
                    }
                    for epoch, stage_map in self.epoch_seq_stage_layer_route_rows.items()
                },
                "epoch_seq_parent_prompt": {
                    str(epoch): {
                        seq_hash: prompt_hash
                        for seq_hash, prompt_hash in seq_map.items()
                    }
                    for epoch, seq_map in self.epoch_seq_parent_prompt.items()
                },
                "timing_stats": {
                    name: {
                        "total_s": float(bucket.get("total_s", 0.0)),
                        "calls": int(bucket.get("calls", 0)),
                        "rows": int(bucket.get("rows", 0)),
                        "max_s": float(bucket.get("max_s", 0.0)),
                    }
                    for name, bucket in sorted(self.timing_stats.items())
                },
            }

    def record_topk_ids(self, ep_rank, layer_id, old_topk_ids, new_topk_ids,
                        dump_every=1000, dump_dir="./moe_stats"):
        rank_token_count = getattr(self, "rank_token_count", None)
        if rank_token_count is None:
            self.rank_token_count = defaultdict(int)
            rank_token_count = self.rank_token_count
            self.step_count = 0

        rank_token_count[self.step_count] = [
            int(layer_id),
            old_topk_ids.detach().clone() if hasattr(old_topk_ids, "detach") else old_topk_ids,
            new_topk_ids.detach().clone() if hasattr(new_topk_ids, "detach") else new_topk_ids,
        ]

        if self.step_count == dump_every:
            os.makedirs(dump_dir, exist_ok=True)
            out = {}
            for step, rec in rank_token_count.items():
                old_x, new_x = rec[1], rec[2]
                out[str(step)] = [
                    rec[0],
                    old_x.detach().cpu().tolist() if hasattr(old_x, "detach") else old_x,
                    new_x.detach().cpu().tolist() if hasattr(new_x, "detach") else new_x,
                ]
            path = os.path.join(dump_dir, f"moe_topk_ids_ep{int(ep_rank)}.json")
            with open(path, "w", encoding="utf-8") as f:
                import json
                json.dump(out, f, ensure_ascii=False)

        self.step_count += 1


def merge_moe_pattern_records(worker_records: Any) -> dict[str, Any]:
    start_total = time.perf_counter()
    records = worker_records if isinstance(worker_records, list) else [worker_records]
    valid_records = [record for record in records if isinstance(record, dict) and record.get("version") == 1]
    if not valid_records:
        return {
            "version": 1,
            "num_experts": 0,
            "long_tail_stages": [],
            "total_expert_counts": [],
            "epoch_expert_counts": {},
            "epoch_stage_expert_counts": {},
            "epoch_layer_expert_counts": {},
            "epoch_stage_layer_expert_counts": {},
            "epoch_prompt_expert_counts": {},
            "epoch_prompt_stage_expert_counts": {},
            "epoch_prompt_layer_expert_counts": {},
            "epoch_prompt_stage_layer_expert_counts": {},
            "epoch_prompt_token_layer_expert_counts": {},
            "epoch_seq_expert_counts": {},
            "epoch_seq_layer_expert_counts": {},
            "epoch_seq_stage_layer_expert_counts": {},
            "epoch_prompt_route_rows": {},
            "epoch_prompt_layer_route_rows": {},
            "epoch_layer_route_rows": {},
            "epoch_stage_layer_route_rows": {},
            "epoch_prompt_stage_route_rows": {},
            "epoch_prompt_stage_layer_route_rows": {},
            "epoch_prompt_token_layer_route_rows": {},
            "epoch_seq_route_rows": {},
            "epoch_seq_layer_route_rows": {},
            "epoch_seq_stage_layer_route_rows": {},
            "epoch_seq_parent_prompt": {},
            "timing_stats": {},
        }

    num_experts = int(valid_records[0]["num_experts"])
    merged_total = torch.zeros(num_experts, dtype=torch.int64)
    merged_epoch_counts: dict[int, torch.Tensor] = {}
    merged_stage_counts: dict[int, dict[int, torch.Tensor]] = {}
    merged_layer_counts: dict[int, dict[int, torch.Tensor]] = {}
    merged_stage_layer_counts: dict[int, dict[int, dict[int, torch.Tensor]]] = {}
    merged_prompt_counts: dict[int, dict[str, torch.Tensor]] = {}
    merged_prompt_stage_counts: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    merged_prompt_layer_counts: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
    merged_prompt_stage_layer_counts: dict[int, dict[int, dict[str, dict[int, torch.Tensor]]]] = {}
    merged_prompt_token_layer_counts: dict[int, dict[str, dict[int, dict[int, torch.Tensor]]]] = {}
    merged_seq_counts: dict[int, dict[str, torch.Tensor]] = {}
    merged_seq_layer_counts: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
    merged_seq_stage_layer_counts: dict[int, dict[int, dict[str, dict[int, torch.Tensor]]]] = {}
    merged_prompt_rows: dict[int, dict[str, int]] = defaultdict(dict)
    merged_layer_rows: dict[int, dict[int, int]] = defaultdict(dict)
    merged_stage_layer_rows: dict[int, dict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    merged_prompt_stage_rows: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    merged_prompt_layer_rows: dict[int, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    merged_prompt_stage_layer_rows: dict[int, dict[int, dict[str, dict[int, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    merged_prompt_token_layer_rows: dict[int, dict[str, dict[int, dict[int, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    merged_seq_rows: dict[int, dict[str, int]] = defaultdict(dict)
    merged_seq_layer_rows: dict[int, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    merged_seq_stage_layer_rows: dict[int, dict[int, dict[str, dict[int, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    merged_seq_parent_prompt: dict[int, dict[str, str]] = defaultdict(dict)
    merged_timing_stats: dict[str, dict[str, float]] = {}
    long_tail_stages = set()

    for record in valid_records:
        merged_total += torch.tensor(record.get("total_expert_counts", []), dtype=torch.int64)
        long_tail_stages.update(int(stage) for stage in record.get("long_tail_stages", []))

        for epoch_str, counts in record.get("epoch_expert_counts", {}).items():
            epoch = int(epoch_str)
            merged_epoch_counts.setdefault(epoch, torch.zeros(num_experts, dtype=torch.int64))
            merged_epoch_counts[epoch] += torch.tensor(counts, dtype=torch.int64)

        for epoch_str, stage_map in record.get("epoch_stage_expert_counts", {}).items():
            epoch = int(epoch_str)
            for stage_str, counts in stage_map.items():
                stage = int(stage_str)
                merged_stage_counts.setdefault(epoch, {}).setdefault(
                    stage, torch.zeros(num_experts, dtype=torch.int64)
                )
                merged_stage_counts[epoch][stage] += torch.tensor(counts, dtype=torch.int64)

        for epoch_str, layer_map in record.get("epoch_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for layer_str, counts in layer_map.items():
                layer_idx = int(layer_str)
                merged_layer_counts.setdefault(epoch, {}).setdefault(
                    layer_idx, torch.zeros(num_experts, dtype=torch.int64)
                )
                merged_layer_counts[epoch][layer_idx] += torch.tensor(counts, dtype=torch.int64)

        for epoch_str, stage_map in record.get("epoch_stage_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for stage_str, layer_map in stage_map.items():
                stage = int(stage_str)
                for layer_str, counts in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_stage_layer_counts.setdefault(epoch, {}).setdefault(stage, {}).setdefault(
                        layer_idx, torch.zeros(num_experts, dtype=torch.int64)
                    )
                    merged_stage_layer_counts[epoch][stage][layer_idx] += torch.tensor(
                        counts, dtype=torch.int64
                    )

        for epoch_str, prompt_map in record.get("epoch_prompt_expert_counts", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, counts in prompt_map.items():
                merged_prompt_counts.setdefault(epoch, {}).setdefault(
                    prompt_hash, torch.zeros(num_experts, dtype=torch.int64)
                )
                merged_prompt_counts[epoch][prompt_hash] += torch.tensor(counts, dtype=torch.int64)

        for epoch_str, stage_map in record.get("epoch_prompt_stage_expert_counts", {}).items():
            epoch = int(epoch_str)
            for stage_str, prompt_map in stage_map.items():
                stage = int(stage_str)
                for prompt_hash, counts in prompt_map.items():
                    merged_prompt_stage_counts.setdefault(epoch, {}).setdefault(stage, {}).setdefault(
                        prompt_hash, torch.zeros(num_experts, dtype=torch.int64)
                    )
                    merged_prompt_stage_counts[epoch][stage][prompt_hash] += torch.tensor(
                        counts, dtype=torch.int64
                    )

        for epoch_str, prompt_map in record.get("epoch_prompt_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, layer_map in prompt_map.items():
                for layer_str, counts in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_prompt_layer_counts.setdefault(epoch, {}).setdefault(
                        prompt_hash, {}).setdefault(
                            layer_idx, torch.zeros(num_experts, dtype=torch.int64))
                    merged_prompt_layer_counts[epoch][prompt_hash][layer_idx] += torch.tensor(
                        counts, dtype=torch.int64)

        for epoch_str, stage_map in record.get("epoch_prompt_stage_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for stage_str, prompt_map in stage_map.items():
                stage = int(stage_str)
                for prompt_hash, layer_map in prompt_map.items():
                    for layer_str, counts in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_prompt_stage_layer_counts.setdefault(epoch, {}).setdefault(
                            stage, {}).setdefault(prompt_hash, {}).setdefault(
                                layer_idx, torch.zeros(num_experts, dtype=torch.int64))
                        merged_prompt_stage_layer_counts[epoch][stage][prompt_hash][layer_idx] += torch.tensor(
                            counts, dtype=torch.int64)

        for epoch_str, prompt_map in record.get("epoch_prompt_token_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, token_map in prompt_map.items():
                for token_pos_str, layer_map in token_map.items():
                    token_pos = int(token_pos_str)
                    for layer_str, counts in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_prompt_token_layer_counts.setdefault(epoch, {}).setdefault(
                            prompt_hash, {}).setdefault(token_pos, {}).setdefault(
                                layer_idx, torch.zeros(num_experts, dtype=torch.int64))
                        merged_prompt_token_layer_counts[epoch][prompt_hash][token_pos][layer_idx] += (
                            torch.tensor(counts, dtype=torch.int64))

        for epoch_str, seq_map in record.get("epoch_seq_expert_counts", {}).items():
            epoch = int(epoch_str)
            for seq_hash, counts in seq_map.items():
                merged_seq_counts.setdefault(epoch, {}).setdefault(
                    seq_hash, torch.zeros(num_experts, dtype=torch.int64))
                merged_seq_counts[epoch][seq_hash] += torch.tensor(
                    counts, dtype=torch.int64)

        for epoch_str, seq_map in record.get("epoch_seq_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for seq_hash, layer_map in seq_map.items():
                for layer_str, counts in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_seq_layer_counts.setdefault(epoch, {}).setdefault(
                        seq_hash, {}).setdefault(
                            layer_idx, torch.zeros(num_experts, dtype=torch.int64))
                    merged_seq_layer_counts[epoch][seq_hash][layer_idx] += torch.tensor(
                        counts, dtype=torch.int64)

        for epoch_str, stage_map in record.get("epoch_seq_stage_layer_expert_counts", {}).items():
            epoch = int(epoch_str)
            for stage_str, seq_map in stage_map.items():
                stage = int(stage_str)
                for seq_hash, layer_map in seq_map.items():
                    for layer_str, counts in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_seq_stage_layer_counts.setdefault(epoch, {}).setdefault(
                            stage, {}).setdefault(seq_hash, {}).setdefault(
                                layer_idx, torch.zeros(num_experts, dtype=torch.int64))
                        merged_seq_stage_layer_counts[epoch][stage][seq_hash][layer_idx] += torch.tensor(
                            counts, dtype=torch.int64)

        for epoch_str, prompt_map in record.get("epoch_prompt_route_rows", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, count in prompt_map.items():
                merged_prompt_rows[epoch][prompt_hash] = (
                    int(merged_prompt_rows[epoch].get(prompt_hash, 0)) + int(count)
                )

        for epoch_str, prompt_map in record.get("epoch_prompt_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, layer_map in prompt_map.items():
                for layer_str, count in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_prompt_layer_rows[epoch][prompt_hash][layer_idx] = (
                        int(merged_prompt_layer_rows[epoch][prompt_hash].get(layer_idx, 0))
                        + int(count)
                    )

        for epoch_str, layer_map in record.get("epoch_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for layer_str, count in layer_map.items():
                layer_idx = int(layer_str)
                merged_layer_rows[epoch][layer_idx] = (
                    int(merged_layer_rows[epoch].get(layer_idx, 0)) + int(count)
                )

        for epoch_str, stage_map in record.get("epoch_stage_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for stage_str, layer_map in stage_map.items():
                stage = int(stage_str)
                for layer_str, count in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_stage_layer_rows[epoch][stage][layer_idx] = (
                        int(merged_stage_layer_rows[epoch][stage].get(layer_idx, 0))
                        + int(count)
                    )

        for epoch_str, stage_map in record.get("epoch_prompt_stage_route_rows", {}).items():
            epoch = int(epoch_str)
            for stage_str, prompt_map in stage_map.items():
                stage = int(stage_str)
                for prompt_hash, count in prompt_map.items():
                    merged_prompt_stage_rows[epoch][stage][prompt_hash] = (
                        int(merged_prompt_stage_rows[epoch][stage].get(prompt_hash, 0)) + int(count)
                    )

        for epoch_str, stage_map in record.get("epoch_prompt_stage_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for stage_str, prompt_map in stage_map.items():
                stage = int(stage_str)
                for prompt_hash, layer_map in prompt_map.items():
                    for layer_str, count in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_prompt_stage_layer_rows[epoch][stage][prompt_hash][layer_idx] = (
                            int(merged_prompt_stage_layer_rows[epoch][stage][prompt_hash].get(layer_idx, 0))
                            + int(count)
                        )

        for epoch_str, prompt_map in record.get("epoch_prompt_token_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for prompt_hash, token_map in prompt_map.items():
                for token_pos_str, layer_map in token_map.items():
                    token_pos = int(token_pos_str)
                    for layer_str, count in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_prompt_token_layer_rows[epoch][prompt_hash][token_pos][layer_idx] = (
                            int(merged_prompt_token_layer_rows[epoch][prompt_hash][token_pos].get(layer_idx, 0))
                            + int(count)
                        )

        for epoch_str, seq_map in record.get("epoch_seq_route_rows", {}).items():
            epoch = int(epoch_str)
            for seq_hash, count in seq_map.items():
                merged_seq_rows[epoch][seq_hash] = (
                    int(merged_seq_rows[epoch].get(seq_hash, 0)) + int(count)
                )

        for epoch_str, seq_map in record.get("epoch_seq_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for seq_hash, layer_map in seq_map.items():
                for layer_str, count in layer_map.items():
                    layer_idx = int(layer_str)
                    merged_seq_layer_rows[epoch][seq_hash][layer_idx] = (
                        int(merged_seq_layer_rows[epoch][seq_hash].get(layer_idx, 0))
                        + int(count)
                    )

        for epoch_str, stage_map in record.get("epoch_seq_stage_layer_route_rows", {}).items():
            epoch = int(epoch_str)
            for stage_str, seq_map in stage_map.items():
                stage = int(stage_str)
                for seq_hash, layer_map in seq_map.items():
                    for layer_str, count in layer_map.items():
                        layer_idx = int(layer_str)
                        merged_seq_stage_layer_rows[epoch][stage][seq_hash][layer_idx] = (
                            int(merged_seq_stage_layer_rows[epoch][stage][seq_hash].get(layer_idx, 0))
                            + int(count)
                        )

        for epoch_str, seq_map in record.get("epoch_seq_parent_prompt", {}).items():
            epoch = int(epoch_str)
            for seq_hash, prompt_hash in seq_map.items():
                merged_seq_parent_prompt[epoch].setdefault(seq_hash, prompt_hash)

        for name, bucket in record.get("timing_stats", {}).items():
            _accumulate_timing(
                merged_timing_stats,
                name,
                total_s=float(bucket.get("total_s", 0.0)),
                calls=int(bucket.get("calls", 0)),
                rows=int(bucket.get("rows", 0)),
                max_s=float(bucket.get("max_s", bucket.get("total_s", 0.0))),
            )

    _accumulate_timing(
        merged_timing_stats,
        "merge_moe_pattern_records.total",
        total_s=time.perf_counter() - start_total,
        calls=1,
        rows=len(valid_records),
    )

    return {
        "version": 1,
        "num_experts": num_experts,
        "long_tail_stages": sorted(long_tail_stages),
        "total_expert_counts": merged_total.tolist(),
        "epoch_expert_counts": {
            str(epoch): counts.tolist() for epoch, counts in sorted(merged_epoch_counts.items())
        },
        "epoch_stage_expert_counts": {
            str(epoch): {
                str(stage): counts.tolist()
                for stage, counts in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_stage_counts.items())
        },
        "epoch_layer_expert_counts": {
            str(epoch): {
                str(layer_idx): counts.tolist()
                for layer_idx, counts in sorted(layer_map.items())
            }
            for epoch, layer_map in sorted(merged_layer_counts.items())
        },
        "epoch_stage_layer_expert_counts": {
            str(epoch): {
                str(stage): {
                    str(layer_idx): counts.tolist()
                    for layer_idx, counts in sorted(layer_map.items())
                }
                for stage, layer_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_stage_layer_counts.items())
        },
        "epoch_prompt_expert_counts": {
            str(epoch): {
                prompt_hash: counts.tolist()
                for prompt_hash, counts in sorted(prompt_map.items())
            }
            for epoch, prompt_map in sorted(merged_prompt_counts.items())
        },
        "epoch_prompt_stage_expert_counts": {
            str(epoch): {
                str(stage): {
                    prompt_hash: counts.tolist()
                    for prompt_hash, counts in sorted(prompt_map.items())
                }
                for stage, prompt_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_prompt_stage_counts.items())
        },
        "epoch_prompt_layer_expert_counts": {
            str(epoch): {
                prompt_hash: {
                    str(layer_idx): counts.tolist()
                    for layer_idx, counts in sorted(layer_map.items())
                }
                for prompt_hash, layer_map in sorted(prompt_map.items())
            }
            for epoch, prompt_map in sorted(merged_prompt_layer_counts.items())
        },
        "epoch_prompt_stage_layer_expert_counts": {
            str(epoch): {
                str(stage): {
                    prompt_hash: {
                        str(layer_idx): counts.tolist()
                        for layer_idx, counts in sorted(layer_map.items())
                    }
                    for prompt_hash, layer_map in sorted(prompt_map.items())
                }
                for stage, prompt_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_prompt_stage_layer_counts.items())
        },
        "epoch_prompt_token_layer_expert_counts": {
            str(epoch): {
                prompt_hash: {
                    str(token_pos): {
                        str(layer_idx): counts.tolist()
                        for layer_idx, counts in sorted(layer_map.items())
                    }
                    for token_pos, layer_map in sorted(token_map.items())
                }
                for prompt_hash, token_map in sorted(prompt_map.items())
            }
            for epoch, prompt_map in sorted(merged_prompt_token_layer_counts.items())
        },
        "epoch_seq_expert_counts": {
            str(epoch): {
                seq_hash: counts.tolist()
                for seq_hash, counts in sorted(seq_map.items())
            }
            for epoch, seq_map in sorted(merged_seq_counts.items())
        },
        "epoch_seq_layer_expert_counts": {
            str(epoch): {
                seq_hash: {
                    str(layer_idx): counts.tolist()
                    for layer_idx, counts in sorted(layer_map.items())
                }
                for seq_hash, layer_map in sorted(seq_map.items())
            }
            for epoch, seq_map in sorted(merged_seq_layer_counts.items())
        },
        "epoch_seq_stage_layer_expert_counts": {
            str(epoch): {
                str(stage): {
                    seq_hash: {
                        str(layer_idx): counts.tolist()
                        for layer_idx, counts in sorted(layer_map.items())
                    }
                    for seq_hash, layer_map in sorted(seq_map.items())
                }
                for stage, seq_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_seq_stage_layer_counts.items())
        },
        "epoch_prompt_route_rows": {
            str(epoch): {prompt_hash: int(count) for prompt_hash, count in sorted(prompt_map.items())}
            for epoch, prompt_map in sorted(merged_prompt_rows.items())
        },
        "epoch_prompt_layer_route_rows": {
            str(epoch): {
                prompt_hash: {
                    str(layer_idx): int(count)
                    for layer_idx, count in sorted(layer_map.items())
                }
                for prompt_hash, layer_map in sorted(prompt_map.items())
            }
            for epoch, prompt_map in sorted(merged_prompt_layer_rows.items())
        },
        "epoch_layer_route_rows": {
            str(epoch): {str(layer_idx): int(count) for layer_idx, count in sorted(layer_map.items())}
            for epoch, layer_map in sorted(merged_layer_rows.items())
        },
        "epoch_stage_layer_route_rows": {
            str(epoch): {
                str(stage): {str(layer_idx): int(count) for layer_idx, count in sorted(layer_map.items())}
                for stage, layer_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_stage_layer_rows.items())
        },
        "epoch_prompt_stage_route_rows": {
            str(epoch): {
                str(stage): {prompt_hash: int(count) for prompt_hash, count in sorted(prompt_map.items())}
                for stage, prompt_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_prompt_stage_rows.items())
        },
        "epoch_prompt_stage_layer_route_rows": {
            str(epoch): {
                str(stage): {
                    prompt_hash: {
                        str(layer_idx): int(count)
                        for layer_idx, count in sorted(layer_map.items())
                    }
                    for prompt_hash, layer_map in sorted(prompt_map.items())
                }
                for stage, prompt_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_prompt_stage_layer_rows.items())
        },
        "epoch_prompt_token_layer_route_rows": {
            str(epoch): {
                prompt_hash: {
                    str(token_pos): {
                        str(layer_idx): int(count)
                        for layer_idx, count in sorted(layer_map.items())
                    }
                    for token_pos, layer_map in sorted(token_map.items())
                }
                for prompt_hash, token_map in sorted(prompt_map.items())
            }
            for epoch, prompt_map in sorted(merged_prompt_token_layer_rows.items())
        },
        "epoch_seq_route_rows": {
            str(epoch): {seq_hash: int(count) for seq_hash, count in sorted(seq_map.items())}
            for epoch, seq_map in sorted(merged_seq_rows.items())
        },
        "epoch_seq_layer_route_rows": {
            str(epoch): {
                seq_hash: {
                    str(layer_idx): int(count)
                    for layer_idx, count in sorted(layer_map.items())
                }
                for seq_hash, layer_map in sorted(seq_map.items())
            }
            for epoch, seq_map in sorted(merged_seq_layer_rows.items())
        },
        "epoch_seq_stage_layer_route_rows": {
            str(epoch): {
                str(stage): {
                    seq_hash: {
                        str(layer_idx): int(count)
                        for layer_idx, count in sorted(layer_map.items())
                    }
                    for seq_hash, layer_map in sorted(seq_map.items())
                }
                for stage, seq_map in sorted(stage_map.items())
            }
            for epoch, stage_map in sorted(merged_seq_stage_layer_rows.items())
        },
        "epoch_seq_parent_prompt": {
            str(epoch): {
                seq_hash: prompt_hash
                for seq_hash, prompt_hash in sorted(seq_map.items())
            }
            for epoch, seq_map in sorted(merged_seq_parent_prompt.items())
        },
        "timing_stats": {
            name: {
                "total_s": float(bucket.get("total_s", 0.0)),
                "calls": int(bucket.get("calls", 0)),
                "rows": int(bucket.get("rows", 0)),
                "max_s": float(bucket.get("max_s", 0.0)),
            }
            for name, bucket in sorted(merged_timing_stats.items())
        },
    }


def write_prompt_token_epoch_artifact(record: dict[str, Any],
                                      output_dir: str,
                                      epoch: int) -> Optional[Path]:
    export_mode = _prompt_token_export_mode()
    if export_mode in ("off", "csv"):
        return None
    if export_mode not in ("compact",):
        logger.warning(
            "Unknown VLLM_MOE_PROMPT_TOKEN_EXPORT_MODE=%s; defaulting to compact export.",
            export_mode,
        )

    num_experts = int(record.get("num_experts", 0))
    if num_experts <= 0:
        return None

    epoch_key = str(int(epoch))
    epoch_prompt_map = record.get("epoch_prompt_token_layer_expert_counts",
                                  {}).get(epoch_key, {})
    if not epoch_prompt_map:
        return None

    epoch_route_rows = record.get("epoch_prompt_token_layer_route_rows",
                                  {}).get(epoch_key, {})
    artifact_dir = (Path(output_dir) / "prompt_token_layer_artifacts" /
                    f"epoch_{int(epoch):04d}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    counts_dtype = torch.uint16
    route_rows_dtype = torch.uint16
    accum_dtype = torch.int32
    prompt_files: list[dict[str, Any]] = []
    total_prompt_token_layer_keys = 0
    total_bytes = 0
    max_num_layers = 0

    for prompt_idx, (prompt_hash,
                     token_map) in enumerate(sorted(epoch_prompt_map.items())):
        if not token_map:
            continue

        max_token_position = max(int(token_pos) for token_pos in token_map)
        max_layer_idx = -1
        for layer_map in token_map.values():
            if layer_map:
                max_layer_idx = max(max_layer_idx,
                                    max(int(layer_idx)
                                        for layer_idx in layer_map.keys()))
        prompt_route_rows = epoch_route_rows.get(prompt_hash, {})
        if prompt_route_rows:
            max_token_position = max(
                max_token_position,
                max(int(token_pos) for token_pos in prompt_route_rows.keys()),
            )
        for layer_map in prompt_route_rows.values():
            if layer_map:
                max_layer_idx = max(max_layer_idx,
                                    max(int(layer_idx)
                                        for layer_idx in layer_map.keys()))
        if max_layer_idx < 0:
            continue

        num_layers = max_layer_idx + 1
        max_num_layers = max(max_num_layers, num_layers)
        counts_tensor = torch.zeros(
            (max_token_position + 1, num_layers, num_experts),
            dtype=accum_dtype,
        )
        route_rows_tensor = torch.zeros(
            (max_token_position + 1, num_layers),
            dtype=accum_dtype,
        )

        nonzero_keys = 0
        for token_pos_str, layer_map in sorted(token_map.items(),
                                               key=lambda item: int(item[0])):
            token_pos = int(token_pos_str)
            for layer_str, counts in sorted(layer_map.items(),
                                            key=lambda item: int(item[0])):
                layer_idx = int(layer_str)
                counts_tensor[token_pos,
                              layer_idx] = torch.as_tensor(counts,
                                                           dtype=counts_dtype)
                nonzero_keys += 1

        for token_pos_str, layer_map in sorted(prompt_route_rows.items(),
                                               key=lambda item: int(item[0])):
            token_pos = int(token_pos_str)
            for layer_str, count in sorted(layer_map.items(),
                                           key=lambda item: int(item[0])):
                layer_idx = int(layer_str)
                route_rows_tensor[token_pos, layer_idx] = int(count)

        safe_hash = _sanitize_filename_fragment(prompt_hash)
        file_name = f"prompt_{prompt_idx:04d}_{safe_hash}.pt"
        file_path = artifact_dir / file_name
        torch.save(
            {
                "version": 1,
                "epoch": int(epoch),
                "prompt_hash": prompt_hash,
                "num_experts": num_experts,
                "counts_dtype": str(counts_dtype).replace("torch.", ""),
                "route_rows_dtype": str(route_rows_dtype).replace(
                    "torch.", ""),
                "counts": counts_tensor,
                "route_rows": route_rows_tensor,
            },
            file_path,
        )

        total_prompt_token_layer_keys += nonzero_keys
        total_bytes += (
            counts_tensor.numel() * counts_tensor.element_size()
            + route_rows_tensor.numel() * route_rows_tensor.element_size())
        prompt_files.append(
            {
                "prompt_hash": prompt_hash,
                "file": file_name,
                "max_token_position": int(max_token_position),
                "num_layers": int(num_layers),
                "nonzero_token_layer_keys": int(nonzero_keys),
                "counts_shape": list(counts_tensor.shape),
                "route_rows_shape": list(route_rows_tensor.shape),
                "storage_bytes": int(
                    counts_tensor.numel() * counts_tensor.element_size()
                    + route_rows_tensor.numel() * route_rows_tensor.element_size()),
            })

    manifest_path = artifact_dir / "manifest.pt"
    torch.save(
        {
            "version": 1,
            "epoch": int(epoch),
            "num_experts": num_experts,
            "counts_dtype": str(counts_dtype).replace("torch.", ""),
            "route_rows_dtype": str(route_rows_dtype).replace("torch.", ""),
            "num_prompts": len(prompt_files),
            "max_num_layers": int(max_num_layers),
            "total_prompt_token_layer_keys": int(total_prompt_token_layer_keys),
            "total_storage_bytes": int(total_bytes),
            "prompt_files": prompt_files,
        },
        manifest_path,
    )
    logger.info(
        "MoE prompt-token artifact written: epoch=%s prompts=%s keys=%s storage_mb=%.3f dir=%s",
        epoch,
        len(prompt_files),
        total_prompt_token_layer_keys,
        float(total_bytes) / (1024.0 * 1024.0),
        artifact_dir,
    )
    return artifact_dir


def write_prompt_token_epoch_artifact_from_worker_records(
        worker_records: Any, output_dir: str, epoch: int) -> Optional[Path]:
    export_mode = _prompt_token_export_mode()
    if export_mode in ("off", "csv"):
        return None
    if export_mode not in ("compact",):
        logger.warning(
            "Unknown VLLM_MOE_PROMPT_TOKEN_EXPORT_MODE=%s; defaulting to compact export.",
            export_mode,
        )

    records = worker_records if isinstance(worker_records,
                                           list) else [worker_records]
    valid_records = [
        record for record in records
        if isinstance(record, dict) and record.get("version") == 1
    ]
    if not valid_records:
        return None

    num_experts = int(valid_records[0].get("num_experts", 0))
    if num_experts <= 0:
        return None

    epoch_key = str(int(epoch))
    artifact_dir = (Path(output_dir) / "prompt_token_layer_artifacts" /
                    f"epoch_{int(epoch):04d}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prompt_sources: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    prompt_meta: dict[str, dict[str, int]] = {}

    for record in valid_records:
        prompt_map = record.get("epoch_prompt_token_layer_expert_counts",
                                {}).get(epoch_key, {})
        route_map = record.get("epoch_prompt_token_layer_route_rows",
                               {}).get(epoch_key, {})
        seen_prompts: set[str] = set()

        for prompt_hash, token_map in prompt_map.items():
            prompt_route_rows = route_map.get(prompt_hash, {})
            prompt_sources[prompt_hash].append((token_map, prompt_route_rows))
            seen_prompts.add(prompt_hash)
            meta = prompt_meta.setdefault(prompt_hash, {
                "max_token_position": -1,
                "max_layer_idx": -1,
            })
            for token_pos_str, layer_map in token_map.items():
                token_pos = int(token_pos_str)
                meta["max_token_position"] = max(meta["max_token_position"],
                                                  token_pos)
                if layer_map:
                    meta["max_layer_idx"] = max(
                        meta["max_layer_idx"],
                        max(int(layer_idx) for layer_idx in layer_map.keys()),
                    )
            for token_pos_str, layer_map in prompt_route_rows.items():
                token_pos = int(token_pos_str)
                meta["max_token_position"] = max(meta["max_token_position"],
                                                  token_pos)
                if layer_map:
                    meta["max_layer_idx"] = max(
                        meta["max_layer_idx"],
                        max(int(layer_idx) for layer_idx in layer_map.keys()),
                    )

        for prompt_hash, prompt_route_rows in route_map.items():
            if prompt_hash in seen_prompts:
                continue
            prompt_sources[prompt_hash].append(({}, prompt_route_rows))
            meta = prompt_meta.setdefault(prompt_hash, {
                "max_token_position": -1,
                "max_layer_idx": -1,
            })
            for token_pos_str, layer_map in prompt_route_rows.items():
                token_pos = int(token_pos_str)
                meta["max_token_position"] = max(meta["max_token_position"],
                                                  token_pos)
                if layer_map:
                    meta["max_layer_idx"] = max(
                        meta["max_layer_idx"],
                        max(int(layer_idx) for layer_idx in layer_map.keys()),
                    )

    if not prompt_sources:
        return None

    counts_dtype = torch.uint16
    route_rows_dtype = torch.uint16
    accum_dtype = torch.int32
    prompt_files: list[dict[str, Any]] = []
    total_prompt_token_layer_keys = 0
    total_bytes = 0
    max_num_layers = 0

    for prompt_idx, prompt_hash in enumerate(sorted(prompt_sources.keys())):
        meta = prompt_meta.get(prompt_hash, {})
        max_token_position = int(meta.get("max_token_position", -1))
        max_layer_idx = int(meta.get("max_layer_idx", -1))
        if max_token_position < 0 or max_layer_idx < 0:
            continue

        num_layers = max_layer_idx + 1
        max_num_layers = max(max_num_layers, num_layers)
        counts_tensor = torch.zeros(
            (max_token_position + 1, num_layers, num_experts),
            dtype=accum_dtype,
        )
        route_rows_tensor = torch.zeros(
            (max_token_position + 1, num_layers),
            dtype=accum_dtype,
        )

        nonzero_keys = 0
        seen_nonzero_keys: set[tuple[int, int]] = set()
        for token_map, prompt_route_rows in prompt_sources[prompt_hash]:
            for token_pos_str, layer_map in token_map.items():
                token_pos = int(token_pos_str)
                for layer_str, counts in layer_map.items():
                    layer_idx = int(layer_str)
                    counts_tensor[token_pos, layer_idx] += torch.as_tensor(
                        counts, dtype=accum_dtype)
                    key = (token_pos, layer_idx)
                    if key not in seen_nonzero_keys:
                        seen_nonzero_keys.add(key)
                        nonzero_keys += 1
            for token_pos_str, layer_map in prompt_route_rows.items():
                token_pos = int(token_pos_str)
                for layer_str, count in layer_map.items():
                    layer_idx = int(layer_str)
                    route_rows_tensor[token_pos, layer_idx] += torch.tensor(
                        int(count), dtype=accum_dtype)

        max_count = (
            int(counts_tensor.to(dtype=torch.int32).max().item())
            if counts_tensor.numel() > 0 else 0
        )
        max_route_rows = (
            int(route_rows_tensor.to(dtype=torch.int32).max().item())
            if route_rows_tensor.numel() > 0 else 0
        )
        if max_count > 65535 or max_route_rows > 65535:
            raise RuntimeError(
                f"Prompt-token artifact overflow for prompt={prompt_hash}: "
                f"max_count={max_count} max_route_rows={max_route_rows}"
            )
        counts_tensor = counts_tensor.to(dtype=counts_dtype)
        route_rows_tensor = route_rows_tensor.to(dtype=route_rows_dtype)

        safe_hash = _sanitize_filename_fragment(prompt_hash)
        file_name = f"prompt_{prompt_idx:04d}_{safe_hash}.pt"
        file_path = artifact_dir / file_name
        torch.save(
            {
                "version": 1,
                "epoch": int(epoch),
                "prompt_hash": prompt_hash,
                "num_experts": num_experts,
                "counts_dtype": str(counts_dtype).replace("torch.", ""),
                "route_rows_dtype": str(route_rows_dtype).replace(
                    "torch.", ""),
                "counts": counts_tensor,
                "route_rows": route_rows_tensor,
            },
            file_path,
        )

        storage_bytes = int(counts_tensor.numel() * counts_tensor.element_size()
                            + route_rows_tensor.numel()
                            * route_rows_tensor.element_size())
        total_prompt_token_layer_keys += nonzero_keys
        total_bytes += storage_bytes
        prompt_files.append(
            {
                "prompt_hash": prompt_hash,
                "file": file_name,
                "max_token_position": int(max_token_position),
                "num_layers": int(num_layers),
                "nonzero_token_layer_keys": int(nonzero_keys),
                "counts_shape": list(counts_tensor.shape),
                "route_rows_shape": list(route_rows_tensor.shape),
                "storage_bytes": storage_bytes,
            })

    manifest_path = artifact_dir / "manifest.pt"
    torch.save(
        {
            "version": 1,
            "epoch": int(epoch),
            "num_experts": num_experts,
            "counts_dtype": str(counts_dtype).replace("torch.", ""),
            "route_rows_dtype": str(route_rows_dtype).replace("torch.", ""),
            "num_prompts": len(prompt_files),
            "max_num_layers": int(max_num_layers),
            "total_prompt_token_layer_keys": int(total_prompt_token_layer_keys),
            "total_storage_bytes": int(total_bytes),
            "prompt_files": prompt_files,
        },
        manifest_path,
    )
    logger.info(
        "MoE prompt-token artifact written from worker records: epoch=%s prompts=%s keys=%s storage_mb=%.3f dir=%s",
        epoch,
        len(prompt_files),
        total_prompt_token_layer_keys,
        float(total_bytes) / (1024.0 * 1024.0),
        artifact_dir,
    )
    return artifact_dir


def strip_prompt_token_from_worker_records(worker_records: Any) -> Any:
    if _prompt_token_export_mode() != "compact":
        return worker_records
    records = worker_records if isinstance(worker_records,
                                           list) else [worker_records]
    for record in records:
        if not isinstance(record, dict):
            continue
        record.pop("epoch_prompt_token_layer_expert_counts", None)
        record.pop("epoch_prompt_token_layer_route_rows", None)
    return worker_records


def write_moe_pattern_csvs(record: dict[str, Any], output_dir: str,
                           prefix: str = "cumulative") -> None:
    start_total = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_experts = int(record.get("num_experts", 0))
    if num_experts <= 0:
        return

    timing_rows = _timing_stats_to_rows(record.get("timing_stats", {}))
    if timing_rows:
        _write_csv(
            out_dir / f"{prefix}_timing_summary.csv",
            [
                "name",
                "total_s",
                "total_ms",
                "calls",
                "avg_ms_per_call",
                "max_ms",
                "rows",
                "avg_us_per_row",
            ],
            timing_rows,
        )

    total_counts = [int(x) for x in record.get("total_expert_counts", [])]
    total_sum = int(sum(total_counts))
    total_ranks = _dense_rank_from_counts(total_counts)
    total_rows = []
    for expert_id, count in enumerate(total_counts):
        total_rows.append({
            "expert_id": expert_id,
            "edge_count": count,
            "share_all_edges": (float(count) / total_sum) if total_sum > 0 else 0.0,
            "hot_rank": total_ranks[expert_id],
        })
    _write_csv(
        out_dir / f"{prefix}_total_hot_experts.csv",
        ["expert_id", "edge_count", "share_all_edges", "hot_rank"],
        total_rows,
    )

    epoch_rows = []
    for epoch_str, counts in sorted(record.get("epoch_expert_counts", {}).items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_counts = [int(x) for x in counts]
        epoch_sum = int(sum(epoch_counts))
        epoch_ranks = _dense_rank_from_counts(epoch_counts)
        for expert_id, count in enumerate(epoch_counts):
            epoch_rows.append({
                "epoch": epoch,
                "expert_id": expert_id,
                "edge_count": count,
                "share_epoch_edges": (float(count) / epoch_sum) if epoch_sum > 0 else 0.0,
                "hot_rank": epoch_ranks[expert_id],
            })
    _write_csv(
        out_dir / f"{prefix}_epoch_hot_experts.csv",
        ["epoch", "expert_id", "edge_count", "share_epoch_edges", "hot_rank"],
        epoch_rows,
    )

    stage_rows = []
    for epoch_str, stage_map in sorted(record.get("epoch_stage_expert_counts", {}).items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        for stage_str, counts in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_counts = [int(x) for x in counts]
            stage_sum = int(sum(stage_counts))
            stage_ranks = _dense_rank_from_counts(stage_counts)
            for expert_id, count in enumerate(stage_counts):
                stage_rows.append({
                    "epoch": epoch,
                    "stage_active_ranks": stage,
                    "expert_id": expert_id,
                    "edge_count": count,
                    "share_stage_edges": (float(count) / stage_sum) if stage_sum > 0 else 0.0,
                    "hot_rank": stage_ranks[expert_id],
                })
    _write_csv(
        out_dir / f"{prefix}_long_tail_stage_hot_experts.csv",
        ["epoch", "stage_active_ranks", "expert_id", "edge_count", "share_stage_edges", "hot_rank"],
        stage_rows,
    )

    layer_rows = []
    layer_route_rows = record.get("epoch_layer_route_rows", {})
    for epoch_str, layer_map in sorted(record.get("epoch_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_layer_rows = layer_route_rows.get(str(epoch), {})
        for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
            layer_idx = int(layer_str)
            layer_counts = [int(x) for x in counts]
            layer_sum = int(sum(layer_counts))
            layer_ranks = _dense_rank_from_counts(layer_counts)
            route_rows = int(epoch_layer_rows.get(str(layer_idx), 0))
            for expert_id, count in enumerate(layer_counts):
                layer_rows.append({
                    "epoch": epoch,
                    "layer_idx": layer_idx,
                    "expert_id": expert_id,
                    "edge_count": count,
                    "share_layer_edges": (float(count) / layer_sum) if layer_sum > 0 else 0.0,
                    "selected": int(count > 0),
                    "hot_rank": layer_ranks[expert_id],
                    "route_rows": route_rows,
                })
    _write_csv(
        out_dir / f"{prefix}_layer_hot_experts.csv",
        [
            "epoch",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_layer_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        layer_rows,
    )

    layer_stage_rows = []
    layer_stage_route_rows = record.get("epoch_stage_layer_route_rows", {})
    for epoch_str, stage_map in sorted(record.get("epoch_stage_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_stage_route_rows = layer_stage_route_rows.get(str(epoch), {})
        for stage_str, layer_map in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_layer_rows = epoch_stage_route_rows.get(str(stage), {})
            for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                layer_idx = int(layer_str)
                count_list = [int(x) for x in counts]
                stage_layer_sum = int(sum(count_list))
                stage_layer_ranks = _dense_rank_from_counts(count_list)
                route_rows = int(stage_layer_rows.get(str(layer_idx), 0))
                for expert_id, count in enumerate(count_list):
                    layer_stage_rows.append({
                        "epoch": epoch,
                        "stage_active_ranks": stage,
                        "layer_idx": layer_idx,
                        "expert_id": expert_id,
                        "edge_count": count,
                        "share_layer_stage_edges": (
                            float(count) / stage_layer_sum) if stage_layer_sum > 0 else 0.0,
                        "selected": int(count > 0),
                        "hot_rank": stage_layer_ranks[expert_id],
                        "route_rows": route_rows,
                    })
    _write_csv(
        out_dir / f"{prefix}_layer_stage_hot_experts.csv",
        [
            "epoch",
            "stage_active_ranks",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_layer_stage_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        layer_stage_rows,
    )

    prompt_layer_rows = []
    prompt_layer_route_rows = record.get("epoch_prompt_layer_route_rows", {})
    prompt_layer_history: dict[str, dict[int, dict[int, list[int]]]] = defaultdict(
        lambda: defaultdict(dict))
    prompt_layer_similarity_rows = []
    for epoch_str, prompt_map in sorted(record.get("epoch_prompt_layer_expert_counts", {}).items(),
                                        key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_prompt_rows = prompt_layer_route_rows.get(str(epoch), {})
        for prompt_hash, layer_map in sorted(prompt_map.items()):
            prompt_route_rows = epoch_prompt_rows.get(prompt_hash, {})
            for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                layer_idx = int(layer_str)
                count_list = [int(x) for x in counts]
                prompt_layer_sum = int(sum(count_list))
                prompt_layer_ranks = _dense_rank_from_counts(count_list)
                route_rows = int(prompt_route_rows.get(str(layer_idx), 0))
                for expert_id, count in enumerate(count_list):
                    prompt_layer_rows.append({
                        "prompt_hash": prompt_hash,
                        "epoch": epoch,
                        "layer_idx": layer_idx,
                        "expert_id": expert_id,
                        "edge_count": count,
                        "share_prompt_layer_edges": (
                            float(count) / prompt_layer_sum) if prompt_layer_sum > 0 else 0.0,
                        "selected": int(count > 0),
                        "hot_rank": prompt_layer_ranks[expert_id],
                        "route_rows": route_rows,
                    })
                prev_epoch = max(
                    (e for e in prompt_layer_history[prompt_hash][layer_idx].keys() if e < epoch),
                    default=None,
                )
                if prev_epoch is not None:
                    prev_counts = prompt_layer_history[prompt_hash][layer_idx][prev_epoch]
                    prev_route_rows = int(
                        prompt_layer_route_rows.get(str(prev_epoch), {}).get(
                            prompt_hash, {}).get(str(layer_idx), 0))
                    top_expert_id = _top_expert_id_from_counts(count_list)
                    prev_top_expert_id = _top_expert_id_from_counts(prev_counts)
                    prompt_layer_similarity_rows.append({
                        "prompt_hash": prompt_hash,
                        "epoch": epoch,
                        "prev_epoch": prev_epoch,
                        "layer_idx": layer_idx,
                        "cosine_similarity": _cosine_similarity_from_counts(
                            count_list, prev_counts),
                        "jaccard_selected_experts": _jaccard_selected_from_counts(
                            count_list, prev_counts),
                        "top_expert_id": top_expert_id,
                        "prev_top_expert_id": prev_top_expert_id,
                        "top1_match": int(top_expert_id >= 0 and top_expert_id == prev_top_expert_id),
                        "route_rows": route_rows,
                        "prev_route_rows": prev_route_rows,
                    })
                prompt_layer_history[prompt_hash][layer_idx][epoch] = count_list
    _write_csv(
        out_dir / f"{prefix}_prompt_layer_hot_experts.csv",
        [
            "prompt_hash",
            "epoch",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_prompt_layer_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        prompt_layer_rows,
    )
    _write_csv(
        out_dir / f"{prefix}_prompt_layer_epoch_similarity.csv",
        [
            "prompt_hash",
            "epoch",
            "prev_epoch",
            "layer_idx",
            "cosine_similarity",
            "jaccard_selected_experts",
            "top_expert_id",
            "prev_top_expert_id",
            "top1_match",
            "route_rows",
            "prev_route_rows",
        ],
        prompt_layer_similarity_rows,
    )

    prompt_token_export_mode = _prompt_token_export_mode()
    if prompt_token_export_mode == "csv":
        prompt_token_layer_rows = []
        prompt_token_layer_route_rows = record.get(
            "epoch_prompt_token_layer_route_rows", {})
        prompt_token_layer_history: dict[str,
                                         dict[int, dict[int, dict[int,
                                                                  list[int]]]]] = defaultdict(
                                                                      lambda: defaultdict(
                                                                          lambda: defaultdict(
                                                                              dict)))
        prompt_token_layer_similarity_rows = []
        for epoch_str, prompt_map in sorted(
                record.get("epoch_prompt_token_layer_expert_counts", {}).items(),
                key=lambda x: int(x[0])):
            epoch = int(epoch_str)
            epoch_prompt_rows = prompt_token_layer_route_rows.get(str(epoch), {})
            for prompt_hash, token_map in sorted(prompt_map.items()):
                prompt_token_rows = epoch_prompt_rows.get(prompt_hash, {})
                for token_pos_str, layer_map in sorted(token_map.items(),
                                                       key=lambda x: int(x[0])):
                    token_pos = int(token_pos_str)
                    token_route_rows = prompt_token_rows.get(str(token_pos), {})
                    for layer_str, counts in sorted(layer_map.items(),
                                                    key=lambda x: int(x[0])):
                        layer_idx = int(layer_str)
                        count_list = [int(x) for x in counts]
                        total_edges = int(sum(count_list))
                        ranks = _dense_rank_from_counts(count_list)
                        route_rows = int(token_route_rows.get(str(layer_idx), 0))
                        for expert_id, count in enumerate(count_list):
                            prompt_token_layer_rows.append({
                                "prompt_hash": prompt_hash,
                                "epoch": epoch,
                                "token_position": token_pos,
                                "layer_idx": layer_idx,
                                "expert_id": expert_id,
                                "edge_count": count,
                                "share_prompt_token_layer_edges": (
                                    float(count) / total_edges) if total_edges > 0 else 0.0,
                                "selected": int(count > 0),
                                "hot_rank": ranks[expert_id],
                                "route_rows": route_rows,
                            })
                        prev_epoch = max(
                            (e for e in prompt_token_layer_history[prompt_hash]
                             [token_pos][layer_idx].keys() if e < epoch),
                            default=None,
                        )
                        if prev_epoch is not None:
                            prev_counts = prompt_token_layer_history[prompt_hash][
                                token_pos][layer_idx][prev_epoch]
                            prev_route_rows = int(
                                prompt_token_layer_route_rows.get(
                                    str(prev_epoch), {}).get(prompt_hash, {}).get(
                                        str(token_pos), {}).get(str(layer_idx), 0))
                            top_expert_id = _top_expert_id_from_counts(count_list)
                            prev_top_expert_id = _top_expert_id_from_counts(
                                prev_counts)
                            current_selected, prev_selected = (
                                _selected_expert_sets_from_counts(count_list,
                                                                  prev_counts))
                            prompt_token_layer_similarity_rows.append({
                                "prompt_hash": prompt_hash,
                                "epoch": epoch,
                                "prev_epoch": prev_epoch,
                                "token_position": token_pos,
                                "layer_idx": layer_idx,
                                "cosine_similarity": _cosine_similarity_from_counts(
                                    count_list, prev_counts),
                                "jaccard_selected_experts":
                                _jaccard_selected_from_counts(count_list,
                                                              prev_counts),
                                "top_expert_id": top_expert_id,
                                "prev_top_expert_id": prev_top_expert_id,
                                "top1_match": int(top_expert_id >= 0
                                                  and top_expert_id
                                                  == prev_top_expert_id),
                                "shared_selected_expert_count": int(
                                    len(current_selected & prev_selected)),
                                "prev_only_expert_count": int(
                                    len(prev_selected - current_selected)),
                                "curr_only_expert_count": int(
                                    len(current_selected - prev_selected)),
                                "route_rows": route_rows,
                                "prev_route_rows": prev_route_rows,
                            })
                        prompt_token_layer_history[prompt_hash][token_pos][
                            layer_idx][epoch] = count_list
        _write_csv(
            out_dir / f"{prefix}_prompt_token_layer_hot_experts.csv",
            [
                "prompt_hash",
                "epoch",
                "token_position",
                "layer_idx",
                "expert_id",
                "edge_count",
                "share_prompt_token_layer_edges",
                "selected",
                "hot_rank",
                "route_rows",
            ],
            prompt_token_layer_rows,
        )
        _write_csv(
            out_dir / f"{prefix}_prompt_token_layer_epoch_similarity.csv",
            [
                "prompt_hash",
                "epoch",
                "prev_epoch",
                "token_position",
                "layer_idx",
                "cosine_similarity",
                "jaccard_selected_experts",
                "top_expert_id",
                "prev_top_expert_id",
                "top1_match",
                "shared_selected_expert_count",
                "prev_only_expert_count",
                "curr_only_expert_count",
                "route_rows",
                "prev_route_rows",
            ],
            prompt_token_layer_similarity_rows,
        )

    prompt_stage_layer_rows = []
    prompt_stage_layer_route_rows = record.get("epoch_prompt_stage_layer_route_rows", {})
    for epoch_str, stage_map in sorted(record.get("epoch_prompt_stage_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_stage_rows = prompt_stage_layer_route_rows.get(str(epoch), {})
        for stage_str, prompt_map in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_prompt_rows = epoch_stage_rows.get(str(stage), {})
            for prompt_hash, layer_map in sorted(prompt_map.items()):
                prompt_route_rows = stage_prompt_rows.get(prompt_hash, {})
                for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                    layer_idx = int(layer_str)
                    count_list = [int(x) for x in counts]
                    total_edges = int(sum(count_list))
                    ranks = _dense_rank_from_counts(count_list)
                    route_rows = int(prompt_route_rows.get(str(layer_idx), 0))
                    for expert_id, count in enumerate(count_list):
                        prompt_stage_layer_rows.append({
                            "prompt_hash": prompt_hash,
                            "epoch": epoch,
                            "stage_active_ranks": stage,
                            "layer_idx": layer_idx,
                            "expert_id": expert_id,
                            "edge_count": count,
                            "share_prompt_stage_layer_edges": (
                                float(count) / total_edges) if total_edges > 0 else 0.0,
                            "selected": int(count > 0),
                            "hot_rank": ranks[expert_id],
                            "route_rows": route_rows,
                        })
    _write_csv(
        out_dir / f"{prefix}_prompt_stage_layer_hot_experts.csv",
        [
            "prompt_hash",
            "epoch",
            "stage_active_ranks",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_prompt_stage_layer_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        prompt_stage_layer_rows,
    )

    seq_rows = []
    seq_route_rows = record.get("epoch_seq_route_rows", {})
    seq_parent_prompt_rows = record.get("epoch_seq_parent_prompt", {})
    for epoch_str, seq_map in sorted(record.get("epoch_seq_expert_counts", {}).items(),
                                     key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_seq_rows = seq_route_rows.get(str(epoch), {})
        epoch_seq_parent = seq_parent_prompt_rows.get(str(epoch), {})
        for seq_hash, counts in sorted(seq_map.items()):
            count_list = [int(x) for x in counts]
            total_edges = int(sum(count_list))
            ranks = _dense_rank_from_counts(count_list)
            for expert_id, count in enumerate(count_list):
                seq_rows.append({
                    "prompt_hash": epoch_seq_parent.get(seq_hash, ""),
                    "seq_hash": seq_hash,
                    "epoch": epoch,
                    "expert_id": expert_id,
                    "edge_count": count,
                    "share_seq_edges": (float(count) / total_edges) if total_edges > 0 else 0.0,
                    "hot_rank": ranks[expert_id],
                    "route_rows": int(epoch_seq_rows.get(seq_hash, 0)),
                })
    _write_csv(
        out_dir / f"{prefix}_seq_epoch_hot_experts.csv",
        [
            "prompt_hash",
            "seq_hash",
            "epoch",
            "expert_id",
            "edge_count",
            "share_seq_edges",
            "hot_rank",
            "route_rows",
        ],
        seq_rows,
    )

    seq_layer_rows = []
    seq_layer_route_rows = record.get("epoch_seq_layer_route_rows", {})
    prompt_seq_layer_similarity_rows = []
    for epoch_str, seq_map in sorted(record.get("epoch_seq_layer_expert_counts", {}).items(),
                                     key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_seq_rows = seq_layer_route_rows.get(str(epoch), {})
        epoch_seq_parent = seq_parent_prompt_rows.get(str(epoch), {})
        layer_prompt_to_seq_counts: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(dict))
        for seq_hash, layer_map in sorted(seq_map.items()):
            prompt_hash = epoch_seq_parent.get(seq_hash, "")
            seq_route_row_map = epoch_seq_rows.get(seq_hash, {})
            for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                layer_idx = int(layer_str)
                count_list = [int(x) for x in counts]
                total_edges = int(sum(count_list))
                ranks = _dense_rank_from_counts(count_list)
                route_rows = int(seq_route_row_map.get(str(layer_idx), 0))
                layer_prompt_to_seq_counts[layer_idx][prompt_hash][seq_hash] = count_list
                for expert_id, count in enumerate(count_list):
                    seq_layer_rows.append({
                        "prompt_hash": prompt_hash,
                        "seq_hash": seq_hash,
                        "epoch": epoch,
                        "layer_idx": layer_idx,
                        "expert_id": expert_id,
                        "edge_count": count,
                        "share_seq_layer_edges": (
                            float(count) / total_edges) if total_edges > 0 else 0.0,
                        "selected": int(count > 0),
                        "hot_rank": ranks[expert_id],
                        "route_rows": route_rows,
                    })
        for layer_idx, prompt_map in layer_prompt_to_seq_counts.items():
            for prompt_hash, seq_count_map in prompt_map.items():
                seq_items = sorted(seq_count_map.items())
                for idx_a in range(len(seq_items)):
                    seq_hash_a, counts_a = seq_items[idx_a]
                    for idx_b in range(idx_a + 1, len(seq_items)):
                        seq_hash_b, counts_b = seq_items[idx_b]
                        top_expert_a = _top_expert_id_from_counts(counts_a)
                        top_expert_b = _top_expert_id_from_counts(counts_b)
                        prompt_seq_layer_similarity_rows.append({
                            "prompt_hash": prompt_hash,
                            "epoch": epoch,
                            "layer_idx": layer_idx,
                            "seq_hash_a": seq_hash_a,
                            "seq_hash_b": seq_hash_b,
                            "cosine_similarity": _cosine_similarity_from_counts(
                                counts_a, counts_b),
                            "jaccard_selected_experts": _jaccard_selected_from_counts(
                                counts_a, counts_b),
                            "top_expert_a": top_expert_a,
                            "top_expert_b": top_expert_b,
                            "top1_match": int(top_expert_a >= 0 and top_expert_a == top_expert_b),
                        })
    _write_csv(
        out_dir / f"{prefix}_seq_layer_hot_experts.csv",
        [
            "prompt_hash",
            "seq_hash",
            "epoch",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_seq_layer_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        seq_layer_rows,
    )
    _write_csv(
        out_dir / f"{prefix}_prompt_seq_layer_similarity.csv",
        [
            "prompt_hash",
            "epoch",
            "layer_idx",
            "seq_hash_a",
            "seq_hash_b",
            "cosine_similarity",
            "jaccard_selected_experts",
            "top_expert_a",
            "top_expert_b",
            "top1_match",
        ],
        prompt_seq_layer_similarity_rows,
    )

    seq_stage_layer_rows = []
    seq_stage_layer_route_rows = record.get("epoch_seq_stage_layer_route_rows", {})
    for epoch_str, stage_map in sorted(record.get("epoch_seq_stage_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_stage_rows = seq_stage_layer_route_rows.get(str(epoch), {})
        epoch_seq_parent = seq_parent_prompt_rows.get(str(epoch), {})
        for stage_str, seq_map in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_seq_rows = epoch_stage_rows.get(str(stage), {})
            for seq_hash, layer_map in sorted(seq_map.items()):
                prompt_hash = epoch_seq_parent.get(seq_hash, "")
                seq_route_row_map = stage_seq_rows.get(seq_hash, {})
                for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                    layer_idx = int(layer_str)
                    count_list = [int(x) for x in counts]
                    total_edges = int(sum(count_list))
                    ranks = _dense_rank_from_counts(count_list)
                    route_rows = int(seq_route_row_map.get(str(layer_idx), 0))
                    for expert_id, count in enumerate(count_list):
                        seq_stage_layer_rows.append({
                            "prompt_hash": prompt_hash,
                            "seq_hash": seq_hash,
                            "epoch": epoch,
                            "stage_active_ranks": stage,
                            "layer_idx": layer_idx,
                            "expert_id": expert_id,
                            "edge_count": count,
                            "share_seq_stage_layer_edges": (
                                float(count) / total_edges) if total_edges > 0 else 0.0,
                            "selected": int(count > 0),
                            "hot_rank": ranks[expert_id],
                            "route_rows": route_rows,
                        })
    _write_csv(
        out_dir / f"{prefix}_seq_stage_layer_hot_experts.csv",
        [
            "prompt_hash",
            "seq_hash",
            "epoch",
            "stage_active_ranks",
            "layer_idx",
            "expert_id",
            "edge_count",
            "share_seq_stage_layer_edges",
            "selected",
            "hot_rank",
            "route_rows",
        ],
        seq_stage_layer_rows,
    )

    logger.info(
        "MoE pattern CSV export done: prefix=%s output_dir=%s elapsed_ms=%.3f timing_rows=%s",
        prefix,
        output_dir,
        (time.perf_counter() - start_total) * 1000.0,
        len(timing_rows),
    )

    sample_rows = []
    prompt_history: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for epoch_str, prompt_map in sorted(record.get("epoch_prompt_expert_counts", {}).items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        for prompt_hash, counts in sorted(prompt_map.items()):
            count_list = [int(x) for x in counts]
            prompt_sum = int(sum(count_list))
            ranks = _dense_rank_from_counts(count_list)
            prev_epoch = max((e for e in prompt_history[prompt_hash].keys() if e < epoch), default=None)
            prev_counts = prompt_history[prompt_hash].get(prev_epoch) if prev_epoch is not None else None
            for expert_id, count in enumerate(count_list):
                prev_count = int(prev_counts[expert_id]) if prev_counts is not None else 0
                prev_share = (float(prev_count) / sum(prev_counts)) if prev_counts and sum(prev_counts) > 0 else 0.0
                share = (float(count) / prompt_sum) if prompt_sum > 0 else 0.0
                sample_rows.append({
                    "prompt_hash": prompt_hash,
                    "epoch": epoch,
                    "expert_id": expert_id,
                    "edge_count": count,
                    "share_sample_edges": share,
                    "hot_rank": ranks[expert_id],
                    "delta_count_from_prev_epoch": count - prev_count,
                    "delta_share_from_prev_epoch": share - prev_share,
                })
            prompt_history[prompt_hash][epoch] = count_list
    _write_csv(
        out_dir / f"{prefix}_sample_epoch_hot_experts.csv",
        [
            "prompt_hash",
            "epoch",
            "expert_id",
            "edge_count",
            "share_sample_edges",
            "hot_rank",
            "delta_count_from_prev_epoch",
            "delta_share_from_prev_epoch",
        ],
        sample_rows,
    )

    sample_stage_rows = []
    route_rows = record.get("epoch_prompt_stage_route_rows", {})
    for epoch_str, stage_map in sorted(record.get("epoch_prompt_stage_expert_counts", {}).items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        for stage_str, prompt_map in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_route_rows = route_rows.get(str(epoch), {}).get(str(stage), {})
            for prompt_hash, counts in sorted(prompt_map.items()):
                count_list = [int(x) for x in counts]
                sample_sum = int(sum(count_list))
                ranks = _dense_rank_from_counts(count_list)
                for expert_id, count in enumerate(count_list):
                    sample_stage_rows.append({
                        "prompt_hash": prompt_hash,
                        "epoch": epoch,
                        "stage_active_ranks": stage,
                        "expert_id": expert_id,
                        "edge_count": count,
                        "share_sample_stage_edges": (float(count) / sample_sum) if sample_sum > 0 else 0.0,
                        "hot_rank": ranks[expert_id],
                        "route_rows": int(stage_route_rows.get(prompt_hash, 0)),
                    })
    _write_csv(
        out_dir / f"{prefix}_sample_stage_hot_experts.csv",
        [
            "prompt_hash",
            "epoch",
            "stage_active_ranks",
            "expert_id",
            "edge_count",
            "share_sample_stage_edges",
            "hot_rank",
            "route_rows",
        ],
        sample_stage_rows,
    )

    overview_rows = []
    overall_route_rows = record.get("epoch_prompt_route_rows", {})
    prompt_stage_rows = record.get("epoch_prompt_stage_route_rows", {})
    for epoch_str, prompt_map in sorted(record.get("epoch_prompt_expert_counts", {}).items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        total_route_rows = overall_route_rows.get(str(epoch), {})
        epoch_stage_rows = prompt_stage_rows.get(str(epoch), {})
        for prompt_hash, counts in sorted(prompt_map.items()):
            count_list = [int(x) for x in counts]
            total_edges = int(sum(count_list))
            if total_edges > 0:
                top_expert_id = max(range(len(count_list)), key=lambda idx: (count_list[idx], -idx))
                top_expert_count = int(count_list[top_expert_id])
                top_expert_share = float(top_expert_count) / total_edges
            else:
                top_expert_id = -1
                top_expert_count = 0
                top_expert_share = 0.0
            row = {
                "prompt_hash": prompt_hash,
                "epoch": epoch,
                "route_rows": int(total_route_rows.get(prompt_hash, 0)),
                "edge_count": total_edges,
                "top_expert_id": top_expert_id,
                "top_expert_count": top_expert_count,
                "top_expert_share": top_expert_share,
            }
            for stage in record.get("long_tail_stages", []):
                row[f"stage_{stage}_route_rows"] = int(
                    epoch_stage_rows.get(str(stage), {}).get(prompt_hash, 0)
                )
            overview_rows.append(row)
    fieldnames = [
        "prompt_hash",
        "epoch",
        "route_rows",
        "edge_count",
        "top_expert_id",
        "top_expert_count",
        "top_expert_share",
    ] + [f"stage_{stage}_route_rows" for stage in record.get("long_tail_stages", [])]
    _write_csv(out_dir / f"{prefix}_sample_overview.csv", fieldnames, overview_rows)

    layer_overview_rows = []
    for epoch_str, layer_map in sorted(record.get("epoch_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_layer_rows = layer_route_rows.get(str(epoch), {})
        for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
            layer_idx = int(layer_str)
            count_list = [int(x) for x in counts]
            total_edges = int(sum(count_list))
            selected_expert_count = int(sum(1 for count in count_list if count > 0))
            never_selected_expert_count = int(num_experts - selected_expert_count)
            if total_edges > 0:
                top_expert_id = max(range(len(count_list)),
                                    key=lambda idx: (count_list[idx], -idx))
                top_expert_count = int(count_list[top_expert_id])
                top_expert_share = float(top_expert_count) / total_edges
            else:
                top_expert_id = -1
                top_expert_count = 0
                top_expert_share = 0.0
            layer_overview_rows.append({
                "epoch": epoch,
                "layer_idx": layer_idx,
                "route_rows": int(epoch_layer_rows.get(str(layer_idx), 0)),
                "edge_count": total_edges,
                "selected_expert_count": selected_expert_count,
                "selected_expert_ratio": float(selected_expert_count) / num_experts,
                "never_selected_expert_count": never_selected_expert_count,
                "never_selected_expert_ratio": float(never_selected_expert_count) / num_experts,
                "top_expert_id": top_expert_id,
                "top_expert_count": top_expert_count,
                "top_expert_share": top_expert_share,
            })
    _write_csv(
        out_dir / f"{prefix}_layer_overview.csv",
        [
            "epoch",
            "layer_idx",
            "route_rows",
            "edge_count",
            "selected_expert_count",
            "selected_expert_ratio",
            "never_selected_expert_count",
            "never_selected_expert_ratio",
            "top_expert_id",
            "top_expert_count",
            "top_expert_share",
        ],
        layer_overview_rows,
    )

    layer_stage_overview_rows = []
    for epoch_str, stage_map in sorted(record.get("epoch_stage_layer_expert_counts", {}).items(),
                                       key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        epoch_stage_route_rows = layer_stage_route_rows.get(str(epoch), {})
        for stage_str, layer_map in sorted(stage_map.items(), key=lambda x: int(x[0])):
            stage = int(stage_str)
            stage_layer_rows = epoch_stage_route_rows.get(str(stage), {})
            for layer_str, counts in sorted(layer_map.items(), key=lambda x: int(x[0])):
                layer_idx = int(layer_str)
                count_list = [int(x) for x in counts]
                total_edges = int(sum(count_list))
                selected_expert_count = int(sum(1 for count in count_list if count > 0))
                never_selected_expert_count = int(num_experts - selected_expert_count)
                if total_edges > 0:
                    top_expert_id = max(range(len(count_list)),
                                        key=lambda idx: (count_list[idx], -idx))
                    top_expert_count = int(count_list[top_expert_id])
                    top_expert_share = float(top_expert_count) / total_edges
                else:
                    top_expert_id = -1
                    top_expert_count = 0
                    top_expert_share = 0.0
                layer_stage_overview_rows.append({
                    "epoch": epoch,
                    "stage_active_ranks": stage,
                    "layer_idx": layer_idx,
                    "route_rows": int(stage_layer_rows.get(str(layer_idx), 0)),
                    "edge_count": total_edges,
                    "selected_expert_count": selected_expert_count,
                    "selected_expert_ratio": float(selected_expert_count) / num_experts,
                    "never_selected_expert_count": never_selected_expert_count,
                    "never_selected_expert_ratio": float(never_selected_expert_count) / num_experts,
                    "top_expert_id": top_expert_id,
                    "top_expert_count": top_expert_count,
                    "top_expert_share": top_expert_share,
                })
    _write_csv(
        out_dir / f"{prefix}_layer_stage_overview.csv",
        [
            "epoch",
            "stage_active_ranks",
            "layer_idx",
            "route_rows",
            "edge_count",
            "selected_expert_count",
            "selected_expert_ratio",
            "never_selected_expert_count",
            "never_selected_expert_ratio",
            "top_expert_id",
            "top_expert_count",
            "top_expert_share",
        ],
        layer_stage_overview_rows,
    )


# 全局实例
moe_stats = MoEStats()
