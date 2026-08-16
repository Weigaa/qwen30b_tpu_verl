#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DONOR_RANKS = list(range(0, 8))
WAVE_RANKS = list(range(8, 12))
FINAL_RANKS = list(range(12, 16))
ALL_RANKS = DONOR_RANKS + WAVE_RANKS + FINAL_RANKS
OFFLINE_PLANNING_HISTORY_FILENAME = "offline_planning_history.json"


@dataclass(frozen=True)
class PromptStats:
    source_idx: int
    mean: float
    p95: float
    max_len: float
    sum_len: float
    peak_active_tokens: float
    clip_count: int
    lengths: tuple[float, ...]
    predicted_tail: float | None = None

    @property
    def load(self) -> float:
        if self.predicted_tail is not None:
            return self.predicted_tail
        return self.max_len


@dataclass(frozen=True)
class BatchPlan:
    prompts: list[PromptStats]
    rank_to_prompt_indices: dict[int, list[int]]
    rank_loads: dict[int, float]
    rank_peak_loads: dict[int, float]
    rank_token_sums: dict[int, float]
    objective: float
    feasible: bool


def _peak_active_tokens(lengths: Iterable[float]) -> float:
    ordered = sorted(float(item) for item in lengths)
    if not ordered:
        return 0.0
    return float(max(length * (len(ordered) - idx)
                     for idx, length in enumerate(ordered)))


def _prompt_to_input(prompt: Any, tokenizer: Any | None = None) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if tokenizer is not None:
        token_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))
    if (isinstance(prompt, list) and len(prompt) == 1
            and isinstance(prompt[0], dict)):
        return f"user\n{prompt[0].get('content', '')}\nassistant\n"
    return str(prompt)


def _prompt_content_key(prompt_input: str) -> str:
    """Return prompt content independent of the supported chat wrappers."""

    value = str(prompt_input).replace("\r\n", "\n")
    wrappers = (
        ("User: ", "\n\nAssistant:"),
        ("user\n", "\nassistant\n"),
    )
    for prefix, suffix in wrappers:
        if value.startswith(prefix) and value.endswith(suffix):
            return value[len(prefix):-len(suffix)]
    return value


_JSON_DECODER = json.JSONDecoder()


def _input_from_rollout_line(line: str) -> str:
    """Decode only the leading input field from one rollout JSON record."""
    index = 0
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != "{":
        raise ValueError("rollout record is not a JSON object")
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    key, index = _JSON_DECODER.raw_decode(line, index)
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != ":":
        raise ValueError("rollout record has no value for its first field")
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    value, _ = _JSON_DECODER.raw_decode(line, index)
    if key != "input":
        value = json.loads(line)["input"]
    return str(value)


def _prompt_stats_from_lengths(values: list[float]) -> PromptStats:
    arr = np.asarray(values, dtype=np.float64)
    return PromptStats(
        source_idx=-1,
        mean=float(arr.mean()),
        p95=float(np.percentile(arr, 95)),
        max_len=float(arr.max()),
        sum_len=float(arr.sum()),
        peak_active_tokens=_peak_active_tokens(arr.tolist()),
        clip_count=int(np.sum(arr >= 16384)),
        lengths=tuple(float(item) for item in arr.tolist()),
    )


def _read_offline_planning_history(
    history_file: Path,
    steps: int,
    responses_per_prompt: int,
) -> dict[str, PromptStats]:
    payload = json.loads(history_file.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"unsupported history schema in {history_file}")
    history_steps = int(payload.get("steps", -1))
    if history_steps <= 0:
        raise RuntimeError(
            f"invalid history step count in {history_file}: {history_steps}")
    history_responses = int(payload.get("responses_per_prompt", -1))
    if history_responses < responses_per_prompt:
        raise RuntimeError(
            f"history has too few responses per prompt in {history_file}: "
            f"required={responses_per_prompt} actual={history_responses}"
        )

    stats: dict[str, PromptStats] = {}
    for record in payload.get("records", []):
        prompt_input = str(record["input"])
        values = [float(item) for item in record["lengths"]]
        if len(values) != history_responses:
            raise RuntimeError(
                f"expected {history_responses} historical lengths for "
                f"{prompt_input[:120]!r}, got {len(values)}"
            )
        values = values[:responses_per_prompt]
        if prompt_input in stats:
            raise RuntimeError(
                f"duplicate prompt in {history_file}: {prompt_input[:120]!r}"
            )
        stats[prompt_input] = _prompt_stats_from_lengths(values)
    if not stats:
        raise RuntimeError(f"offline planning history is empty: {history_file}")
    return stats


def _read_baseline_stats(baseline_dir: Path, steps: int,
                         responses_per_prompt: int) -> dict[str, PromptStats]:
    history_file = baseline_dir / OFFLINE_PLANNING_HISTORY_FILENAME
    if history_file.is_file():
        return _read_offline_planning_history(
            history_file, steps, responses_per_prompt
        )

    rollout_data = baseline_dir / "rollout_data"
    rollout_length = baseline_dir / "rollout_length"
    stats: dict[str, PromptStats] = {}
    for logical_step, step, data_file in _discover_rollout_data_files(
            rollout_data, steps):
        length_file = rollout_length / f"length_{step}.txt"
        if length_file.exists():
            lengths = [
                float(line.strip()) for line in length_file.read_text().splitlines()
                if line.strip()
            ]
            prompt_inputs: list[str] = []
            with data_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        prompt_inputs.append(_input_from_rollout_line(line))
        else:
            rows: list[dict[str, Any]] = []
            with data_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            lengths = [_response_length_from_rollout_row(row) for row in rows]
            prompt_inputs = [str(row["input"]) for row in rows]
        if len(prompt_inputs) != len(lengths):
            raise RuntimeError(
                f"step={step} row/length mismatch: rows={len(prompt_inputs)} "
                f"lengths={len(lengths)}")

        by_input: dict[str, list[float]] = defaultdict(list)
        for prompt_input, length in zip(prompt_inputs, lengths, strict=True):
            by_input[prompt_input].append(float(length))
        for prompt_input, values in by_input.items():
            if len(values) != responses_per_prompt:
                raise RuntimeError(
                    f"logical_step={logical_step} source_step={step} expected "
                    f"{responses_per_prompt} responses per "
                    f"prompt, got {len(values)} for {prompt_input[:120]!r}")
            if prompt_input in stats:
                raise RuntimeError(f"duplicate prompt in baseline: {prompt_input[:120]!r}")
            stats[prompt_input] = _prompt_stats_from_lengths(values)
    return stats


def _discover_rollout_data_files(
        rollout_data: Path, steps: int) -> list[tuple[int, int, Path]]:
    exact = [(idx, idx, rollout_data / f"{idx}.jsonl")
             for idx in range(1, steps + 1)]
    if all(path.exists() for _, _, path in exact):
        return exact

    candidates = [
        path for path in rollout_data.glob("*.jsonl")
        if path.stem.isdigit()
    ]
    candidates.sort(key=lambda path: int(path.stem))
    if len(candidates) < steps:
        raise FileNotFoundError(
            f"expected rollout_data 1..{steps} under {rollout_data}, "
            f"or at least {steps} numeric jsonl files; found {len(candidates)}")
    return [(logical_idx, int(path.stem), path)
            for logical_idx, path in enumerate(candidates[:steps], start=1)]


def _response_length_from_rollout_row(row: dict[str, Any]) -> float:
    response_mask = row.get("response_mask")
    if response_mask is not None:
        return float(sum(float(item) for item in response_mask))

    responses = row.get("responses")
    if responses is not None:
        if isinstance(responses, list):
            return float(len(responses))
        return float(len(str(responses)))

    output = row.get("output")
    if output is not None:
        return float(len(str(output).split()))

    raise RuntimeError(
        "rollout row has no response_mask/responses/output field to infer length")


def _map_stats_to_dataset(train_file: Path,
                          stats_by_input: dict[str, PromptStats],
                          dataset_fraction: float,
                          max_samples: int,
                          min_samples: int,
                          repeat_prompt_set_to_fill: bool = False,
                          prompt_tokenizer: Any | None = None,
                          ) -> tuple[pd.DataFrame, list[PromptStats]]:
    df = pd.read_parquet(train_file)
    sample_size = int(len(df) * dataset_fraction)
    if max_samples > 0:
        sample_size = min(sample_size, max_samples)
    sample_size = max(sample_size, min_samples)
    # Performance runs consume exactly ``min_samples`` rows from the
    # unshuffled dataset.  Restrict matching to that execution prefix so a
    # repeated prompt later in the fraction cannot be mistaken for another
    # occurrence in the measured epoch.  Stress tests intentionally scan the
    # full sampled set before repeating the matched prompts.
    selected_size = sample_size if repeat_prompt_set_to_fill else min_samples
    selected = df.iloc[:selected_size].copy()

    stats_by_content: dict[str, PromptStats] = {}
    for prompt_input, stat in stats_by_input.items():
        key = _prompt_content_key(prompt_input)
        previous = stats_by_content.get(key)
        if previous is not None and previous is not stat:
            raise RuntimeError(
                "multiple history prompts collapse to the same content key: "
                f"{key[:200]!r}")
        stats_by_content[key] = stat

    rows: list[tuple[int, PromptStats]] = []
    missing: list[str] = []
    for source_idx, row in selected.iterrows():
        prompt_input = _prompt_to_input(
            row["prompt"], tokenizer=prompt_tokenizer)
        stat = stats_by_input.get(prompt_input)
        if stat is None:
            stat = stats_by_content.get(_prompt_content_key(prompt_input))
        if stat is None:
            missing.append(prompt_input)
            continue
        rows.append((int(source_idx), PromptStats(
            source_idx=int(source_idx),
            mean=stat.mean,
            p95=stat.p95,
            max_len=stat.max_len,
            sum_len=stat.sum_len,
            peak_active_tokens=stat.peak_active_tokens,
            clip_count=stat.clip_count,
            lengths=stat.lengths,
            predicted_tail=stat.predicted_tail,
        )))
    if repeat_prompt_set_to_fill and rows and len(rows) < min_samples:
        repeated_source_indices = [
            rows[idx % len(rows)][0] for idx in range(min_samples)
        ]
        repeated_df = df.loc[repeated_source_indices].reset_index(drop=True)
        repeated_stats: list[PromptStats] = []
        for new_source_idx in range(min_samples):
            original = rows[new_source_idx % len(rows)][1]
            repeated_stats.append(PromptStats(
                source_idx=new_source_idx,
                mean=original.mean,
                p95=original.p95,
                max_len=original.max_len,
                sum_len=original.sum_len,
                peak_active_tokens=original.peak_active_tokens,
                clip_count=original.clip_count,
                lengths=original.lengths,
                predicted_tail=original.predicted_tail,
            ))
        return repeated_df, repeated_stats
    if len(rows) != min_samples:
        preview = missing[0][:200] if missing else ""
        raise RuntimeError(
            f"expected {min_samples} mapped prompts, got {len(rows)}; "
            f"first_missing={preview!r}")
    return df, [stat for _, stat in rows]


def _assign_pairs_to_ranks(
    group: list[PromptStats],
    ranks: list[int],
    *,
    max_rank_peak_tokens: float,
) -> tuple[dict[int, list[int]], dict[int, float], dict[int, float], dict[int, float], float]:
    if len(group) != len(ranks) * 2:
        raise ValueError(f"group size mismatch: {len(group)} prompts for {len(ranks)} ranks")
    ordered = sorted(group, key=lambda item: (item.load, item.source_idx))
    pair_metrics: dict[tuple[int, int], tuple[float, float, float, list[int]]] = {}
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            left = ordered[first]
            right = ordered[second]
            response_lengths = left.lengths + right.lengths
            pair_metrics[(first, second)] = (
                _peak_active_tokens(response_lengths),
                float(left.sum_len + right.sum_len),
                float(max(left.load, right.load)),
                [int(left.source_idx), int(right.source_idx)],
            )

    if len(ranks) <= 4:
        best_pairs: list[tuple[float, float, float, list[int]]] | None = None
        best_score: tuple[float, float, float, float, float] | None = None
        for pairing in _all_pairings(tuple(range(len(ordered)))):
            candidate = [
                pair_metrics[(min(first, second), max(first, second))]
                for first, second in pairing
            ]
            score = _pairing_score(
                candidate, max_rank_peak_tokens=max_rank_peak_tokens)
            if best_score is None or score < best_score:
                best_score = score
                best_pairs = candidate
    else:
        best_pairs = _minimax_peak_pairs(
            len(ordered), pair_metrics, max_rank_peak_tokens=max_rank_peak_tokens)
    if best_pairs is None:
        raise RuntimeError("failed to build exact rank pairs")
    pairs = best_pairs
    pairs.sort(key=lambda item: (item[2], item[0], item[1]))

    mapping: dict[int, list[int]] = {}
    loads: dict[int, float] = {}
    peak_loads: dict[int, float] = {}
    token_sums: dict[int, float] = {}
    for rank, (rank_peak, rank_sum, max_len, source_indices) in zip(
            ranks, pairs, strict=True):
        mapping[int(rank)] = source_indices
        loads[int(rank)] = float(max_len)
        peak_loads[int(rank)] = float(rank_peak)
        token_sums[int(rank)] = float(rank_sum)
    vals = list(loads.values())
    peak_vals = list(peak_loads.values())
    objective_spread = max(vals) - min(vals)
    return mapping, loads, peak_loads, token_sums, float(objective_spread)


def _pairing_score(
    pairs: list[tuple[float, float, float, list[int]]],
    *,
    max_rank_peak_tokens: float,
) -> tuple[float, float, float, float, float]:
    peaks = [item[0] for item in pairs]
    sums = [item[1] for item in pairs]
    max_lens = [item[2] for item in pairs]
    overflow = max(0.0, max(peaks) - max_rank_peak_tokens)
    return (
        float(overflow > 0.0),
        float(overflow),
        float(max(max_lens) - min(max_lens)),
        float(max(peaks)),
        float(max(sums) - min(sums)),
    )


def _minimax_peak_pairs(
    item_count: int,
    pair_metrics: dict[tuple[int, int], tuple[float, float, float, list[int]]],
    *,
    max_rank_peak_tokens: float,
) -> list[tuple[float, float, float, list[int]]]:
    unique_peaks = sorted({item[0] for item in pair_metrics.values()})
    feasible_peaks = [peak for peak in unique_peaks if peak <= max_rank_peak_tokens]
    candidates = feasible_peaks or unique_peaks

    best_matching: list[tuple[int, int]] | None = None
    lo = 0
    hi = len(candidates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        matching = _find_matching_with_peak_limit(
            item_count, pair_metrics, candidates[mid])
        if matching is not None:
            best_matching = matching
            hi = mid - 1
        else:
            lo = mid + 1
    if best_matching is None:
        raise RuntimeError("failed to find minimax active-peak matching")

    # Keep the minimax peak guarantee, then choose a low-spread matching among
    # pairs under the selected threshold.
    threshold = max(pair_metrics[pair][0] for pair in best_matching)
    spread_matching = _find_matching_with_peak_limit(
        item_count, pair_metrics, threshold, minimize_load_spread=True)
    if spread_matching is not None:
        best_matching = spread_matching
    return [pair_metrics[pair] for pair in best_matching]


def _find_matching_with_peak_limit(
    item_count: int,
    pair_metrics: dict[tuple[int, int], tuple[float, float, float, list[int]]],
    peak_limit: float,
    *,
    minimize_load_spread: bool = False,
) -> list[tuple[int, int]] | None:
    candidate_pairs = {
        pair for pair, metrics in pair_metrics.items()
        if metrics[0] <= peak_limit
    }
    best_matching: list[tuple[int, int]] | None = None
    best_score: tuple[float, float, float] | None = None

    def search(remaining: tuple[int, ...],
               chosen: list[tuple[int, int]]) -> bool:
        nonlocal best_matching, best_score
        if not remaining:
            if not minimize_load_spread:
                best_matching = list(chosen)
                return True
            loads = [pair_metrics[pair][2] for pair in chosen]
            sums = [pair_metrics[pair][1] for pair in chosen]
            peaks = [pair_metrics[pair][0] for pair in chosen]
            score = (
                float(max(loads) - min(loads)),
                float(max(sums) - min(sums)),
                float(max(peaks) - min(peaks)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_matching = list(chosen)
            return False

        first = remaining[0]
        rest = remaining[1:]
        for pos, second in enumerate(rest):
            pair = (min(first, second), max(first, second))
            if pair not in candidate_pairs:
                continue
            next_remaining = rest[:pos] + rest[pos + 1:]
            if search(next_remaining, chosen + [pair]) and not minimize_load_spread:
                return True
        return False

    search(tuple(range(item_count)), [])
    return best_matching


def _score_plan(loads: dict[int, float], peak_loads: dict[int, float],
                group_spreads: Iterable[float],
                *, max_rank_peak_tokens: float) -> float:
    donor_max = max(loads[r] for r in DONOR_RANKS)
    final_max = max(loads[r] for r in FINAL_RANKS)
    wave_max = max(loads[r] for r in WAVE_RANKS)
    peak_max = max(peak_loads.values())
    order_penalty = 0.0
    if donor_max >= min(loads[r] for r in FINAL_RANKS + WAVE_RANKS):
        order_penalty += 1_000_000.0 + donor_max
    if wave_max >= final_max:
        order_penalty += 1_000_000.0 + wave_max
    if peak_max > max_rank_peak_tokens:
        order_penalty += 1_000_000.0 + 10.0 * (peak_max - max_rank_peak_tokens)
    return float(sum(group_spreads) + order_penalty)


def _solve_one_batch(donor_items: list[PromptStats],
                     wave_items: list[PromptStats],
                     final_items: list[PromptStats],
                     *, max_rank_peak_tokens: float) -> BatchPlan:
    if len(donor_items) != 16 or len(wave_items) != 8 or len(final_items) != 8:
        raise ValueError(
            "one optimized step requires 16 donor, 8 wave, 8 final prompts; "
            f"got donor={len(donor_items)} wave={len(wave_items)} "
            f"final={len(final_items)}")
    rank_map: dict[int, list[int]] = {}
    loads: dict[int, float] = {}
    peak_loads: dict[int, float] = {}
    token_sums: dict[int, float] = {}
    spreads: list[float] = []
    for group, ranks in (
        (donor_items, DONOR_RANKS),
        (final_items, FINAL_RANKS),
        (wave_items, WAVE_RANKS),
    ):
        mapping, group_loads, group_peaks, group_sums, spread = (
            _assign_pairs_to_ranks(
                group, ranks, max_rank_peak_tokens=max_rank_peak_tokens))
        rank_map.update(mapping)
        loads.update(group_loads)
        peak_loads.update(group_peaks)
        token_sums.update(group_sums)
        spreads.append(spread)
    objective = _score_plan(
        loads, peak_loads, spreads,
        max_rank_peak_tokens=max_rank_peak_tokens)
    return BatchPlan(
        prompts=list(donor_items) + list(wave_items) + list(final_items),
        rank_to_prompt_indices=rank_map,
        rank_loads=loads,
        rank_peak_loads=peak_loads,
        rank_token_sums=token_sums,
        objective=objective,
        feasible=objective < 1_000_000.0,
    )


def _solve_batches(stats: list[PromptStats], batch_size: int,
                   steps: int, *, max_rank_peak_tokens: float) -> list[BatchPlan]:
    if len(stats) != batch_size * steps:
        raise ValueError(
            f"expected {batch_size * steps} prompt stats, got {len(stats)}")
    ordered = sorted(stats, key=lambda item: (item.load, item.source_idx))
    donor = ordered[:steps * 16]
    wave = ordered[steps * 16:steps * 24]
    final = ordered[steps * 24:]
    if len(donor) != steps * 16 or len(wave) != steps * 8 \
            or len(final) != steps * 8:
        raise RuntimeError(
            f"unexpected bucket sizes: donor={len(donor)} wave={len(wave)} "
            f"final={len(final)}")

    donor_batches: list[list[PromptStats]] = [[] for _ in range(steps)]
    wave_batches: list[list[PromptStats]] = [[] for _ in range(steps)]
    final_batches: list[list[PromptStats]] = [[] for _ in range(steps)]
    for step_idx in range(steps):
        donor_start = step_idx * 16
        wave_start = step_idx * 8
        donor_batches[step_idx] = donor[donor_start:donor_start + 16]
        wave_batches[step_idx] = wave[wave_start:wave_start + 8]

    final_order = sorted(
        final,
        key=lambda item: (item.load, item.clip_count, item.sum_len,
                          item.peak_active_tokens, item.source_idx),
        reverse=True,
    )
    for idx, item in enumerate(final_order):
        final_batches[idx % steps].append(item)

    plans = []
    for step_idx in range(steps):
        if len(donor_batches[step_idx]) + len(wave_batches[step_idx]) \
                + len(final_batches[step_idx]) != batch_size:
            raise RuntimeError(
                f"step={step_idx + 1} has invalid prompt count")
        plans.append(_solve_one_batch(
            donor_batches[step_idx],
            wave_batches[step_idx],
            final_batches[step_idx],
            max_rank_peak_tokens=max_rank_peak_tokens))
    return plans


def _all_pairings(items: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    if not items:
        yield ()
        return
    first = items[0]
    rest = items[1:]
    for pos, second in enumerate(rest):
        remaining = rest[:pos] + rest[pos + 1:]
        for tail in _all_pairings(remaining):
            yield ((first, second),) + tail


def _write_outputs(
    full_df: pd.DataFrame,
    plans: list[BatchPlan],
    output_train: Path,
    output_plan: Path,
    output_summary: Path,
    output_oracle: Path,
) -> None:
    if not isinstance(full_df.index, pd.RangeIndex) or list(full_df.index) != list(
        range(len(full_df))
    ):
        raise ValueError(
            "planner source_idx requires a zero-based contiguous dataset RangeIndex"
        )
    ordered_source_indices: list[int] = []
    plan_payload: list[dict[str, Any]] = []
    row_map: dict[int, int] = {}
    row_loads: dict[str, float] = {}
    next_row = 0
    for step_idx, plan in enumerate(plans, start=1):
        load_by_source = {
            int(item.source_idx): float(item.load)
            for item in plan.prompts
        }
        for rank in ALL_RANKS:
            for source_idx in plan.rank_to_prompt_indices[rank]:
                ordered_source_indices.append(int(source_idx))
                row_map[int(source_idx)] = next_row
                row_loads[str(next_row)] = load_by_source[int(source_idx)]
                next_row += 1
        plan_payload.append({
            "step": step_idx,
            "rank_to_dataset_item_idx": {
                str(rank): [
                    row_map[int(source_idx)]
                    for source_idx in plan.rank_to_prompt_indices[rank]
                ]
                for rank in ALL_RANKS
            },
            "rank_to_source_idx": {
                str(rank): [
                    int(source_idx)
                    for source_idx in plan.rank_to_prompt_indices[rank]
                ]
                for rank in ALL_RANKS
            },
            "rank_loads": {
                str(rank): float(plan.rank_loads[rank])
                for rank in ALL_RANKS
            },
            "rank_peak_loads": {
                str(rank): float(plan.rank_peak_loads[rank])
                for rank in ALL_RANKS
            },
            "rank_token_sums": {
                str(rank): float(plan.rank_token_sums[rank])
                for rank in ALL_RANKS
            },
            "objective": float(plan.objective),
            "feasible": bool(plan.feasible),
        })

    out_df = full_df.iloc[ordered_source_indices].copy()
    extra_info_values: list[dict[str, Any]] = []
    if "extra_info" in out_df.columns:
        raw_extra_info = out_df["extra_info"].tolist()
    else:
        raw_extra_info = [None] * len(out_df)
    for source_idx, extra_info in zip(
        ordered_source_indices, raw_extra_info, strict=True
    ):
        if extra_info is None:
            preserved: dict[str, Any] = {}
        elif isinstance(extra_info, dict):
            preserved = dict(extra_info)
        else:
            raise ValueError(
                "dataset extra_info must be a mapping or null, got "
                f"{type(extra_info).__name__} for source_idx={source_idx}"
            )
        for identity_field in ("index", "prompt_occurrence_ordinal"):
            if identity_field not in preserved:
                continue
            existing_identity = preserved[identity_field]
            if (
                isinstance(existing_identity, bool)
                or not isinstance(existing_identity, (int, np.integer))
                or int(existing_identity) != int(source_idx)
            ):
                raise ValueError(
                    f"dataset extra_info.{identity_field} already carries a "
                    "conflicting identity for "
                    f"source_idx={source_idx}: {existing_identity!r}"
                )
        preserved["index"] = int(source_idx)
        preserved["prompt_occurrence_ordinal"] = int(source_idx)
        extra_info_values.append(preserved)
    out_df["extra_info"] = extra_info_values
    output_train.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_train, index=False)
    output_plan.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    output_oracle.write_text(
        json.dumps(row_loads, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    summary: list[dict[str, Any]] = []
    for step_idx, plan in enumerate(plans, start=1):
        donor = [plan.rank_loads[r] for r in DONOR_RANKS]
        final = [plan.rank_loads[r] for r in FINAL_RANKS]
        wave = [plan.rank_loads[r] for r in WAVE_RANKS]
        donor_peak = [plan.rank_peak_loads[r] for r in DONOR_RANKS]
        final_peak = [plan.rank_peak_loads[r] for r in FINAL_RANKS]
        wave_peak = [plan.rank_peak_loads[r] for r in WAVE_RANKS]
        summary.append({
            "step": step_idx,
            "feasible": plan.feasible,
            "objective": plan.objective,
            "donor_range": [min(donor), max(donor)],
            "wave_range": [min(wave), max(wave)],
            "final_range": [min(final), max(final)],
            "donor_peak_range": [min(donor_peak), max(donor_peak)],
            "wave_peak_range": [min(wave_peak), max(wave_peak)],
            "final_peak_range": [min(final_peak), max(final_peak)],
            "max_rank_peak_tokens": max(donor_peak + wave_peak + final_peak),
            "expect_16_to_8": max(donor) < min(wave + final),
            "expect_8_to_4": max(wave) < max(final),
        })
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[mode1 optimized rank plan] train={output_train}")
    print(f"[mode1 optimized rank plan] plan={output_plan}")
    print(f"[mode1 optimized rank plan] oracle={output_oracle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-oracle", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-rank-peak-tokens", type=float, default=262144.0)
    args = parser.parse_args()

    stats_by_input = _read_baseline_stats(
        Path(args.baseline_dir), args.steps, args.responses_per_prompt)
    full_df, stats = _map_stats_to_dataset(
        Path(args.train_file),
        stats_by_input,
        args.dataset_fraction,
        args.max_samples,
        args.batch_size * args.steps,
    )
    plans = _solve_batches(
        stats,
        args.batch_size,
        args.steps,
        max_rank_peak_tokens=args.max_rank_peak_tokens,
    )
    _write_outputs(
        full_df,
        plans,
        Path(args.output_train),
        Path(args.output_plan),
        Path(args.output_summary),
        Path(args.output_oracle),
    )


if __name__ == "__main__":
    main()
