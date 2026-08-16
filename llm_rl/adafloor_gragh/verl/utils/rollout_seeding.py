"""Stable per-request seeds for matched rollout experiments."""

from __future__ import annotations

import hashlib
import numbers
import struct
from collections.abc import Iterable

import numpy as np


_UINT64_MASK = (1 << 64) - 1
_MAX_VLLM_SEED = (1 << 31) - 1


def build_rollout_sample_indices(
    prompt_count: int,
    responses_per_prompt: int,
    *,
    interleave: bool,
) -> np.ndarray:
    """Return each repeated row's response index within its source prompt."""
    if prompt_count < 0:
        raise ValueError(f"prompt_count must be nonnegative, got {prompt_count}")
    if responses_per_prompt <= 0:
        raise ValueError(
            "responses_per_prompt must be positive, "
            f"got {responses_per_prompt}"
        )
    response_indices = np.arange(responses_per_prompt, dtype=np.int64)
    if interleave:
        return np.tile(response_indices, prompt_count)
    return np.repeat(response_indices, prompt_count)


def validate_prompt_occurrence_ordinals(
    repeated_indices: Iterable[int],
    responses_per_prompt: int,
) -> np.ndarray:
    """Validate stable source occurrences after rollout-n repetition."""
    if responses_per_prompt <= 0:
        raise ValueError(
            "responses_per_prompt must be positive, "
            f"got {responses_per_prompt}"
        )
    values: list[int] = []
    for position, raw_value in enumerate(repeated_indices):
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Integral):
            raise ValueError(
                "prompt occurrence ordinals must be integers, got "
                f"{raw_value!r} at position {position}"
            )
        value = int(raw_value)
        if value < 0:
            raise ValueError(
                "prompt occurrence ordinals must be nonnegative, got "
                f"{value} at position {position}"
            )
        values.append(value)
    if not values or len(values) % responses_per_prompt != 0:
        raise ValueError(
            "repeated prompt occurrences do not match rollout.n, got "
            f"rows={len(values)} rollout_n={responses_per_prompt}"
        )
    prompt_count = len(values) // responses_per_prompt
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    invalid = {
        occurrence: count
        for occurrence, count in counts.items()
        if count != responses_per_prompt
    }
    if len(counts) != prompt_count or invalid:
        raise ValueError(
            "each source prompt occurrence must appear exactly rollout.n times, "
            f"expected_occurrences={prompt_count} observed={len(counts)} "
            f"invalid_counts={dict(list(sorted(invalid.items()))[:8])}"
        )
    return np.asarray(values, dtype=np.int64)


def validate_fixed_work_restored_order(
    expected_occurrences: Iterable[int],
    expected_sample_indices: Iterable[int],
    observed_occurrences: Iterable[int],
    observed_sample_indices: Iterable[int],
    observed_row_ordinals: Iterable[int],
) -> None:
    """Fail if rollout output was not restored to its pre-schedule row order."""
    expected_occurrences_array = np.asarray(
        list(expected_occurrences), dtype=np.int64
    )
    expected_samples_array = np.asarray(
        list(expected_sample_indices), dtype=np.int64
    )
    observed_occurrences_array = np.asarray(
        list(observed_occurrences), dtype=np.int64
    )
    observed_samples_array = np.asarray(
        list(observed_sample_indices), dtype=np.int64
    )
    observed_rows_array = np.asarray(list(observed_row_ordinals), dtype=np.int64)
    row_count = len(expected_occurrences_array)
    if len(expected_samples_array) != row_count:
        raise ValueError("fixed-work pre-schedule identity snapshot is inconsistent")
    expected_rows_array = np.arange(row_count, dtype=np.int64)
    if not (
        np.array_equal(observed_occurrences_array, expected_occurrences_array)
        and np.array_equal(observed_samples_array, expected_samples_array)
        and np.array_equal(observed_rows_array, expected_rows_array)
    ):
        raise ValueError(
            "fixed-work rollout output was not restored to its pre-schedule "
            "occurrence, sample, and physical row order"
        )


def prompt_token_hash(prompt_token_ids: Iterable[int]) -> str:
    """Hash unpadded prompt token IDs independently of row and rank order."""
    digest = hashlib.blake2b(digest_size=16, person=b"adafloor.prompt")
    count = 0
    for token_id in prompt_token_ids:
        digest.update(struct.pack("<q", int(token_id)))
        count += 1
    digest.update(struct.pack("<Q", count))
    return digest.hexdigest()


def derive_request_seed(
    base_seed: int,
    prompt_token_ids: Iterable[int],
    rollout_sample_index: int,
) -> tuple[str, int]:
    """Derive a vLLM seed from trial seed, prompt identity, and sample index."""
    if rollout_sample_index < 0:
        raise ValueError(
            "rollout_sample_index must be nonnegative, "
            f"got {rollout_sample_index}"
        )
    prompt_hash = prompt_token_hash(prompt_token_ids)
    digest = hashlib.blake2b(digest_size=8, person=b"adafloor.seed")
    digest.update(struct.pack("<Q", int(base_seed) & _UINT64_MASK))
    digest.update(bytes.fromhex(prompt_hash))
    digest.update(struct.pack("<Q", int(rollout_sample_index)))
    seed = int.from_bytes(digest.digest(), byteorder="little") % _MAX_VLLM_SEED
    return prompt_hash, seed
