"""Stable per-request seeds for matched rollout experiments."""

from __future__ import annotations

import hashlib
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
