"""Stable request seeds, identical to the v0.11 paired-rollout algorithm."""
import hashlib
import struct

import numpy as np


def build_rollout_sample_indices(prompt_count, responses_per_prompt, *, interleave):
    if prompt_count < 0 or responses_per_prompt <= 0:
        raise ValueError("Invalid rollout repetition dimensions")
    indices = np.arange(responses_per_prompt, dtype=np.int64)
    return np.tile(indices, prompt_count) if interleave else np.repeat(indices, prompt_count)


def prompt_token_hash(prompt_token_ids):
    digest = hashlib.blake2b(digest_size=16, person=b"adafloor.prompt")
    count = 0
    for token_id in prompt_token_ids:
        digest.update(struct.pack("<q", int(token_id)))
        count += 1
    digest.update(struct.pack("<Q", count))
    return digest.hexdigest()


def derive_request_seed(base_seed, prompt_token_ids, rollout_sample_index):
    if rollout_sample_index < 0:
        raise ValueError("rollout_sample_index must be nonnegative")
    prompt_hash = prompt_token_hash(prompt_token_ids)
    digest = hashlib.blake2b(digest_size=8, person=b"adafloor.seed")
    digest.update(struct.pack("<Q", int(base_seed) & ((1 << 64) - 1)))
    digest.update(bytes.fromhex(prompt_hash))
    digest.update(struct.pack("<Q", int(rollout_sample_index)))
    return prompt_hash, int.from_bytes(digest.digest(), byteorder="little") % ((1 << 31) - 1)
