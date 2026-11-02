# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import vllm.v1.sample.rejection_sampler as rs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (RejectionSampler, compute_probs,
                                              generate_uniform_probs)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32


class AscendRejectionSampler(RejectionSampler, nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        if min(num_draft_tokens) == 1 and max(
                num_draft_tokens) == 1 and sampling_metadata.all_greedy:
            rejection_greedy_sample_spec_len_1_pytorch(
                output_token_ids,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
            )
        else:
            rejection_greedy_sample_pytorch(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                num_draft_tokens,
                max_spec_len,
                is_greedy,
            )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        IS_NGRAM=draft_probs is None,
        # num_warps=1,
    )
    return output_token_ids


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    if batch_size == 0:
        return x.new_empty(num_tokens)
    
    zero = torch.tensor([0], device=cu_num_tokens.device, dtype=cu_num_tokens.dtype)
    cu_num_tokens_with_zero = torch.cat((zero, cu_num_tokens))
    lengths = torch.diff((cu_num_tokens_with_zero))

    src_vals = x
    if replace_from != replace_to:
        src_vals = torch.where(x == replace_from, replace_to, x)
    
    expanded_x = torch.repeat_interleave(src_vals, repeats=lengths, dim=0)
    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_pytorch(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
        output_token_ids,  # [batch_size, 2]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[accept_req_mask, 1] = bonus_token_ids[accept_req_mask]


def rejection_greedy_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        max_spec_len,
        is_greedy=None,  # [batch_size] or None
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(
        num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = (draft_token_ids != target_argmax)
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size,
                                                 dtype=torch.long,
                                                 device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len),
                                     False,
                                     dtype=torch.bool,
                                     device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix,
                                         max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = (first_mismatch_pos_per_req == max_spec_len * 2)
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[
            no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1,
                             draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1,
                                device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[
        global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req
                               >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    """
    Performs batched rejection sampling for speculative decoding.

    For each sequence, determines the first rejection point k where:
        P_target(τ_i) / P_draft(τ_i) < U_i (U_i ~ Uniform(0,1))
    
    Tokens before k are accepted. Token k is replaced with recovered_token.
    Tokens after k are discarded. If no rejection occurs, appends bonus_token.
    """

    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    # 1. Reshape flat draft arrays to [batch, max_draft_len].
    
    # Calculate per-batch draft token ranges.
    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]
    valid_mask = pos_indices < num_draft_per_batch[:, None]

    # Map batch positions to global indices in the flattened input
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    
    draft_tokens = draft_token_ids[global_token_indices]

    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()

    # 2. Calculate P_draft and P_target for the Acceptance Check.

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    
    recovered_tokens = recovered_token_ids[global_token_indices]

    # 3. Determine Acceptance and the Rejection Point.

    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    # Acceptance criterion: P_target/P_draft ≥ U (with P_draft > 0 safeguard)
    first_acceptance = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )

    # The first index where rejection occurs.
    rejection_mask = (~first_acceptance) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        rejection_mask.any(dim=1, keepdim=True),
        rejection_mask.float().argmax(dim=1, keepdim=True),
        default_pos
    )
    
    # 4. Apply Masks and Update Output Tokens

    after_rejection = pos_indices >= first_reject_pos
    after_rejection = after_rejection & valid_mask
    
    final_acceptance = first_acceptance & (~after_rejection)
    
    non_greedy_seqs = ~is_greedy
    
    # The positions before the rejection point.
    accepted_positions = non_greedy_seqs[:, None] & valid_mask & (~after_rejection)

    # The rejection point. 
    rejection_point = (pos_indices == first_reject_pos) & valid_mask & non_greedy_seqs[:, None]
    
    final_update_positions = accepted_positions | rejection_point
    
    # Select the final tokens:
    #   - At k: recovered token.
    #   - Before k: accepted draft token.
    final_tokens = torch.where(
        rejection_point,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len])
    )

    #   Otherwise: keep the existing output.
    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_positions,
        final_tokens,
        output_token_ids[:, :max_draft_len]
    )

    # 5. Handle the Bonus Token

    # Insert bonus tokens only when no rejection occurs.
    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_seqs & no_rejection

    bonus_positions = num_draft_per_batch
    seq_len = output_token_ids.shape[1]
    
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]

    batch_bonus_positions = bonus_positions[:, None]

    # Boundary check for bonus tokens.
    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    bonus_pos_match = (all_positions == batch_bonus_positions)
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    # Insert the bonus tokens.
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)

def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    batch_size = len(input_ptr)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_tokens_ptr[req_idx - 1]
        end_idx = cu_num_tokens_ptr[req_idx]
        num_tokens = end_idx - start_idx

        src_val = input_ptr[req_idx]
        src_val = replace_to if src_val == replace_from else src_val

        offset = torch.arange(MAX_NUM_TOKENS, device=num_tokens.device)
        mask = offset < num_tokens

        output_slice = start_idx + offset[mask]
        output_ptr[output_slice] = src_val


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    device = target_probs.device
    num_tokens = target_probs.shape[0]
    batch_size = len(cu_num_draft_tokens)

    if IS_NGRAM:
        prob = target_probs.clone()
        token_indices = torch.arange(num_tokens, device=device)
        prob[token_indices, draft_token_ids] = 0.0
    else:
        prob = torch.relu(target_probs - draft_probs)
    
    index_device = cu_num_draft_tokens.device

    num_draft_tokens_per_req = torch.diff(
        cu_num_draft_tokens,
        prepend=torch.tensor([0], device=index_device)
    )

    request_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=index_device),
        num_draft_tokens_per_req
    ).to(device)

    q_values = q[request_indices]
    recovered_ids = torch.argmax(prob / q_values, dim=1)

    output_token_ids[:num_tokens] = recovered_ids


rs.expand_batch_to_tokens = expand_batch_to_tokens
