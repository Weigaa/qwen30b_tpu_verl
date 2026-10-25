"""Profile-guided ACLGraph bucket selection for one rollout step."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence


MAX_LOCAL_BATCH_SIZE = 32


def predicted_global_occupancy(
    rank_lengths: Mapping[int, Sequence[float]],
    *,
    response_cap: int,
) -> Counter[int]:
    """Return DP-synchronous decode rounds by largest live local batch."""
    expected_ranks = list(range(16))
    if sorted(int(rank) for rank in rank_lengths) != expected_ranks:
        raise ValueError("ACLGraph bucket planning requires ranks 0..15")
    normalized: dict[int, list[int]] = {}
    for rank in expected_ranks:
        values = [min(int(value), int(response_cap)) for value in rank_lengths[rank]]
        if len(values) != MAX_LOCAL_BATCH_SIZE:
            raise ValueError(
                f"rank {rank} must own {MAX_LOCAL_BATCH_SIZE} predicted responses"
            )
        if any(value <= 0 for value in values):
            raise ValueError("predicted response lengths must be positive")
        normalized[rank] = values

    histogram: Counter[int] = Counter()
    max_length = max(max(values) for values in normalized.values())
    for token_index in range(max_length):
        global_batch = max(
            sum(length > token_index for length in values)
            for values in normalized.values()
        )
        histogram[int(global_batch)] += 1
    return histogram


def optimal_capture_sizes(
    histogram: Mapping[int, int], graph_count: int
) -> tuple[list[int], int]:
    """Choose an exact minimum-padding bucket set ending at batch size 32."""
    if not 1 <= int(graph_count) <= MAX_LOCAL_BATCH_SIZE:
        raise ValueError("graph_count must be in [1, 32]")
    weights = [0] + [
        int(histogram.get(size, 0)) for size in range(1, MAX_LOCAL_BATCH_SIZE + 1)
    ]
    prefix = [0]
    for weight in weights[1:]:
        prefix.append(prefix[-1] + weight)

    def segment_cost(previous: int, bucket: int) -> int:
        return bucket * (prefix[bucket] - prefix[previous])

    infinity = 10**30
    dp = [
        [infinity] * (MAX_LOCAL_BATCH_SIZE + 1)
        for _ in range(graph_count + 1)
    ]
    previous = [
        [-1] * (MAX_LOCAL_BATCH_SIZE + 1)
        for _ in range(graph_count + 1)
    ]
    dp[0][0] = 0
    for count in range(1, graph_count + 1):
        for bucket in range(1, MAX_LOCAL_BATCH_SIZE + 1):
            for prior in range(count - 1, bucket):
                candidate = dp[count - 1][prior] + segment_cost(prior, bucket)
                if candidate < dp[count][bucket]:
                    dp[count][bucket] = candidate
                    previous[count][bucket] = prior

    bucket = MAX_LOCAL_BATCH_SIZE
    buckets: list[int] = []
    for count in range(graph_count, 0, -1):
        buckets.append(bucket)
        bucket = previous[count][bucket]
    buckets.reverse()
    exact = sum(int(size) * int(rounds) for size, rounds in histogram.items())
    return buckets, int(dp[graph_count][MAX_LOCAL_BATCH_SIZE] - exact)


def rank_lengths_from_prompt_assignment(
    prompts_by_source: Mapping[int, Sequence[float]],
    rank_to_source_idx: Mapping[int | str, Iterable[int]],
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for rank in range(16):
        source_indices = list(
            rank_to_source_idx.get(rank, rank_to_source_idx.get(str(rank), []))
        )
        if len(source_indices) != 2:
            raise ValueError(f"rank {rank} must own exactly two prompts")
        lengths: list[float] = []
        for source_idx in source_indices:
            values = prompts_by_source.get(int(source_idx))
            if values is None:
                raise ValueError(f"missing predicted lengths for source {source_idx}")
            lengths.extend(float(value) for value in values)
        result[rank] = lengths
    return result
