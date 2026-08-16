"""Small model-config helpers shared by rollout launch paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUTED_EXPERT_COUNT_KEYS = (
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
)


def get_routed_expert_count(config: Mapping[str, Any] | Any) -> int:
    """Return the global routed-expert count used by common MoE configs.

    DeepSeek names this field ``n_routed_experts``. Qwen uses
    ``num_experts`` and Mixtral-style configs may use
    ``num_local_experts`` for the global logical expert count.
    """

    for key in ROUTED_EXPERT_COUNT_KEYS:
        if isinstance(config, Mapping):
            value = config.get(key)
        else:
            value = getattr(config, key, None)
        if isinstance(value, bool):
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 0
