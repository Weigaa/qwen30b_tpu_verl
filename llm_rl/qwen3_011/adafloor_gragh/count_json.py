import json

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def count_dict_keys_by_depth_from_list_top(obj: Any) -> Dict[int, Dict[str, int]]:
    """
    Assumed structure:
      - depth 1: list
      - depth 2: dict (each item in list)
      - depth 3+: dict (nested values)
    Returns stats per depth:
      stats[depth] = {
        "nodes": how many containers at this depth,
        "total_keys": sum of len(dict) across nodes (0 for non-dict),
        "min_keys": min len(dict) among nodes,
        "max_keys": max len(dict) among nodes,
      }
    """
    if not isinstance(obj, list):
        raise TypeError(f"Top-level must be list, got {type(obj).__name__}")

    # We'll treat "depth" as:
    # depth 1 = list itself
    # depth 2 = dicts inside the list
    # depth 3 = dicts inside values of depth 2 dicts, recursively, etc.
    stats = defaultdict(lambda: {"nodes": 0, "total_keys": 0, "min_keys": None, "max_keys": None})

    # depth 1
    stats[1]["nodes"] = 1
    stats[1]["total_keys"] = len(obj)
    stats[1]["min_keys"] = len(obj)
    stats[1]["max_keys"] = len(obj)

    def record_dict(depth: int, d: Dict[Any, Any]):
        k = len(d)
        s = stats[depth]
        s["nodes"] += 1
        s["total_keys"] += k
        s["min_keys"] = k if s["min_keys"] is None else min(s["min_keys"], k)
        s["max_keys"] = k if s["max_keys"] is None else max(s["max_keys"], k)

    def walk(depth: int, node: Any):
        # Only recurse into dict values (as per your "each level is dict" assumption from depth 3+)
        if isinstance(node, dict):
            record_dict(depth, node)
            for v in node.values():
                if isinstance(v, dict):
                    walk(depth + 1, v)
                # if occasionally lists appear deeper, you can extend here if needed

    # depth 2: each element in the list is expected to be dict
    for item in obj:
        if isinstance(item, dict):
            walk(2, item)
        else:
            # If you want to be strict, change to raise TypeError(...)
            pass

    return dict(stats)


def summarize_json_levels(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    stats = count_dict_keys_by_depth_from_list_top(obj)

    # Print key depths you asked for (2 and 3), plus deeper if exists
    for depth in sorted(stats.keys()):
        s = stats[depth]
        avg = (s["total_keys"] / s["nodes"]) if s["nodes"] else 0
        print(
            f"Depth {depth}: nodes={s['nodes']}, total_keys/items={s['total_keys']}, "
            f"min={s['min_keys']}, max={s['max_keys']}, avg={avg:.2f}"
        )


# usage
summarize_json_levels("moe_step_0.json")
