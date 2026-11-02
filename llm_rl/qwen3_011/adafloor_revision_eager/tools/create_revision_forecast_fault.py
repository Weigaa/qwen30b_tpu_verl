#!/usr/bin/env python3
"""Create a derived historical forecast, never edit the real epoch0 history."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.scale) or not 0 < args.scale <= 1:
        raise ValueError("fault scale must be in (0,1]")
    if args.output_dir.exists():
        raise ValueError("refusing existing derived history")
    original = args.history.read_bytes()
    payload = copy.deepcopy(json.loads(original))
    for record in payload["records"]:
        record["lengths"] = [max(1, math.ceil(v * args.scale)) for v in record["lengths"]]
    payload["revision_fault"] = {"kind": "globally_correlated_forecast_underestimation",
                                 "scale": args.scale, "source": str(args.history),
                                 "source_sha256": hashlib.sha256(original).hexdigest(),
                                 "note": "Only predictor input is perturbed; actual sampler hard cap is configured separately."}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "offline_planning_history.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    assert args.history.read_bytes() == original
    print(f"derived_history={args.output_dir} scale={args.scale}")


if __name__ == "__main__":
    main()
