#!/usr/bin/env python3
"""Create read-only derived forecast histories for the Natural-F2 stress test."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _factor(name: str, prompt: str, mixed_seed: int) -> float:
    if name == "under_0p7":
        return 0.7
    if name == "over_1p3":
        return 1.3
    if name == "mixed":
        digest = hashlib.sha256(f"{mixed_seed}:{prompt}".encode()).digest()
        return 0.7 if digest[0] % 2 == 0 else 1.3
    raise ValueError(f"unknown scenario: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mixed-seed", type=int, default=505)
    ap.add_argument("--regular-artifacts", action="store_true",
                    help="Copy once and hardlink regular artifacts for the original driver's find-based validation")
    args = ap.parse_args()
    if args.output_root.exists():
        raise ValueError(f"refusing existing scenario root: {args.output_root}")
    if not args.history.is_file():
        raise FileNotFoundError(args.history)
    original_bytes = args.history.read_bytes()
    original = json.loads(original_bytes)
    if original.get("schema_version") != 1 or original.get("responses_per_prompt") != 16:
        raise ValueError("expected compact schema1 history with 16 responses/prompt")
    if len(original.get("records", [])) != 160:
        raise ValueError("expected 160 prompt records")
    args.output_root.mkdir(parents=True)
    manifest = {"source": str(args.history), "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "mixed_seed": args.mixed_seed, "scenarios": {}}
    for scenario in ("under_0p7", "over_1p3", "mixed"):
        out = args.output_root / scenario
        out.mkdir()
        payload = copy.deepcopy(original)
        factors = {}
        for record in payload["records"]:
            factor = _factor(scenario, str(record["input"]), args.mixed_seed)
            factors[hashlib.sha256(str(record["input"]).encode()).hexdigest()] = factor
            record["lengths"] = [max(1, int(math.ceil(float(x) * factor))) for x in record["lengths"]]
        payload["revision_forecast_scenario"] = {
            "name": scenario,
            "mixed_seed": args.mixed_seed,
            "factors_by_prompt_sha256": factors,
            "actual_sampler_unchanged": True,
            "source_history_sha256": hashlib.sha256(original_bytes).hexdigest(),
        }
        (out / "offline_planning_history.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        # Planner/runtime still needs the original response files and length
        # files to map prompt identities. They are read-only symlinks.
        for name in ("rollout_data", "rollout_length"):
            source = args.history.parent / name
            if not source.exists():
                raise FileNotFoundError(source)
            if args.regular_artifacts:
                (out / name).mkdir()
                shared = args.output_root / "_shared_artifacts" / name
                shared.mkdir(parents=True, exist_ok=True)
                for artifact in sorted(source.iterdir()):
                    if not artifact.is_file():
                        continue
                    target = shared / artifact.name
                    if not target.exists():
                        shutil.copyfile(artifact, target)
                    os.link(target, out / name / artifact.name)
            else:
                os.symlink(source, out / name)
        manifest["scenarios"][scenario] = {
            "history": str(out / "offline_planning_history.json"),
            "history_sha256": _sha(out / "offline_planning_history.json"),
            "under_fraction": sum(v == 0.7 for v in factors.values()) / len(factors),
            "over_fraction": sum(v == 1.3 for v in factors.values()) / len(factors),
        }
    (args.output_root / "scenario_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    if args.history.read_bytes() != original_bytes:
        raise RuntimeError("source history changed")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
