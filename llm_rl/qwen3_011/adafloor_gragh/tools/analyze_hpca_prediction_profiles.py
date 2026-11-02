#!/usr/bin/env python3
"""Read-only response-profile admission stress; no torch/NPU imports."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

try:
    from .analyze_hpca_prediction_robustness import FLOORS, DEFAULT_FACTORS, sha256
except ImportError:
    from analyze_hpca_prediction_robustness import FLOORS, DEFAULT_FACTORS, sha256


def peak(profile):
    """Logical token peak, inclusive completion tick, no byte conversion."""
    return max((sum(prompt + tick for prompt, response in profile if response >= tick)
                for tick in {0.0, *(response for _, response in profile)}), default=0.0)


def inflate(profiles, factor, scenario):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("inflation must be finite and >= 1")
    hottest = max(sorted(profiles), key=lambda rank: peak(profiles[rank]))
    result = {}
    for rank, pairs in profiles.items():
        longest = max(response for _, response in pairs)
        result[rank] = [(prompt, response * factor if (
            scenario == "global" or scenario == "hottest_rank" and rank == hottest
            or scenario == "local_tail" and response == longest) else response)
            for prompt, response in pairs]
    return result


def evaluate(step, profiles, caps, factor, scenario, mu, max_response):
    inflated = inflate(profiles, factor, scenario)
    # Stress exceeding the configured response bound is deliberately not clipped:
    # rejection must remain observable rather than being hidden by censoring.
    adjusted = {rank: [(prompt, max(response, min(response * mu, max_response)))
                       for prompt, response in pairs] for rank, pairs in inflated.items()}
    peaks = {rank: peak(pairs) for rank, pairs in adjusted.items()}
    inflated_peak = max(peak(pairs) for pairs in inflated.values())
    current = int(step["selected_floor"])
    chosen = next((floor for floor in FLOORS if floor >= current and caps.get(floor, 0) > 0
                   and max(peaks.values()) <= caps[floor]), None)
    disposition = ("unsafe_workload_rejection" if chosen is None else
                   "safe_current_floor" if chosen == current else
                   "full16_fallback" if chosen == 16 else "safe_floor_escalation")
    finish = {rank: max(response for _, response in pairs) for rank, pairs in inflated.items()}
    tail = max(finish.values())
    area = 0.0
    previous = set(finish)
    if chosen is not None:
        for floor, survivors in zip(step["shrink_stages"], step["stage_survivor_ranks"], strict=True):
            if floor < chosen:
                break
            survivors = set(survivors)
            if not survivors.issubset(previous):
                raise ValueError("shrink survivor sets must be nested")
            detached = previous - survivors
            if not detached:
                continue
            threshold = max(finish[rank] for rank in set(finish) - survivors)
            area += len(detached) * max(0.0, tail - threshold)
            previous = survivors
    return {"step": step["step"], "scenario": scenario, "inflation": factor,
            "nominal_floor": current, "selected_floor": chosen, "disposition": disposition,
            "fallback_reason": None if chosen == current else "prompt_plus_inflated_response_peak_exceeds_cap",
            "rank_kv_peak_tokens": {str(k): v for k, v in peaks.items()},
            "current_floor_admitted": max(peaks.values()) <= caps[current],
            "inflated_profile_peak_tokens": inflated_peak,
            "inflated_profile_within_nominal_cap": inflated_peak <= caps[current],
            "full16_admitted": max(peaks.values()) <= caps[16],
            "selected_headroom_tokens": None if chosen is None else caps[chosen] - max(peaks.values()),
            "predicted_release_rank_token_proxy": area,
            "original_release_rank_token_proxy": step["release_area"],
            "response_bound_exceedance_count": sum(response > max_response for pairs in inflated.values()
                                                    for _, response in pairs)}


def build_profiles(manifest, provenance):
    import pandas as pd
    source = Path(manifest["paper_formatter"])
    spec = importlib.util.spec_from_file_location("paper_planner_profile_formatter", source)
    module = importlib.util.module_from_spec(spec)
    # dataclass needs its module registered; this module imports numpy/pandas only.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    provenance[str(source)] = sha256(source)
    train = Path(manifest["train_file"])
    provenance[str(train)] = sha256(train)
    frame = pd.read_parquet(train)
    prompt_lengths = {}
    decoder = json.JSONDecoder()
    for text in manifest["rollout_files"]:
        path = Path(text)
        provenance[str(path)] = sha256(path)
        with path.open() as handle:
            for line in handle:
                # Original files put input first. Decode only that field for
                # already-seen prompts to avoid parsing large response tensors.
                start = line.index('"input"') + len('"input"')
                start = line.index(":", start) + 1
                while line[start].isspace():
                    start += 1
                prompt_input, _ = decoder.raw_decode(line, start)
                if prompt_input in prompt_lengths:
                    continue
                tokens = json.loads(line)["prompts"]
                first = next((i for i, token in enumerate(tokens)
                              if token != manifest["pad_token_id"]), len(tokens))
                length = len(tokens) - first
                if not length:
                    raise ValueError("empty/all-padding prompt")
                prompt_lengths[prompt_input] = length
    by_source = {int(index): module._prompt_to_input(row["prompt"])
                 for index, row in frame.iterrows() if int(index) < manifest["source_prompt_count"]}
    if len(by_source) != manifest["source_prompt_count"]:
        raise ValueError("source prompt count does not match dataset indices")
    return by_source, prompt_lengths


def analyze(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["cap_units"] != "tokens":
        raise ValueError("caps must explicitly be in logical tokens, not bytes/blocks")
    provenance = {str(manifest_path): sha256(manifest_path), str(Path(__file__)): sha256(Path(__file__))}
    helper = Path(__file__).with_name("analyze_hpca_prediction_robustness.py")
    provenance[str(helper)] = sha256(helper)
    for evidence in manifest.get("cap_evidence_files", []):
        provenance[evidence] = sha256(Path(evidence))
    source_inputs, prompt_lengths = build_profiles(manifest, provenance)
    records, cases = [], []
    for case in manifest["cases"]:
        path, history_path, summary_path = map(Path, (case["plan"], case["history"], case["summary"]))
        for input_path in (path, history_path, summary_path):
            provenance[str(input_path)] = sha256(input_path)
        plan = json.loads(path.read_text())
        summaries = json.loads(summary_path.read_text())
        history = {row["input"]: row["lengths"] for row in json.loads(history_path.read_text())["records"]}
        caps = {int(k): float(v) for k, v in case["caps"].items()}
        if set(caps) != set(FLOORS) or any(not math.isfinite(v) or v < 0 for v in caps.values()):
            raise ValueError("invalid per-policy floor caps")
        if caps[16] <= 0:
            raise ValueError("Full16 capacity is required")
        for step in plan:
            if caps[int(step["selected_floor"])] != float(step["kv_cap"]):
                raise ValueError(f"{case['label']}: caps disagree with original plan")
            summary = next(row for row in summaries if row["step"] == step["step"])
            mu, bound = summary["active_peak_safety_factor"], summary["max_response_len"]
            profiles = {}
            for rank, indices in step["rank_to_source_idx"].items():
                pairs = [(prompt_lengths[source_inputs[index]], float(response))
                         for index in indices for response in history[source_inputs[index]]]
                if len(pairs) != 32:
                    raise ValueError("each original rank must have 32 response profiles")
                profiles[int(rank)] = pairs
                raw = peak([(0, response) for _, response in pairs])
                adjusted = peak([(0, min(response * mu, bound)) for _, response in pairs])
                if not math.isclose(raw, step["rank_peak_loads"][rank], abs_tol=1e-5) or not math.isclose(
                        adjusted, step["rank_adjusted_peak_loads"][rank], abs_tol=1e-5):
                    raise ValueError(f"{case['label']} step {step['step']} rank {rank}: history does not reproduce plan")
            for factor in DEFAULT_FACTORS:
                for scenario in ("global", "hottest_rank", "local_tail"):
                    records.append({"case": case["label"], **evaluate(step, profiles, caps, factor, scenario, mu, bound)})
        cases.append({**case, "plan_sha256": provenance[str(path)], "baseline_response_peaks_reproduced": True})
    counts = {}
    for row in records:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return {"schema_version": 2, "status": "OFFLINE_PROFILE_AUDIT_COMPLETE",
            "cases": cases, "records": records, "disposition_counts": counts, "input_sha256": provenance,
            "unique_plan_count": len({case["plan_sha256"] for case in cases}),
            "claim_boundary": "Exact logical prompt+response KV profiles at the retained fixed rank layout; "
            "not allocator-block accounting, higher-floor reoptimization, runtime no-preemption proof, or independent seed replication. "
            "Counterfactual admission recomputation assumes the inflated profile is available; "
            "it does not demonstrate online detection of unknown forecast error or automatic runtime escalation. "
            "Inflation=1 reproduces the original response-only peak; adding prompt KV can change admission. "
            "Local tail inflates every response tied for longest on each rank. Response-bound exceedances are not clipped. "
            "Release is recomputed as logical rank-token proxy, not measured rank-time."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.manifest)
    for name in ("hpca_prediction_robustness.json", "hpca_prediction_robustness.md"):
        if str((args.output_dir / name).resolve()) in {str(Path(p).resolve()) for p in payload["input_sha256"]}:
            raise ValueError("output would overwrite an input artifact")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hpca_prediction_robustness.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Response-profile prediction stress", "", payload["claim_boundary"], "",
             f"Cases: {len(payload['cases'])}; unique plans: {payload['unique_plan_count']}; scenarios: {len(payload['records'])}.",
             "", "| Outcome | Count |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in sorted(payload["disposition_counts"].items())]
    lines += ["", "Complete rank peaks, floors, headroom, predicted release and input SHA256 are in the paired JSON."]
    (args.output_dir / "hpca_prediction_robustness.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "counts": payload["disposition_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
