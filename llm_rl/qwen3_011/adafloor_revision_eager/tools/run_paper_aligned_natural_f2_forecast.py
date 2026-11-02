#!/usr/bin/env python3
"""Replay the successful paper Natural-F2 launcher chain with audited changes.

The driver and child launcher set their own original defaults. A shell shim
captures the *final* environment/argv immediately before launching main_ppo.
No NPU, torch, model, or graph module is imported by the preflight tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
PAPER = PACKAGE.parent / "qwen3_shrink_aware"
COMMON = Path("/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe")
CHECKPOINT = COMMON / "checkpoints/qwen3moe_for_eagle3/global_step_5"
HISTORICAL = Path("/data/adafloor_shared_state/paper_fair_suite_seed2_retry")
DRIVER_LOG = HISTORICAL / "driver_logs/07_adafloor_n_f2.log"
HISTORICAL_RUN = HISTORICAL / "results/adafloor_natural_floor2_tailguard_common_epoch0_epoch1_2/epoch_001_mode1_natural"
DRIVER = PAPER / "run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
SHIM = PACKAGE / "internal/capture_paper_natural_f2_launcher.sh"
ENV_PREFIXES = ("VLLM", "VERL", "DYNAMIC_", "FAIR_", "ADAFLOOR_", "ROLLOUT_", "HCCL_", "TORCH", "PYTORCH", "ASCEND_", "NPU_")
# These are observations from the successful driver header, not new tuning.
HEADER_ENV = {
    "VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS": "2048",
    "VLLM_ASCEND_MODE1_PROFILE_EXTRA_MC2_DUMMY": "0",
    "VLLM_ASCEND_MODE1_PROFILE_MC2_DUMMY_TOKENS": "128",
    "VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY": "1",
    "VLLM_ASCEND_MODE1_PROFILE_MC2_AVOID_TOKENS": "640",
    "VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS": "2048",
    "VLLM_ASCEND_MODE1_USE_COLD_INIT_KV_CAP": "1",
    "VLLM_ASCEND_MODE1_BOOTSTRAP_KV_BEFORE_WEIGHT_ONLOAD": "1",
    "VLLM_ASCEND_MODE1_BOOTSTRAP_KV_TOKENS": "131072",
    "VLLM_ASCEND_MODE1_STAGE_WEIGHTS_BEFORE_KV_REBUILD": "1",
    "VLLM_ASCEND_MODE1_PRESERVE_KV_ON_SAME_FLOOR_NEXT_STEP": "0",
    "VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC": "1",
    "VLLM_ASCEND_MODE1_REFRESH_GROUPS_ON_KV_RESIZE": "1",
    "VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP": "1",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP": "1",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS": "32",
    "VLLM_ASCEND_MODE1_PARITY_MC2_WARMUP_ROUTE": "global",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2": "131072",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4": "280576",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8": "315648",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS": "380800",
    "VLLM_ASCEND_SHRINK_AWARE_STAGES": "8,4,2",
    "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE": "2",
    "VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE": "0",
    "VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS": "0",
    "VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS": "0",
    "VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS": "0",
    "VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE": "0",
    "VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE": "1",
    "VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_EXIT_THRESHOLD": "4096",
    "VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_TOKENS": "4096",
    "VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_FLOORS": "2,4",
    "VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME": "1",
    "MIN_ADAPTIVE_FLOOR": "2",
    "ACTIVE_PEAK_SAFETY_FACTOR": "1.16",
    "LENGTH_EMA_DECAY": "0.3",
}
PATH_KEYS = {
    "data.train_files", "trainer.default_local_dir", "trainer.rollout_data_dir",
    "trainer.rollout_length_dir", "actor_rollout_ref.rollout.shrink_aware.oracle_trace_path",
    "actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path",
}
# Only differences necessary for the requested frozen single-window ablation.
EXPERIMENT_KEYS = {
    "actor_rollout_ref.actor.optim.lr": "0.0",
    "actor_rollout_ref.rollout.seed": "505",
    "trainer.save_freq": "-1",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def hydra_values(argv):
    result = {}
    for arg in argv:
        if "=" in arg and not arg.startswith("--"):
            k, v = arg.split("=", 1)
            result[k.lstrip("+")] = v
    return result


def historical_protocol():
    raw = DRIVER_LOG.read_text()
    prefix = "+ python3 -m verl.trainer.main_ppo "
    command = next(line[len("+ python3 "):] for line in raw.splitlines() if line.startswith(prefix))
    env = dict(HEADER_ENV)
    first_epoch = raw.split(prefix, 1)[0]
    for line in first_epoch.splitlines():
        match = re.match(r"^\++ export (.*)$", line)
        if match:
            for token in shlex.split(match.group(1)):
                if "=" in token:
                    k, v = token.split("=", 1)
                    if k.startswith(ENV_PREFIXES) or k in ("TASK_QUEUE_ENABLE", "USE_ALLTOALL_OVERLAP", "ALL_TO_ALL_RESHARD"):
                        env[k] = v
    assert "profile_max_tokens=2048" in raw
    assert "cold_init_kv_tokens=2048 use_cold_init_kv_cap=1" in raw
    assert "disable_tail_guard=0 short_step_cap_enable=1" in raw
    return {"argv": shlex.split(command), "hydra": hydra_values(shlex.split(command)), "env": env}


def compare_configuration(reference, actual, env):
    errors, differences = [], []
    for k in sorted(set(reference["hydra"]) | set(actual)):
        old, new = reference["hydra"].get(k), actual.get(k)
        if old == new:
            continue
        permitted = k in PATH_KEYS or (k in EXPERIMENT_KEYS and new == EXPERIMENT_KEYS[k])
        differences.append({"key": k, "historical": old, "current": new, "permitted": permitted})
        if not permitted:
            errors.append(f"Hydra mismatch {k}: {old!r} != {new!r}")
    for k, value in reference["env"].items():
        # Log/artifact destination variables necessarily follow the new run root.
        if k in ("VLLM_ASCEND_DRAFT_DUMP_DIR",):
            continue
        if env.get(k) != value:
            errors.append(f"environment mismatch {k}: {value!r} != {env.get(k)!r}")
    for k in ("ADAFLOOR_REVISION_HARD_KV_GUARD", "VERL_PAIRED_REQUEST_SAMPLING_SEEDS"):
        if env.get(k, "0") != "0":
            errors.append(f"later revision feature must not be enabled: {k}")
    return errors, differences


def boundary(argv):
    reference = historical_protocol()
    env = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES) or k in HEADER_ENV or k in ("TASK_QUEUE_ENABLE", "USE_ALLTOALL_OVERLAP", "ALL_TO_ALL_RESHARD", "PYTHONPATH", "PATCH_TREE")}
    actual = hydra_values(argv)
    errors, differences = compare_configuration(reference, actual, env)
    if Path.cwd().resolve() != PAPER.resolve():
        errors.append("trainer working directory is not the paper tree")
    python_paths = env.get("PYTHONPATH", "").split(":")
    # CANN's unchanged set_env.sh prepends its two system operator paths.
    # The first project-source root must still be PAPER, and cwd is PAPER.
    project_paths = [p for p in python_paths if p and not p.startswith("/usr/local/Ascend/")]
    if not project_paths or Path(project_paths[0]).resolve() != PAPER.resolve():
        errors.append("paper tree is not first project source in PYTHONPATH")
    if any("adafloor_gragh" in p or "adafloor_revision_eager" in p for p in python_paths):
        errors.append("graph/revision tree leaked into PYTHONPATH")
    payload = {"status": "PASS" if not errors else "FAIL", "errors": errors,
               "permitted_differences": differences, "env": env, "argv": argv,
               "cwd": str(Path.cwd()), "historical_driver_sha256": sha(DRIVER_LOG)}
    audit_path = Path(os.environ["ADAFLOOR_PROTOCOL_BOUNDARY_JSON"])
    write_json(audit_path, payload)
    if errors:
        raise SystemExit("paper protocol audit failed before NPU launch:\n" + "\n".join(errors))
    print(f"[paper protocol] PASS env_checks={len(reference['env'])} hydra_keys={len(actual)} differences={len(differences)}", flush=True)


def clean_environment(root, scenario, capture):
    # Prevent later graph/revision/diagnostic variables from silently changing
    # defaults. CANN paths are set by the unchanged original internal launcher.
    exact = {"TASK_QUEUE_ENABLE", "USE_ALLTOALL_OVERLAP", "ALL_TO_ALL_RESHARD", "MIN_ADAPTIVE_FLOOR", "ACTIVE_PEAK_SAFETY_FACTOR", "FLOOR_KV_CAPS", "LENGTH_EMA_DECAY", "PYTHONPATH", "PATCH_TREE", "REPO_ROOT", "LOCAL_TEST_LAUNCHER", "SAVE_CKPT_ENABLE", "TRAINER_SAVE_FREQ", "MODEL_PATH", "DISTCP_PATH", "TRAIN_FILE_ORIG", "TEST_FILE", "MAX_RESPONSE_LEN", "MAX_RESPONSE_LENGTH", "MAX_PROMPT_LENGTH", "BASELINE_DIR", "BASELINE_DIRS", "PLAN_DIR", "OUTPUT_ROOT", "OUTPUT_SUBDIR", "TRAIN_BATCH_SIZE", "ROLLOUT_N", "PLAN_STEPS", "PREBUILT_PLAN_DIR"}
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES) and k not in exact and not k.startswith(("FORCE_SELECTED", "TAIL_GUARD_", "MAX_RANK_", "MAX_CROSS_", "REPAIR_", "RANK_MATCHING", "KV_SAFE_", "DATASET_FRACTION", "TRAINER_", "OPT_", "DRAFT_", "TENSORBOARD_"))}
    branch = "preflight" if capture else "runs"
    env.update({
        "PATCH_TREE": str(PAPER), "PYTHONPATH": str(PAPER),
        "DYNAMIC_INITIAL_BASELINE_DIR": str(root / "scenarios" / scenario),
        "DYNAMIC_INITIAL_RESUME_CKPT": str(CHECKPOINT),
        "DYNAMIC_OUTPUT_ROOT": str(root / branch / scenario),
        "DYNAMIC_RUN_NAME": "natural_f2", "DYNAMIC_START_EPOCH": "1",
        "DYNAMIC_TOTAL_EPOCHS": "2", "DYNAMIC_ENABLE_CKPT_CHAIN": "0",
        # The original fair outer wrapper supplied these two shared defaults.
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.9", "DYNAMIC_LENGTH_EMA_DECAY": "0.3",
        "SAVE_CKPT_ENABLE": "0", "TRAINER_SAVE_FREQ": "-1",
        "MAX_ACTOR_CKPT_TO_KEEP": "1",
        "LOCAL_TEST_LAUNCHER": str(SHIM), "ADAFLOOR_PROTOCOL_TOOL": str(HERE),
        "ADAFLOOR_PROTOCOL_CAPTURE_ONLY": "1" if capture else "0",
        "ADAFLOOR_PROTOCOL_BOUNDARY_JSON": str(root / branch / scenario / "launch_boundary.json"),
    })
    return env


def launch(root, scenario, capture):
    env = clean_environment(root, scenario, capture)
    out = root / ("preflight" if capture else "runs") / scenario
    out.mkdir(parents=True, exist_ok=True)
    # Progress reset, epochs, profile, KV, TailGuard and comm parameters are
    # supplied by the historical outer chain, not reimplemented here.
    command = ["bash", str(DRIVER), "trainer.resume_mode=resume_path",
               f"trainer.resume_from_path={CHECKPOINT}",
               "actor_rollout_ref.actor.optim.lr=0.0", "actor_rollout_ref.rollout.seed=505"]
    with (out / "driver.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=PAPER, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(out / "process.json", {"pid": proc.pid, "command": command, "capture_only": capture})
        rc = proc.wait()
    write_json(out / "exit.json", {"exit_code": rc, "capture_only": capture})
    expected = 80 if capture else 0
    if rc != expected:
        raise RuntimeError(f"{scenario} {'preflight' if capture else 'run'} exit={rc}, expected={expected}. Evidence: {out}")


def validate_history_and_code(root):
    assert (CHECKPOINT / ".PRESERVE_COMMON_EPOCH0").is_file()
    assert (CHECKPOINT / "actor").is_dir()
    logs = sorted((HISTORICAL_RUN / "logs").glob("*.txt"))
    assert len(logs) == 1
    text = logs[0].read_text(errors="replace")
    assert "After trainer.fit" in text and "Training Progress: 100%" in text
    assert text.count("rollout_output_time_s:") == 5
    assert "Preempting request" not in text
    snapshot = json.loads((PACKAGE / "PAPER_SOURCE_SNAPSHOT.json").read_text())
    entries = {r["relative_path"]: r for r in snapshot["copied_regular_files"]}
    names = ["vllm_ascend/worker/model_runner_v1.py", "vllm_ascend/worker/worker_v1.py",
             "vllm_ascend/ops/fused_moe.py", "vllm/v1/engine/core.py", "vllm_ascend/core/scheduler.py",
             "verl/trainer/ppo/ray_trainer.py", "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
             "tools/build_mode1_length_sorted_e2e_plan.py", str(DRIVER.relative_to(PAPER)),
             "run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh",
             "run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"]
    code = {name: sha(PAPER / name) for name in names}
    mismatches = [name for name in names if code[name] != entries[name]["sha256"]]
    if mismatches:
        raise RuntimeError(f"paper-source snapshot mismatch: {mismatches}")
    code[str(HERE)] = sha(HERE)
    code[str(SHIM)] = sha(SHIM)
    code["forecast_scenario_generator"] = sha(PACKAGE / "tools/create_revision_forecast_scenarios.py")
    code["internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"] = sha(PAPER / "internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh")
    write_json(root / "code_sha256.json", code)
    write_json(root / "historical_reference.json", {
        "historical_driver": str(DRIVER_LOG), "driver_sha256": sha(DRIVER_LOG),
        "successful_log": str(logs[0]), "successful_log_sha256": sha(logs[0]),
        "suite_manifest_sha256": sha(HISTORICAL / "SUITE_MANIFEST.txt"),
        "code_scope": "matches preserved paper-source snapshot, not a complete July binary/environment archive",
        "common_history_sha256": sha(COMMON / "offline_planning_history.json"),
        "common_checkpoint_marker_sha256": sha(CHECKPOINT / ".PRESERVE_COMMON_EPOCH0"),
    })


def audit_run(root, scenario):
    run = root / "runs" / scenario / "natural_f2/epoch_001_mode1_natural"
    data = sorted((run / "rollout_data").glob("*.jsonl"))
    lengths = sorted((run / "rollout_length").glob("length_*.txt"))
    counts = [sum(1 for _ in p.open()) for p in data]
    length_counts = [len(p.read_text().splitlines()) for p in lengths]
    logs = sorted((run / "logs").glob("*.txt"))
    text = "\n".join(p.read_text(errors="replace") for p in logs)
    preemptions = text.count("Preempting request")
    ooms = bool(re.search(r"NPU out of memory|OutOfMemoryError|Memory_Allocation_Failure", text))
    aborted = bool(re.search(r"response/aborted_ratio:(?:0\.[0-9]*[1-9]|[1-9])", text))
    errors = []
    if len(data) != 5 or len(lengths) != 5 or counts != [512] * 5 or length_counts != [512] * 5:
        errors.append("incomplete 5x512 rollout/length artifacts")
    if preemptions or ooms or aborted:
        errors.append(f"preemptions={preemptions} oom={ooms} aborted={aborted}")
    if text.count("rollout_output_time_s:") != 5 or "After trainer.fit" not in text:
        errors.append("trainer did not complete five rollouts")
    if "original=17408 cap=2048 effective=2048" not in text:
        errors.append("missing runtime proof of historical profile cap")
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors,
              "run_dir": str(run), "response_counts": counts, "length_counts": length_counts,
              "preemption_count": preemptions, "oom": ooms, "aborted": aborted,
              "sha256": {str(p): sha(p) for p in data + lengths + logs}}
    write_json(root / "runs" / scenario / "audit.json", report)
    if errors:
        raise RuntimeError(f"{scenario}: validation failed, evidence retained: {errors}")


def main():
    if sys.argv[1:2] == ["boundary"]:
        argv = sys.argv[2:]
        boundary(argv[1:] if argv[:1] == ["--"] else argv)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("dry-run", "run"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--resume-preflight", action="store_true",
                    help="Start NPU arms from a completed CPU preflight without repeating the planner")
    args = ap.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(Path("/workspace/adafloor_revision_results")) or root.is_relative_to(COMMON):
        raise ValueError("unsafe output root")
    resume = args.resume_preflight
    if resume:
        protocol = json.loads((root / "protocol.json").read_text())
        if args.action != "run" or protocol.get("status") != "PREFLIGHT_PASS" or (root / "runs").exists():
            raise ValueError("resume-preflight requires a successful preflight with no NPU runs")
        for scenario in ("under_0p7", "mixed"):
            audit = json.loads((root / "preflight" / scenario / "launch_boundary.json").read_text())
            if audit.get("status") != "PASS":
                raise ValueError(f"preflight incomplete: {scenario}")
        code = json.loads((root / "code_sha256.json").read_text())
        for name, digest in code.items():
            if name == str(HERE):
                continue  # Supervisor-only CLI extension, never model/runtime code.
            path = PACKAGE / "tools/create_revision_forecast_scenarios.py" if name == "forecast_scenario_generator" else Path(name) if name.startswith("/") else PAPER / name
            if sha(path) != digest:
                raise ValueError(f"source changed after preflight: {name}")
    elif root.exists():
        raise FileExistsError("Use a new root. Existing evidence is never overwritten.")
    live = subprocess.run(["pgrep", "-f", "^python3 -m verl.trainer.main_ppo"], capture_output=True, text=True)
    if live.returncode == 0:
        raise RuntimeError("a trainer is already running")
    root.mkdir(parents=True, exist_ok=resume)
    process_file = "run_process.json" if resume else "process.json"
    write_json(root / process_file, {"supervisor_pid": os.getpid(), "action": args.action})
    try:
        if not resume:
            validate_history_and_code(root)
            subprocess.run([sys.executable, str(PACKAGE / "tools/create_revision_forecast_scenarios.py"),
                            "--history", str(COMMON / "offline_planning_history.json"),
                            "--output-root", str(root / "scenarios"), "--mixed-seed", "505",
                            "--regular-artifacts"], check=True, stdout=subprocess.DEVNULL)
            for scenario in ("under_0p7", "mixed"):
                launch(root, scenario, True)
            write_json(root / "protocol.json", {"status": "PREFLIGHT_PASS", "variant": "natural_f2",
                       "seed": 505, "frozen": True, "steps": 5, "normal_reused": True,
                       "npu_scenarios": ["under_0p7", "mixed"], "cpu_only_scenarios": ["over_1p3"]})
        print(f"[paper aligned] preflight PASS root={root}", flush=True)
        if args.action == "dry-run":
            return
        for scenario in ("under_0p7", "mixed"):
            print(f"[paper aligned] starting {scenario}", flush=True)
            launch(root, scenario, False)
            audit_run(root, scenario)
            print(f"[paper aligned] validated {scenario}", flush=True)
        ref = json.loads((root / "historical_reference.json").read_text())
        assert sha(COMMON / "offline_planning_history.json") == ref["common_history_sha256"]
        assert sha(CHECKPOINT / ".PRESERVE_COMMON_EPOCH0") == ref["common_checkpoint_marker_sha256"]
        write_json(root / "completion.json", {"status": "PASS", "common_epoch0_preserved": True})
    except Exception as error:
        write_json(root / "failure.json", {"status": "FAIL", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
