import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_sidecar_paired_windows.sh"
FAIR_RUNNER = ROOT / "run_paper_fair_epoch1_2_from_common_epoch0.sh"
DYNAMIC_RUNNER = ROOT / "run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"


def test_runner_uses_minimal_prespecified_protocol() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        'SIDECAR_PAIR_SEEDS="${SIDECAR_PAIR_SEEDS:-101 202 303}"',
        "101) printf '%s\\n' off on",
        "202) printf '%s\\n' on off",
        "303) printf '%s\\n' off on",
        "FAIR_PROMPTS_PER_EPOCH=160",
        "FAIR_TRAIN_STEPS=1",
        "FAIR_TOTAL_EPOCHS=2",
        "FAIR_FREEZE_ACTOR=1",
        "VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1",
        "DYNAMIC_DISABLE_TAIL_GUARD=1",
        "DYNAMIC_EXPECT_NO_RESPONSE_CAPS=1",
        "DYNAMIC_SHORT_STEP_CAP_ENABLE=0",
        "VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8",
        "VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1",
        "VERL_SIDECAR_SHRINK_QUORUM_SIZE=16",
        "VERL_SIDECAR_WORLD_SIZE=16",
        "VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1",
        "VERL_SIDECAR_REPLICA_COUNT=8",
        "VERL_SIDECAR_DATA_SPLIT=train",
        "VERL_SIDECAR_TEMPERATURE=0.0",
        "VERL_SIDECAR_TOP_P=1.0",
        "VERL_SIDECAR_MAX_TOKENS=4096",
        '"fast_step_subset": False',
        '"source_plan_step": 1',
        '"sidecar_trigger_active_ranks": 8',
        "Qwen2.5-1.5B-Instruct",
        '"sidecar_model_weights_sha256": model_weights_sha',
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        "VERL_SIDECAR_MAX_NUM_SEQS=128",
        "VERL_SIDECAR_GENERATE_CHUNK_SIZE=128",
        "VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=0",
        '"eager_weight_sync_group_init": False',
        "PRIMARY_HCCL_ALLOCATOR_START=12000",
        'HCCL_IF_BASE_PORT="$PRIMARY_HCCL_ALLOCATOR_START"',
        'VERL_HCCL_IF_BASE_PORT_START="$PRIMARY_HCCL_ALLOCATOR_START"',
        '"primary_hccl_allocator_start": int(primary_hccl_allocator_start)',
        '"primary_moe_shared_expert_overlap": False',
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False",
        "from verl.single_controller.ray.base import _alloc_hccl_if_base_port",
        '[[ "$allocated_hccl_base" == "$PRIMARY_HCCL_ALLOCATOR_START" ]]',
    )
    for fragment in required:
        assert fragment in source


def test_runner_changes_only_sidecar_enable_between_pair_arms() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '[[ "$arm" == "on" ]] && enable_sidecar=1' in source
    assert 'VERL_SIDECAR_ENABLE="$enable_sidecar"' in source
    assert "adafloor_p_f4" in source
    assert '"actor_rollout_ref.rollout.seed=$seed"' in source
    assert (
        "actor_rollout_ref.actor.megatron.override_transformer_config."
        "moe_shared_expert_overlap=False"
    ) in source
    assert "FAIR_KEEP_COMPLETED_CHECKPOINTS=0" in source


def test_fair_runner_can_plan_five_steps_and_execute_one() -> None:
    source = FAIR_RUNNER.read_text(encoding="utf-8")
    assert 'FAIR_TRAIN_STEPS="${FAIR_TRAIN_STEPS:-$FAIR_STEPS_PER_EPOCH}"' in source
    assert 'export DYNAMIC_PLAN_STEPS="$FAIR_STEPS_PER_EPOCH"' in source
    assert 'export DYNAMIC_TRAIN_STEPS="$FAIR_TRAIN_STEPS"' in source
    assert '${#rollout_files[@]} != FAIR_TRAIN_STEPS' in source
    assert 'global_step_${FAIR_TRAIN_STEPS}' in source

    dynamic_source = DYNAMIC_RUNNER.read_text(encoding="utf-8")
    assert 'validate_rollout_dir "$output_dir" "epoch $epoch mode1" "$train_steps"' in dynamic_source
    assert 'build_offline_planning_history "$output_dir" "$train_steps"' in dynamic_source


def _runner_function(name: str, next_name: str) -> str:
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index(f"{name}() {{")
    end = source.index(f"\n}}\n\n{next_name}() {{", start) + 2
    return source[start:end]


def _run_immutable_install(candidate: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    function = _runner_function(
        "install_or_verify_immutable_file", "root_has_arm_state"
    )
    command = "\n".join(
        (
            "set -euo pipefail",
            function,
            "install_or_verify_immutable_file "
            f"{shlex.quote(str(candidate))} {shlex.quote(str(destination))} test",
        )
    )
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        text=True,
        capture_output=True,
    )


def test_immutable_contract_install_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "protocol.env"
    first = tmp_path / "first.tmp"
    first.write_text("version=one\n", encoding="utf-8")
    result = _run_immutable_install(first, destination)
    assert result.returncode == 0, result.stderr
    assert not first.exists()
    assert destination.read_text(encoding="utf-8") == "version=one\n"

    same = tmp_path / "same.tmp"
    same.write_text("version=one\n", encoding="utf-8")
    result = _run_immutable_install(same, destination)
    assert result.returncode == 0, result.stderr
    assert not same.exists()

    changed = tmp_path / "changed.tmp"
    changed.write_text("version=two\n", encoding="utf-8")
    result = _run_immutable_install(changed, destination)
    assert result.returncode != 0
    assert "immutable test differs" in result.stderr
    assert not changed.exists()
    assert destination.read_text(encoding="utf-8") == "version=one\n"


def test_status_updates_preserve_fixed_launch_order_on_resume(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.tsv"
    manifest.write_text(
        "seed\tarm\tlaunch_order\trun_dir\tstatus\n"
        "101\toff\t1\t/run/101/off\tcomplete\n"
        "101\ton\t2\t/run/101/on\tcomplete\n"
        "202\ton\t3\t/run/202/on\tcomplete\n"
        "202\toff\t4\t/run/202/off\tcomplete\n",
        encoding="utf-8",
    )
    function = _runner_function("record_status", "check_disk")
    command = "\n".join(
        (
            "set -euo pipefail",
            f"MANIFEST={shlex.quote(str(manifest))}",
            function,
            'record_status 101 off 1 /run/101/off complete',
            'record_status 101 on 2 /run/101/on complete',
        )
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["seed", "arm", "launch_order", "run_dir", "status"]
    assert [int(row[2]) for row in rows[1:]] == [1, 2, 3, 4]
    assert len({(row[0], row[1]) for row in rows[1:]}) == 4


def test_runner_seals_complete_execution_code_and_resume_protocol() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for path in (
        "verl/experimental/dataset/shrink_aware_assignment.py",
        "verl/single_controller/ray/base.py",
        "verl/workers/megatron_workers.py",
        "vllm/v1/engine/llm_engine.py",
        "vllm_ascend/worker/worker_v1.py",
    ):
        assert path in source
    assert 'code_candidate=$(mktemp "$SIDECAR_PAIR_ROOT/.code_sha256.txt.XXXXXX")' in source
    assert 'protocol_candidate=$(mktemp "$SIDECAR_PAIR_ROOT/.protocol.env.XXXXXX")' in source
    assert 'ln -- "$candidate" "$destination"' in source
    assert 'cmp -s -- "$candidate" "$destination"' in source
    assert "existing arm state is missing an immutable root contract" in source
    assert "sort -t $'\\t' -k3,3n" in source
    assert source.count("seal_or_verify_root_contracts") >= 6
    run_arm = source[
        source.index("run_arm() {") : source.index("\nverify_results() {")
    ]
    assert run_arm.index("seal_or_verify_root_contracts\n    write_arm_manifest") < run_arm.rindex(
        'record_status "$seed" "$arm" "$order" "$run_dir" complete'
    )
    assert source.rindex("seal_or_verify_root_contracts") < source.rindex(
        "verify_results"
    )
