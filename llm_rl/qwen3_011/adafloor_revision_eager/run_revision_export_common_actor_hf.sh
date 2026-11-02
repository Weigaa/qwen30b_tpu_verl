#!/usr/bin/env bash
# Export a derived inference view of the preserved actor, never mutate epoch0.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent
ACTOR="$COMMON/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5/actor"
OUT=${REVISION_FRONTIER_HF_ROOT:-/workspace/adafloor_revision_results/frontier_common_actor_hf_v1}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ -f "$COMMON/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" && -s "$ACTOR/dist_ckpt/metadata.json" ]]
[[ -s "$ACTOR/huggingface/config.json" ]]
[[ $(realpath -m "$OUT") == /workspace/adafloor_revision_results/* && ! -e "$OUT" ]]
free_kib=$(df --output=avail -k /workspace | tail -1 | tr -d '[:space:]')
(( free_kib >= 80*1024*1024 )) || { echo 'need80GiB free for derived HF export' >&2; exit 2; }
printf 'operation=derived_common_actor_hf_export\nactor_source=%s\noutput=%s/model\nuse_cpu_initialization=true\ntraining=false\nsource_checkpoint_readonly=true\n' "$ACTOR" "$OUT"
[[ "$ACTION" == run ]] || exit 0
[[ -z $(pgrep -f '^python3 -m verl.trainer.main_ppo' || true) ]] || exit 3
mkdir -p "$OUT"
printf 'actor_source=%s\noperation=derived_common_actor_hf_export\ntraining=false\nsource_readonly=true\nsource_code=paper_eager_snapshot\n' "$ACTOR" > "$OUT/protocol.env"
sha256sum "$ACTOR/dist_ckpt/metadata.json" "$ACTOR/huggingface/config.json" \
    "$COMMON/epoch_000_mode0_probe/offline_planning_history.json" > "$OUT/input_sha256.txt"
sha256sum "$0" "$REPO/verl/model_merger/megatron_model_merger.py" \
    "$REPO/verl/model_merger/base_model_merger.py" "$REPO/verl/utils/megatron/dist_checkpointing.py" > "$OUT/code_sha256.txt"
cd "$REPO"
# Vendor environment scripts assume unset shell-detection variables are legal.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT
set +e
python3 -m verl.model_merger merge --backend megatron --local_dir "$ACTOR" \
    --target_dir "$OUT/model" --use_cpu_initialization > "$OUT/export.log" 2>&1
rc=$?
set -e
printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
sha256sum -c "$OUT/input_sha256.txt"
[[ "$rc" == 0 ]] || { echo "HF export failed rc=$rc; evidence retained" >&2; exit "$rc"; }
[[ -s "$OUT/model/config.json" && -s "$OUT/model/model.safetensors.index.json" ]] || exit 4
# Conversion still needs tensor coverage/parity checks before static controls
# may consume it. A successful exporter is not itself a model-parity proof.
echo 'HF_EXPORT_CREATED_NOT_YET_PARITY_CERTIFIED'
