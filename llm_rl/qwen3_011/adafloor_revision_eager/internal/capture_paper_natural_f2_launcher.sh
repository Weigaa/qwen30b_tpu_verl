#!/usr/bin/env bash
# Use the unchanged paper launcher. Intercept only its final trainer command,
# after all environment defaults and Hydra arguments have been resolved.
set -euo pipefail
python3() {
    if [[ "${1:-}" == -m && "${2:-}" == verl.trainer.main_ppo ]]; then
        command python3 "$ADAFLOOR_PROTOCOL_TOOL" boundary -- "$@" || return $?
        if [[ "$ADAFLOOR_PROTOCOL_CAPTURE_ONLY" == 1 ]]; then
            return 80
        fi
    fi
    command python3 "$@"
}
export -f python3
exec bash "$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh" "$@"
