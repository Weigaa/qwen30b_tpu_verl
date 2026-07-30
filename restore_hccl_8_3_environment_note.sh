#!/usr/bin/env bash
# This repository did not overwrite /usr/local/Ascend HCCL libs during the HCCL master test path.
# To return to normal 8.3.RC1 runtime in a fresh shell, do not source use_hccl_master_9_1.sh.
# If /usr/local/Ascend HCCL libs are later overwritten manually, restore with:
#   bash /workspace/cann-recipes-train/llm_rl/qwen3/hccl_local_backups/20260630T121826Z/restore_hccl_backup.sh
cat <<'MSG'
Open a fresh shell or re-source the original CANN 8.3 environment.
Current /usr/local/Ascend HCCL backup restore script:
  /workspace/cann-recipes-train/llm_rl/qwen3/hccl_local_backups/20260630T121826Z/restore_hccl_backup.sh
MSG
