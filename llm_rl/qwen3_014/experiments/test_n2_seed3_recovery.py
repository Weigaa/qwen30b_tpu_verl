from pathlib import Path
import pytest
from recover_n2_seed3_epoch2 import require_preexecution_failure,resume_script,natural_resume_script
from run_n2_f8_campaign import MAIN


def test_cannot_retry_executed_generation_or_unknown_failure():
    text="Owner's node has crashed. The actor never ran - it was cancelled before it started running."
    require_preexecution_failure(text)
    for marker in ('rollout_output_time_s:', 'MODE1_STEP_RUNTIME_AUDIT ',
                   'rollout_forward_begin', 'Initializing a V1 LLM engine'):
        with pytest.raises(RuntimeError):require_preexecution_failure(text+marker)
    with pytest.raises(RuntimeError):require_preexecution_failure('OOM')


def test_keep_all_original_validation_and_parameters():
    original=(MAIN/'run_paper_fair_epoch1_2_from_common_epoch0.sh').read_text()
    text=resume_script(original,MAIN,Path('/owned/storage'),Path('/test/run'))
    assert text.count('validate_completed_epoch')==original.count('validate_completed_epoch')
    assert 'resume_checkpoint="$resume_epoch1/checkpoints/' in text
    assert 'DYNAMIC_START_EPOCH=2 DYNAMIC_INITIAL_BASELINE_DIR="$resume_history"' in text
    assert 'if [[ "$variant" != adafloor_n_f2 ]]' in text
    assert text.index('validate_completed_epoch "$resume_run_dir" 001')<text.index('ln -s -- /owned/storage')
    # No changes below the initial resume adapter, including all training,
    # completion, output-shard and private checkpoint cleanup validators.
    boundary='if [[ "$finalize_existing" != "1" ]]; then'
    assert text.split(boundary)[1:]==original.split(boundary)[1:]


def test_natural_wrapper_keeps_full_colon_history_and_runtime():
    original=(MAIN/'run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh').read_text()
    text=natural_resume_script(original,MAIN)
    assert '${BASELINE_EPOCH0%%:*}/rollout_data' in text
    assert 'DYNAMIC_INITIAL_BASELINE_DIR="$BASELINE_EPOCH0"' in text
    boundary='export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural'
    assert text.split(boundary)[1]==original.split(boundary)[1]
