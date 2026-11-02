from types import SimpleNamespace as NS
import pytest
from cached_planner import replay_groups
from prepare_code_dataset import normalize,semantic


def test_repair_membership_replay_preserves_every_prompt_once():
    stats=[NS(load=i,source_idx=i) for i in range(8)]
    text='[mode1 length-sorted e2e repair] swap=1 step1:src3 <-> step2:src4\n'
    groups,n=replay_groups(stats,4,2,text)
    assert [[s.source_idx for s in group] for group in groups]==[[0,1,2,4],[3,5,6,7]]
    assert n==1 and sorted(s.source_idx for group in groups for s in group)==list(range(8))
    with pytest.raises(RuntimeError,match='source identity/group'):
        replay_groups(stats,4,2,text.replace('step1:src3','step1:src6'))


def test_only_row_id_metadata_changes_not_prompts_labels_or_other_metadata():
    rows=[dict(prompt=[dict(role='user',content='code')],reward_model={'ground_truth':'untouched'},
               extra_info=dict(index=0,split='dummy')) for _ in range(3)]
    original=repr(rows);new=normalize(rows)
    assert repr(rows)==original and semantic(new)==semantic(rows)
    for i,row in enumerate(new):
        assert row['extra_info']==dict(index=i,prompt_occurrence_ordinal=i,original_index=0,split='dummy')
