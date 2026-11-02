from types import SimpleNamespace
from code_vanilla_bucket_plan import attach_tokenizer
import pytest


def test_history_mapper_adds_only_explicit_tokenizer_not_assignment_changes():
    seen = {}
    def original(*args,**kwargs):
        seen.update(args=args,kwargs=kwargs);return 'unchanged-result'
    module = SimpleNamespace(_map_stats_to_dataset=original)
    tok = object(); attach_tokenizer(module,tok)
    assert module._map_stats_to_dataset('train','history',1.0,-1,160)=='unchanged-result'
    assert seen['args']==('train','history',1.0,-1,160)
    assert seen['kwargs']==dict(prompt_tokenizer=tok)
    with pytest.raises(RuntimeError):module._map_stats_to_dataset(prompt_tokenizer=object())
