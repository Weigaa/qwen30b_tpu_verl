import json
from pathlib import Path
import pytest
from rank_occupancy import PAPER,parse_epoch


def test_matches_existing_paper_native_sleep_restore_ledger():
    ledger=json.loads((PAPER.parent/'evidence/rank_time_accounting.json').read_text())
    for policy in ('vanilla','planned_f4','natural_f2'):
        run=next(r for r in ledger['runs'] if r['policy']==policy and r['seed']==0)
        for epoch in run['epochs']:
            result=parse_epoch(Path(epoch['epoch_dir']),{s['step']:s['response_tokens'] for s in epoch['steps']},'FULL')
            expected=[sum(s['occupied_rank_s_bounds'][i] for s in epoch['steps']) for i in range(2)]
            assert result['occupied_rank_s_bounds']==pytest.approx(expected,abs=1e-6)
            assert result['Q_response_tokens_s']==pytest.approx(result['response_tokens']/result['rollout_seconds'])


def test_graph_trace_cannot_be_relabelled_eager():
    ledger=json.loads((PAPER.parent/'evidence/rank_time_accounting.json').read_text())
    epoch=next(r for r in ledger['runs'] if r['policy']=='vanilla' and r['seed']==0)['epochs'][0]
    with pytest.raises(RuntimeError,match='execution mode'):
        parse_epoch(Path(epoch['epoch_dir']),{s['step']:s['response_tokens'] for s in epoch['steps']},'eager')
