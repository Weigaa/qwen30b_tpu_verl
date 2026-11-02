import ast
from pathlib import Path
from types import SimpleNamespace


def test_elastic_recapture_keeps_live_request_and_batch_identity():
    root = Path(__file__).parents[1]
    path = root / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "reset_aclgraph_recapture_state")
    env = {"logger": SimpleNamespace(info=lambda *args: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    request = object()
    requests = {"live_request": request}
    batch = SimpleNamespace(req_id_to_index={"live_request": 0})
    runner = SimpleNamespace(requests=requests, input_batch=batch, dp_rank=0,
                             attn_metadata={"stale": True},
                             spec_decode_common_attn_metadata=object())
    env[method.name](runner, preserve_live_requests=True)
    assert runner.requests is requests
    assert runner.requests["live_request"] is request
    assert runner.input_batch is batch
    assert batch.req_id_to_index == {"live_request": 0}
    assert runner.attn_metadata == {}
    assert runner.spec_decode_common_attn_metadata is None
    worker = (root / "vllm_ascend/worker/worker_v1.py").read_text()
    assert "reset_state(preserve_live_requests=True)" in worker
