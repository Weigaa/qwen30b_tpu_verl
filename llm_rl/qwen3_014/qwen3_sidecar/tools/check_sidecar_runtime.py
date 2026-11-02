"""Audit nested model-registry imports without loading weights or using an NPU."""
import json
from pathlib import Path
import os


def origins():
    import vllm
    import vllm.model_executor.models.registry as registry
    import vllm.model_executor.models.pangu_pro_moe_native as pangu
    return dict(cwd=os.getcwd(),vllm=str(Path(vllm.__file__).resolve()),
                registry=str(Path(registry.__file__).resolve()),
                pangu=str(Path(pangu.__file__).resolve()))


def main():
    from vllm.model_executor.models.registry import _run_in_subprocess
    root=Path(__file__).resolve().parents[1]
    local=origins();nested=_run_in_subprocess(origins)
    for label,record in [('parent',local),('registry_subprocess',nested)]:
        if Path(record['cwd']).resolve()!=root:
            raise RuntimeError(f'{label} inherited wrong cwd: {record}')
        for field in ('vllm','registry','pangu'):
            if not Path(record[field]).is_relative_to(root):
                raise RuntimeError(f'{label} imported another checkout: {record}')
    print(json.dumps(dict(status='PASS',expected_root=str(root),parent=local,
        registry_subprocess=nested,weight_loading=False,npu_execution=False)),flush=True)


if __name__=='__main__':main()
