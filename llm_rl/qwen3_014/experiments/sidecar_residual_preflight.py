"""Bounded, owned NPU residual reservation for a colocation preflight only."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(directory,device,gib):
    import torch
    import torch_npu
    torch.npu.set_device(device)
    before,total=torch.npu.mem_get_info()
    chunks=[torch.empty(1024**3,dtype=torch.uint8,device=f'npu:{device}') for _ in range(gib)]
    for chunk in chunks:chunk.zero_()
    torch.npu.synchronize()
    after,_=torch.npu.mem_get_info()
    requested=False
    def stop(signum,frame):
        nonlocal requested
        requested=True
    signal.signal(signal.SIGTERM,stop)
    record=dict(pid=os.getpid(),local_device=device,held_torch_bytes=sum(x.numel() for x in chunks),
                physical_free_before=before,physical_free_after=after,total_bytes=total)
    (directory/f'ready.{device}.json').write_text(json.dumps(record)+'\n')
    while not requested:time.sleep(.2)
    del chunks
    torch.npu.empty_cache()


@contextmanager
def reserve(directory,gib,env):
    if not gib:
        yield
        return
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    env=dict(env,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    processes=[];logs=[];records=[]
    try:
        for device in range(8):
            log=(directory/f'device{device}.log').open('x');logs.append(log)
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',
                str(directory),str(device),str(gib)],env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            processes.append(process)
        deadline=time.monotonic()+180
        while len(list(directory.glob('ready.*.json')))<8:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Residual reservation failed; see '+str(directory))
            if time.monotonic()>deadline:raise RuntimeError('Residual reservation readiness timeout')
            time.sleep(.2)
        records=[json.loads(p.read_text()) for p in sorted(directory.glob('ready.*.json'))]
        assert len(records)==8 and all(r['held_torch_bytes']==gib*1024**3 for r in records)
        (directory/'RESERVATION.json').write_text(json.dumps(dict(diagnostic_only=True,
            gib_per_device=gib,device_count=8,records=records),indent=2)+'\n')
        yield
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=20)
        for log in logs:log.close()
        (directory/'RELEASED.json').write_text(json.dumps(dict(
            own_pids=[p.pid for p in processes],exit_codes=[p.returncode for p in processes],
            all_owned_processes_exited=all(p.poll() is not None for p in processes)))+'\n')


if __name__=='__main__':
    if len(sys.argv)!=5 or sys.argv[1]!='--worker':raise SystemExit('Internal owned residual worker only')
    worker(Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
