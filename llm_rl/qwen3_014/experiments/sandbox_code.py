#!/usr/bin/env python3
"""Run one generated Python answer in an OS sandbox, never on the host.

This limits damage and resource use; it is not a tamper-proof contest judge.
The grader and all available test cases are visible inside the sandbox.
"""
import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

CODE = Path(__file__).resolve().parent
BWRAP = CODE / '.deps/bwrap-root/usr/bin/bwrap'
PYTHON = '/usr/local/python3.11.14/bin/python3'
UTIL = CODE.parent / 'qwen3_main/verl/utils/reward_score/prime_code/testing_util.py'
PYEXT = CODE / '.deps/pyext-0.7/pyext.py'
ENV = dict(PATH='/usr/local/python3.11.14/bin:/usr/bin', LANG='C.UTF-8',
           LD_LIBRARY_PATH='/usr/local/python3.11.14/lib', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def export_filter(fd):
    """Export BPF only. Do not install a syscall filter in this parent."""
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.c_void_p]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = lib.seccomp_init(0x7fff0000)  # default ALLOW, restricted below
    if not context:
        raise RuntimeError('seccomp_init failed')
    denied = '''clone clone3 fork vfork socket socketpair connect bind listen accept accept4
        sendto sendmsg sendmmsg recvfrom recvmsg recvmmsg ptrace process_vm_readv process_vm_writev
        mount umount2 pivot_root chroot unshare setns bpf perf_event_open keyctl add_key request_key
        kexec_load kexec_file_load init_module finit_module delete_module reboot swapon swapoff
        open_by_handle_at name_to_handle_at userfaultfd io_uring_setup memfd_create'''.split()
    try:
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add_array(context, 0x50000 | errno.EPERM, number, 0, None):
                raise RuntimeError('Cannot restrict syscall ' + name)
        if lib.seccomp_export_bpf(context, fd):
            raise RuntimeError('Cannot export seccomp BPF')
    finally:
        lib.seccomp_release(context)
    os.lseek(fd, 0, os.SEEK_SET)


def run_case(answer, tests, *, wall_seconds=20, cpu_seconds=8, diagnostic=False):
    if not all(path.is_file() for path in (BWRAP, UTIL, PYEXT)):
        raise RuntimeError('Sandbox dependencies missing; no host-execution fallback')
    if not isinstance(answer, str) or len(answer.encode()) > 1024 * 1024:
        raise ValueError('Answer must fit the declared 1MiB input limit')
    if not 1 <= wall_seconds <= 60 or not 1 <= cpu_seconds <= 30:
        raise ValueError('Invalid sandbox limits')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mlsys_code_') as directory, tempfile.TemporaryFile() as bpf:
        root = Path(directory)
        (root / 'case.json').write_text(json.dumps(dict(answer=answer, tests=tests, diagnostic=diagnostic)))
        export_filter(bpf.fileno())
        command = ['prlimit', '--as=1073741824', f'--cpu={cpu_seconds}', '--fsize=1048576',
                   '--nofile=64', '--core=0', '--', str(BWRAP), '--unshare-all',
                   '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                   '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
                   '--ro-bind', '/lib64', '/lib64', '--dev', '/dev', '--proc', '/proc',
                   '--size', '67108864', '--tmpfs', '/tmp',
                   '--ro-bind', str(root), '/case',
                   '--ro-bind', str(CODE / 'sandbox_grader.py'), '/grade/grader.py',
                   '--ro-bind', str(PYEXT), '/grade/pyext.py',
                   '--ro-bind', str(UTIL), '/grade/testing_util.py', '--clearenv']
        for key, value in ENV.items():
            command.extend(['--setenv', key, value])
        command.extend(['--chdir', '/tmp', '--seccomp', str(bpf.fileno()),
                        PYTHON, '-I', '/grade/grader.py'])
        # Regular temporary files, rather than unbounded PIPE buffers, are also
        # covered by RLIMIT_FSIZE. No preexec_fn in a threaded scoring parent.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(command, env=ENV, pass_fds=(bpf.fileno(),),
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       start_new_session=True)
            timeout = False
            try:
                process.wait(timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                timeout = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            out.seek(0); stdout = out.read(1048576).decode(errors='replace')
            err.seek(0); stderr = err.read(1048576).decode(errors='replace')
        result = dict(passed=False, exit_code=process.returncode, timed_out=timeout,
                      elapsed_s=time.monotonic() - started, stderr_tail=stderr[-2000:])
        if process.returncode == 0 and not timeout:
            lines = [line[12:] for line in stdout.splitlines() if line.startswith('JUDGE_JSON: ')]
            if len(lines) != 1:
                result['error'] = 'Missing unique sandbox judge result'
            else:
                result.update(json.loads(lines[0]))
        else:
            result['error'] = 'Sandbox execution failed or exceeded fixed limits'
        return result
