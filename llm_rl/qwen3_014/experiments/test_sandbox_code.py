import unittest
from sandbox_code import run_case


class SandboxTests(unittest.TestCase):
    def test_correct_and_wrong_function(self):
        # PRIME's call-based wire format is one JSON argument per line.
        cases = dict(fn_name='add', inputs=['2\n3', '10\n-2'], outputs=['5', '8'])
        self.assertTrue(run_case('def add(a,b): return a+b', cases)['passed'])
        self.assertFalse(run_case('def add(a,b): return 0', cases)['passed'])

    def test_standard_input_full_suite(self):
        cases = dict(inputs=['2 3\n', '10 -2\n'], outputs=['5\n', '8\n'])
        self.assertTrue(run_case('a,b=map(int,input().split())\nprint(a+b)', cases)['passed'])

    def test_host_hidden_fork_and_network_denied(self):
        result = run_case('''import os, socket
result = {'host_root':os.path.exists('/root/.codex'), 'host_data':os.path.exists('/data'),
          'npu':os.path.exists('/dev/davinci0'), 'secret':os.getenv('OPENAI_API_KEY')}
for name, fn in [('fork', os.fork), ('socket', socket.socket)]:
    try:
        fn()
        result[name] = 'UNEXPECTED_ALLOWED'
    except PermissionError:
        result[name] = 'denied'
''', {}, diagnostic=True)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['diagnostic'], dict(host_root=False, host_data=False, npu=False,
                                                   secret=None, fork='denied', socket='denied'))

    def test_wall_timeout_is_failure(self):
        result = run_case('while True: pass', {}, diagnostic=True, wall_seconds=1, cpu_seconds=4)
        self.assertFalse(result['passed'])
        self.assertTrue(result['timed_out'])


if __name__ == '__main__':
    unittest.main()
