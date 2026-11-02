#!/usr/bin/env python3
"""Trusted entry point, only run inside sandbox_code's restricted namespace."""
import importlib.util
import inspect
import json
import os
import sys


def main():
    if os.path.exists('/root/.codex') or os.path.exists('/workspace') or os.path.exists('/data'):
        raise RuntimeError('Refusing a host-visible execution environment')
    case = json.load(open('/case/case.json'))
    # pyext0.7's Python3 implementation uses getfullargspec; its import also
    # references the removed legacy name and an optional interactive frontend.
    # Compatibility is confined to this disposable child, not installed into
    # the live RL interpreter or the frozen runtime source tree.
    inspect.getargspec = inspect.getfullargspec
    sys.modules['IPython'] = None
    pyext_spec = importlib.util.spec_from_file_location('pyext', '/grade/pyext.py')
    pyext = importlib.util.module_from_spec(pyext_spec)
    sys.modules['pyext'] = pyext
    pyext_spec.loader.exec_module(pyext)
    spec = importlib.util.spec_from_file_location('isolated_prime_testing', '/grade/testing_util.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if case.get('diagnostic'):
        # Tests use this ONLY for sandbox isolation checks, never for scoring.
        namespace = {}
        exec(case['answer'], namespace)
        result = dict(passed=True, diagnostic=namespace.get('result'))
    else:
        solution = case['answer'].split('```python')[-1].split('```')[0]
        tests = case['tests']
        if not isinstance(tests, dict) or not tests.get('inputs') or len(tests['inputs']) != len(tests['outputs']):
            raise ValueError('Invalid full test suite')
        results, _ = module.run_test(tests, solution, debug=False, timeout=5)
        # bool(np.bool_) is safe; negative compilation/time-out codes are not
        # truthy passes. Do not compare only an early passing prefix.
        flags = [bool(value) if isinstance(value, (bool, module.np.bool_)) else value == 1 for value in results]
        result = dict(passed=len(flags) == len(tests['inputs']) and all(flags),
                      tests=len(tests['inputs']), returned_tests=len(flags), test_passes=sum(flags))
    print('JUDGE_JSON: ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
