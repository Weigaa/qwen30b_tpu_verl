import json
import pytest
from code_test_wire import normalize,infrastructure_format_error
from sandbox_code import run_case


def test_native_function_arguments_and_wrapped_expected_values_are_lossless():
    original=dict(fn_name='divisible_by',inputs=[[[1,2,3,4,5,6],2]],outputs=[[[2,4,6]]])
    wire,changed=normalize(original)
    assert changed and original['inputs']==[[[1,2,3,4,5,6],2]]
    assert [json.loads(x) for x in wire['inputs'][0].split('\n')]==original['inputs'][0]
    assert json.loads(wire['outputs'][0])==original['outputs'][0]
    assert run_case('def divisible_by(numbers,d): return [n for n in numbers if n%d==0]',wire)['passed']
    assert not run_case('def divisible_by(numbers,d): return []',wire)['passed']


def test_native_stdin_lines_preserve_blank_lines_and_match_pinned_judge():
    original=dict(inputs=[['2 3',''],['10 -2','']],outputs=[['5'],['8']])
    wire,changed=normalize(original)
    assert changed and wire==dict(inputs=['2 3\n','10 -2\n'],outputs=['5','8'])
    assert run_case('a,b=map(int,input().split())\nprint(a+b)',wire)['passed']


def test_preencoded_tests_and_string_arguments_are_not_reinterpreted():
    encoded=dict(fn_name='echo',inputs=['"hello\\nworld"'],outputs=['"hello\\nworld"'])
    assert normalize(encoded)==(encoded,False)
    native=dict(fn_name='echo',inputs=[['hello\nworld']],outputs=[['hello\nworld']])
    wire,changed=normalize(native)
    assert changed and json.loads(wire['inputs'][0])=='hello\nworld'
    assert json.loads(wire['outputs'][0])==['hello\nworld']


def test_fail_closed_for_unsupported_schemas_and_detect_host_judge_errors():
    with pytest.raises(ValueError):normalize(dict(inputs=[[1,2]],outputs=['3']))
    with pytest.raises(ValueError):normalize(dict(fn_name='f',inputs=[[]],outputs=[0]))
    assert infrastructure_format_error(dict(stderr_tail='/grade/testing_util.py truncatefn assert isinstance(s, str)'))
    assert not infrastructure_format_error(dict(stderr_tail='model SyntaxError'))
