"""Lossless adapter for Eurus native test cases to the pinned PRIME wire format.

Only representation changes. Never inspect an answer or choose a more
favorable expected result. The original judge and sandbox are unchanged.
"""
import copy
import json


def _json(value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False)
    if json.loads(text)!=value:
        raise ValueError('Test value does not losslessly round-trip through JSON')
    return text


def normalize(tests):
    if (not isinstance(tests,dict) or not tests.get('inputs') or
            len(tests['inputs'])!=len(tests['outputs'])):
        raise ValueError('Invalid complete test suite')
    wire=copy.deepcopy(tests);changed=False
    call=tests.get('fn_name') is not None
    for i,(inputs,outputs) in enumerate(zip(tests['inputs'],tests['outputs'])):
        if call:
            if isinstance(inputs,list):
                if not inputs:raise ValueError('Empty argument lists are unsupported by pinned judge')
                encoded='\n'.join(_json(arg) for arg in inputs)
                if [json.loads(v) for v in encoded.split('\n')]!=inputs:
                    raise ValueError('Argument list changed during wire conversion')
                wire['inputs'][i]=encoded
                wire['outputs'][i]=_json(outputs)
                changed=True
            elif isinstance(inputs,str) and isinstance(outputs,str):
                # Already encoded upstream. Validate, but preserve bytewise.
                [json.loads(v) for v in inputs.split('\n')]
                json.loads(outputs)
            else:raise ValueError('Unknown call-based test representation')
        else:
            for field,value in (('inputs',inputs),('outputs',outputs)):
                if isinstance(value,str):continue
                if not isinstance(value,list) or not all(isinstance(v,str) for v in value):
                    raise ValueError('Unknown stdin/stdout line representation')
                # This is exactly the line-list join in the original judge,
                # moved before its string-only debug truncation code.
                wire[field][i]='\n'.join(value);changed=True
    if len(wire['inputs'])!=len(tests['inputs']):raise ValueError('Test cases were removed')
    return wire,changed


def infrastructure_format_error(row):
    stderr=row.get('stderr_tail','')
    return ('/grade/testing_util.py' in stderr and
        (('truncatefn' in stderr and 'assert isinstance(s, str)' in stderr) or
         "AttributeError: 'list' object has no attribute 'split'" in stderr))
