"""Execute the actual embedded streaming loop with a deterministic fake engine."""
import ast
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]/'qwen3_sidecar'


class StopCheckpointTests(unittest.TestCase):
    def run_stream(self, interval):
        source=(REPO/'internal/run_elastic_sidecar_infer.sh').read_text()
        # Select only the real function, without importing model code or NPU.
        start=source.index('def _generate_chunk_streaming(')
        end=source.index('\nmodel_path = ',start)
        tree=ast.parse(source[start:end])
        records=[dict(prompt_id=i,prompt=f'question{i}') for i in range(2)]
        writes=[];snapshots=[]

        class Engine:
            step_index=0
            def add_request(self,*args,**kwargs):pass
            def has_unfinished_requests(self):return self.step_index<100
            def step(self):
                self.step_index+=1
                return [SimpleNamespace(request_id=str(i),finished=False,
                    prompt=records[i]['prompt'],outputs=[SimpleNamespace(text='x'*self.step_index)])
                    for i in range(2)]
            def abort_request(self,ids):self.aborted=ids

        engine=Engine();llm=SimpleNamespace(llm_engine=engine,request_counter=iter(range(2)))
        namespace=dict(Path=Path,os=os,time=SimpleNamespace(perf_counter=lambda:0,time=lambda:0),
            json=__import__('json'),RequestOutputKind=SimpleNamespace(CUMULATIVE='cumulative'),
            _sampling_params_for_source=lambda params,item:SimpleNamespace(),
            _completion_payload=lambda out,prefix:({'text':prefix+out.text},len(out.text),prefix+out.text,len(out.text)),
            _write_skipped_output=lambda *args:None,
            _write_final_output=lambda *args:None,
            _append_jsonl=lambda path,rows:writes.append(list(rows)),
            _write_resume_records=lambda path,rows:snapshots.append(dict(rows)),
            _shutdown_requested=False,sidecar_epoch=0,shard_index=0,num_shards=2)
        def stop():
            namespace['_shutdown_requested']=engine.step_index==100
            return namespace['_shutdown_requested']
        namespace['_stop_requested']=stop
        sys.path.insert(0,str(REPO))
        try:
            exec(compile(tree,'embedded_sidecar_stream','exec'),namespace)
            with patch.dict(os.environ,VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS=str(interval),
                            VERL_SIDECAR_EXTERNAL_LAUNCHER='1'),contextlib.redirect_stdout(io.StringIO()):
                result=namespace['_generate_chunk_streaming'](llm,records,None,
                    Path('outputs'),Path('completed'),Path('resume'),Path('partials'),{},0,0)
        finally:sys.path.remove(str(REPO))
        return engine,writes,snapshots,result

    def test_stop_persists_only_latest_snapshot_without_historical_dump(self):
        engine,writes,snapshots,_=self.run_stream(0)
        self.assertEqual(engine.step_index,100)
        self.assertEqual(engine.aborted,['0','1'])
        self.assertEqual(len(writes),1)
        self.assertEqual(len(writes[0]),2)
        self.assertEqual({r['prompt_id'] for r in writes[0]},{0,1})
        self.assertTrue(all(r['outputs'][0]['text']=='x'*100 for r in writes[0]))
        self.assertEqual(set(snapshots[-1]),{0,1})
        self.assertTrue(all(r['partial_text']=='x'*100 for r in snapshots[-1].values()))

    def test_opt_in_trace_does_not_change_resume_state(self):
        _,_,without,result_without=self.run_stream(0)
        _,_,with_trace,result_with=self.run_stream(7)
        self.assertEqual(without[-1],with_trace[-1])
        self.assertEqual(result_without,result_with)


if __name__=='__main__':unittest.main()
