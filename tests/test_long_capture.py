"""Fresh long captures and fail-closed long assessments, without a GPU."""
import array
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import gates as g
import long_gates as lg
import long_capture as capture
import long_run as runner


class LongCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(lg,'VOCAB',4).start()
        self.engine = self.root/'engine'
        self.engine.write_bytes(b'engine')
        self.ids = self.root/'ids'
        self.ids.write_text(','.join(['1']*1024))
        self.args = SimpleNamespace(checkpoint=self.root,engine=self.engine,gpu_run=Path('/gpu-run'),
                                    priority=3,tokens=2,mtp_prefix=None)
        self.prompt = {'file':str(self.ids),'tokens':1024,'sha256':g.digest(self.ids)}
        self.mode = ''
        self.wait = '7200'
        self.low = 0

    def process(self,queued,stdout,stderr,**kwargs):
        time.sleep(.003)
        # The lease runs mem_floor.sh first (Oct 14): gpu-run ... timeout 600 env MEM_FLOOR_* bash mem_floor.sh 126 LOG -- CMD.
        self.assertEqual(queued[:10],['/gpu-run','-p','3','-w',self.wait,'--','timeout','-k','10','600'])
        floor = queued[10:queued.index('--',10)]
        self.assertEqual(floor[:4],['env','MEM_FLOOR_WAIT_S=300','MEM_FLOOR_POLL_S=5','bash'])
        self.assertEqual(floor[4:6],[str(g.ROOT/'tools/mem_floor.sh'),'126'])
        command = queued[queued.index('--',10)+1:]
        if self.mode=='status': return SimpleNamespace(returncode=75)
        if self.low:
            self.low -= 1
            with open(floor[6],'a') as log: log.write('00:00:00 MemAvailable 74000000 kB floor 126 GB below: no start after 300 s\n')
            return SimpleNamespace(returncode=76)
        if '--dump-logits' in command:
            path = Path(command[command.index('--dump-logits')+1])
            values = [0.,1.,2.,3.] if self.mode!='shape' else [0.]
            path.write_bytes(array.array('f',values).tobytes())
            stdout.write(b'ids: 2 2\n' if self.mode=='argmax' else b'ids: 3 2\n')
            self.assertIn(command[command.index('--prefill')+1],('0','4096'))
        else:
            points = tuple(map(int,Path(command[3]).read_text().strip().split(',')))
            rows = int(command[4]) or 8
            root = Path(command[-1]);root.mkdir()
            (root/'logits.f32').write_bytes(array.array('f',[0.,1.,2.,3.]*len(points)).tobytes())
            plan = [list(block) for block in lg.schedule(1024,points,rows)]
            if self.mode=='float': plan[-1][-1]=float(plan[-1][-1])
            if self.mode=='schedule': plan[-1][-1]+=1
            if self.mode=='extra': (root/'unexpected').write_bytes(b'wrong')
            events = [{'event':'point','position':point,'argmax':3,'copied_argmax':True} for point in points]
            if self.mode=='argmax': events[-1]['argmax']=2
            events.append({'event':'schedule','kind':'clipped schedule','blocks':plan})
            stdout.write(('\n'.join(json.dumps(event) for event in events)+'\n').encode())
        return SimpleNamespace(returncode=0)

    def test_a_lease_below_the_memory_floor_runs_again(self):
        self.low = 2
        with patch.object(capture.subprocess,'run',self.process):
            result = capture.native(self.engine,self.prompt,self.root/'retry',4096,self.args,None)
        self.assertEqual(result['returncode'],0)
        self.assertEqual(result['memory_retries'],2)
        self.assertEqual((self.root/'retry/memory.log').read_text().count('below: no start'),2)

    def test_the_memory_retries_end_at_the_queue_wait(self):
        self.low = 10**6
        self.args.queue_wait, self.wait = 60, '60'
        clock = iter(range(0,10**12,40*10**9))
        with patch.object(capture.subprocess,'run',self.process),patch.object(capture.time,'monotonic_ns',lambda:next(clock)):
            with self.assertRaisesRegex(ValueError,'memory floor'):
                capture.native(self.engine,self.prompt,self.root/'starve',4096,self.args,None)

    def test_native_full_shape_argmax_and_process_status(self):
        with patch.object(capture.subprocess,'run',side_effect=self.process),\
                patch.dict(capture.os.environ,{'CRUCIBLE_PF_PAD_H':'1'}):
            result = capture.native(self.engine,self.prompt,self.root/'native',0,self.args,None)
            self.assertEqual(result['environment']['CRUCIBLE_PF_PAD_H'],'1')
            self.assertEqual(result['output_ids'],[3,2])
            self.assertEqual(result['logits']['bytes'],16)
            for mode in ('argmax','shape','status'):
                self.mode=mode
                with self.assertRaises(ValueError): capture.native(self.engine,self.prompt,self.root/mode,0,self.args,None)

    def test_queue_wait_reaches_the_gpu_run_command(self):
        self.wait = '21600'
        args = SimpleNamespace(**{**vars(self.args), 'queue_wait': 21600})
        with patch.object(capture.subprocess,'run',side_effect=self.process):
            result = capture.native(self.engine,self.prompt,self.root/'wait',0,args,None)
        self.assertEqual(result['queued_command'][3:5],['-w','21600'])

    def test_compact_native_and_clipped_reference_environments(self):
        baseline = self.root/'baseline'
        baseline.write_bytes(b'baseline')
        def process(queued, **kwargs):
            expected = '1' if str(self.engine) in queued else '0'
            self.assertEqual(kwargs['env']['CRUCIBLE_COMPACT_STATE'], expected)
            return self.process(queued, **kwargs)
        with patch.dict(capture.os.environ, {'CRUCIBLE_COMPACT_STATE': '1'}, clear=True), \
                patch.object(capture.subprocess, 'run', side_effect=process):
            capture.native(self.engine, self.prompt, self.root/'compact-native', 4096, self.args, None)
            capture.native(baseline, self.prompt, self.root/'full-native', 4096, self.args, None)
            capture.clipped({'path': str(self.engine)}, self.prompt, self.root/'compact-clipped',
                            4096, self.args, reference=False)
            capture.clipped({'path': str(baseline)}, self.prompt, self.root/'full-clipped',
                            4096, self.args, reference=True)
            self.assertEqual(capture.os.environ['CRUCIBLE_COMPACT_STATE'], '1')

    def test_clipped_full_schedule_and_negative_exports(self):
        probe = {'path':str(self.engine)}
        with patch.object(capture.subprocess,'run',side_effect=self.process):
            result = capture.clipped(probe,self.prompt,self.root/'clipped',4096,self.args)
            self.assertEqual(result['points'],list(lg.sample_points(1024)))
            self.assertEqual(result['argmax_ids'],[3]*len(result['points']))
            for mode in ('argmax','schedule','float','extra','status'):
                self.mode=mode
                with self.assertRaises(ValueError): capture.clipped(probe,self.prompt,self.root/mode,4096,self.args)

    def test_prepare_fixed_corpus_and_independent_binary(self):
        baseline=self.root/'baseline'
        baseline.write_bytes(b'baseline')
        self.args.baseline_engine=baseline
        self.args.baseline_name='named-main'
        self.args.long_candidate_probe=self.engine
        self.args.long_baseline_probe=baseline
        self.args.long_kl_ids_files=sorted((g.ROOT/'bench/suite/kl').glob('*.ids'))
        with patch.object(capture,'usage',return_value={}),patch.object(capture,'probe_identity',return_value={}):
            plan=capture.prepare(self.args,self.engine)
            self.assertEqual(len(plan['kl_files']),7)
            self.assertEqual(plan['fixed_corpus_sha256'],g.digest(g.ROOT/'bench/suite/kl/manifest.json'))
            self.args.long_kl_ids_files=self.args.long_kl_ids_files[:-1]
            with self.assertRaisesRegex(ValueError,'seven'): capture.prepare(self.args,self.engine)
            self.args.long_kl_ids_files=[]
            self.args.long_bitwise_ids_files=[self.ids]
            with self.assertRaisesRegex(ValueError,'1K, 4K, and 16K'): capture.prepare(self.args,self.engine)

    def test_usage_requires_native_logit_export(self):
        for text,code in (('--dump-logits --prefill',2),('--dump-logits',2),('--dump-logits --prefill',0)):
            with patch.object(capture.subprocess,'run',return_value=SimpleNamespace(stdout=text,stderr='',returncode=code)):
                if text=='--dump-logits --prefill' and code==2:
                    self.assertFalse(capture.usage(self.engine)['mtp_prefix_capable'])
                else:
                    with self.assertRaises(ValueError): capture.usage(self.engine)

    def test_primary_bytes_ids_and_provenance_recheck(self):
        left=self.root/'left.f32';right=self.root/'right.f32'
        left.write_bytes(b'abcd');right.write_bytes(b'abcd')
        a={'logit_path':str(left),'output_ids':[1]};b={'logit_path':str(right),'output_ids':[1]}
        runner.equality(a,b)
        with self.assertRaises(ValueError): runner.equality(a,{**b,'output_ids':[2]})
        right.write_bytes(b'abce')
        with self.assertRaises(ValueError): runner.equality(a,b)
        plan={'candidate':str(self.engine),'candidate_sha256':g.digest(self.engine),'candidate_probe':None,
              'baseline':str(self.engine),'baseline_sha256':g.digest(self.engine),'baseline_probe':None,
              'prompts':[self.prompt],'tool_hashes':{},'captures':{},
              'fixed_corpus_sha256':g.digest(g.ROOT/'bench/suite/kl/manifest.json')}
        runner.verify(plan)
        self.ids.write_text('1')
        with self.assertRaisesRegex(ValueError,'prompt changed'): runner.verify(plan)

    def test_diagnostics_reject_candidate_schedule_mismatch(self):
        path=self.root/'logits.f32'
        path.write_bytes(array.array('f',[0.,1.,2.,3.]).tobytes())
        record={'logit_path':str(path),'point_count':1,'points':[1023],'schedule':[[0,1024]]}
        records={name:record for name in ('candidate','baseline','control','candidate-clipped','baseline-clipped','control-clipped')}
        records['baseline-clipped']={**record,'schedule':[[0,512],[512,512]]}
        with self.assertRaisesRegex(ValueError,'schedules differ'): runner.diagnostics({},self.prompt,records)
        records['baseline-clipped']=record
        samples,final,invariant=runner.diagnostics({},self.prompt,records)
        self.assertTrue(invariant['candidate'])
        self.assertEqual(samples[0]['phase'],'clipped schedule')
        self.assertEqual(final['phase'],'production schedule')

    def test_complete_long_exact_and_kl_pipeline(self):
        baseline=self.root/'baseline-engine';baseline.write_bytes(b'baseline')
        probe=lambda path,engine:{'path':str(path)}
        plan={'prompts':[self.prompt],'candidate':str(self.engine),'candidate_sha256':g.digest(self.engine),
              'baseline':str(baseline),'baseline_sha256':g.digest(baseline),
              'prefill_rows':4096,'usage':{'baseline':{'native_draft_head':True}},'tool_hashes':{},
              'captures':{},'bitwise_files':[str(self.ids)],'kl_files':[str(self.ids)],'absolute_kl_max':None,
              'candidate_probe':probe(self.engine,self.engine),'baseline_probe':probe(baseline,baseline),'status':'FAIL',
              'fixed_corpus_sha256':g.digest(g.ROOT/'bench/suite/kl/manifest.json')}
        with patch.object(capture.subprocess,'run',side_effect=self.process),patch.object(capture,'probe_identity',side_effect=probe):
            runner.run(self.args,plan,self.root/'complete',{'native':True,'mtp_prefix_capable':True})
        self.assertEqual(plan['status'],'PASS')
        self.assertEqual(plan['sampled_kl']['status'],'PASS')
        self.assertEqual(plan['production_final_kl']['status'],'PASS')
        self.assertEqual(len(plan['captures']),6)
        self.assertTrue(plan['boundary_invariance'][str(self.ids)]['candidate'])

    def test_kl_failure_cannot_publish_pass(self):
        measured = {'prompt':'test','candidate':{'forward':.1,'reverse':.1,'top1':0},
                    'baseline':{'forward':1e-6,'reverse':1e-6,'top1':0,'margin':1.},
                    'production':False,'control_top1':0}
        plan = {'prompts':[self.prompt],'candidate':str(self.engine),'baseline':str(self.engine),
                'prefill_rows':4096,'usage':{'baseline':{'native_draft_head':True}},
                'captures':{},'bitwise_files':[],'kl_files':[str(self.ids)],'absolute_kl_max':None,
                'candidate_probe':{'path':'fake'},'baseline_probe':{'path':'fake'},'status':'FAIL'}
        with patch.object(capture,'native',return_value={}),patch.object(capture,'clipped',return_value={}),\
             patch.object(runner,'diagnostics',return_value=([measured],{**measured,'production':True},{})),\
             patch.object(runner,'verify'),self.assertRaises(ValueError):
            runner.run(self.args,plan,self.root/'assessment',{'native':True})
        report = json.loads((self.root/'assessment/long-report.json').read_text())
        self.assertEqual(report['status'],'FAIL')
        self.assertEqual(report['sampled_kl']['status'],'FAIL')




class ProbeRecordLayoutTest(unittest.TestCase):
    def test_probe_record_keys_from_either_layout(self):
        # a probe record built before the layout move keeps its keys.
        row = g.layout.Layout(g.layout.load_map()).row('engine/cli/gen_cli.h')
        with tempfile.TemporaryDirectory() as directory:
            probe, engine = Path(directory) / 'probe', Path(directory) / 'engine'
            probe.write_bytes(b'probe'); engine.write_bytes(b'engine')
            for key in (row.old, row.new):
                record = {'capture_origin': 'o', 'probe_hashes': {}, 'binary_sha256': g.digest(probe),
                          'bound_engine_binary_sha256': g.digest(engine),
                          'source_hashes': {'engine/engine.hip': 'x', key: 'x', 'engine/exl3/safetensors.h': 'x'}}
                Path(str(probe) + '.build.json').write_text(json.dumps(record))
                capture.probe_identity(probe, engine, probe_files=(), origin='o')


if __name__ == '__main__': unittest.main()
