import array
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import gates as g
import long_capture as old
import long_gates as lg
import resident_capture as resident
from build_resident_probe import RESIDENT_FILES,RESIDENT_ORIGIN

class ResidentCaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for module in (g,lg):
            changed=patch.object(module,'VOCAB',4);changed.start();self.addCleanup(changed.stop)
        self.engine=self.root/'engine';self.engine.write_bytes(b'candidate')
        self.baseline=self.root/'main';self.baseline.write_bytes(b'baseline')
        self.probes=[]
        for index,engine in enumerate((self.engine,self.baseline)):
            path=self.root/f'probe{index}';path.write_bytes(b'probe'+bytes([index]))
            g.save(Path(str(path)+'.build.json'),dict(capture_origin=RESIDENT_ORIGIN,
                   probe_hashes={name:g.digest(g.ROOT/name) for name in RESIDENT_FILES},
                   binary_sha256=g.digest(path),bound_engine_binary_sha256=g.digest(engine),
                   source_hashes={name:'0'*64 for name in ('engine/engine.hip','engine/cli/gen_cli.h','engine/exl3/safetensors.h')}))
            self.probes.append(path)
        self.prompts=[]
        for index in range(2):
            path=self.root/f'prompt{index}.ids';path.write_text(','.join([str(index)]*1024)+'\n')
            self.prompts.append(dict(file=str(path),tokens=1024,sha256=g.digest(path)))
        self.plan=dict(prompts=self.prompts,kl_files=[p['file'] for p in self.prompts],prefill_rows=4096,
                       candidate=str(self.engine),baseline=str(self.baseline),captures={})
        self.args=SimpleNamespace(long_resident_candidate_probe=self.probes[0],long_resident_baseline_probe=self.probes[1],
                       long_tokens=3,tokens=16,mtp_prefix=None,checkpoint=self.root,engine=self.engine,
                       gpu_run=Path('/queue'),priority=1)
        self.out=self.root/'output';self.out.mkdir()

    def execute(self,command,**kwargs):
        # The engine budget (300, or 600 for the 16K control) plus the 300 s memory wait inside the lease (Oct 14).
        budget=command[9]
        self.assertIn(budget,('600','900'))
        priority='2' if budget=='900' else '1'
        self.assertEqual(command[:10],['/queue','-p',priority,'-w','7200','--','timeout','-k','10',budget])
        self.assertEqual(command[10:14],['env','MEM_FLOOR_WAIT_S=300','MEM_FLOOR_POLL_S=5','bash'])
        rows=int(command[-2]);manifest=Path(command[-1])
        lines=manifest.read_text().splitlines()
        kwargs['stdout'].write((json.dumps(dict(event='resident',requests=len(lines),bulk_rows=rows))+'\n').encode())
        for line in lines:
            ids,positions,out,tokens=line.split('\t');out=Path(out);out.mkdir()
            native=out/'native';native.mkdir()
            (native/'stdout.log').write_text('ids:'+' 0'*int(tokens)+'\n')
            (native/'logits.f32').write_bytes(array.array('f',[1,0,0,0]).tobytes())
            if positions=='-':continue
            points=[int(n) for n in Path(positions).read_text().split(',')]
            clipped=out/'clipped';clipped.mkdir()
            (clipped/'logits.f32').write_bytes(array.array('f',[1,0,0,0]*len(points)).tobytes())
            events=[dict(event='point',position=p,argmax=0,copied_argmax=True) for p in points]
            length=len(Path(ids).read_text().strip().split(','))
            blocks=lg.schedule(length,points,rows if rows else 8)
            events.append(dict(event='schedule',kind='clipped schedule',blocks=[list(v) for v in blocks]))
            (clipped/'stdout.log').write_text(''.join(json.dumps(e)+'\n' for e in events))
        return SimpleNamespace(returncode=0)

    def test_compact_resident_reference_environment(self):
        prepared = resident.prepare(self.args, self.plan)
        def process(command, **kwargs):
            candidate = str(self.probes[0]) in command
            self.assertEqual(kwargs['env']['CRUCIBLE_COMPACT_STATE'], '1' if candidate else '0')
            return self.execute(command, **kwargs)
        with patch.dict(old.os.environ, {'CRUCIBLE_COMPACT_STATE': '1'}, clear=True), \
                patch.object(old.subprocess, 'run', side_effect=process):
            resident.collect(self.args, self.plan, self.out, prepared)
            self.assertEqual(old.os.environ['CRUCIBLE_COMPACT_STATE'], '1')

    def test_three_grants_and_complete_native_clipped_artifacts(self):
        prepared=resident.prepare(self.args,self.plan)
        with patch.object(old.subprocess,'run',side_effect=self.execute) as execute:
            records=resident.collect(self.args,self.plan,self.out,prepared)
        self.assertEqual(execute.call_count,3)
        self.assertEqual(set(records),{0,1})
        for row in records.values():
            self.assertEqual(len(row),6)
            for role in ('candidate','baseline','control'):
                self.assertEqual(row[role]['output_ids'],[0]*3)
                self.assertEqual(row[role+'-clipped']['point_count'],67)
                self.assertEqual(len(row[role+'-clipped']['argmax_ids']),67)

    def test_full_long_guard_reuses_all_three_role_captures(self):
        import long_run
        prepared=resident.prepare(self.args,self.plan)
        plan={**self.plan,'resident':prepared,'candidate_probe':None,'baseline_probe':None,
              'candidate_sha256':g.digest(self.engine),'baseline_sha256':g.digest(self.baseline),
              'fixed_corpus_sha256':g.digest(g.ROOT/'bench/suite/kl/manifest.json'),
              'tool_hashes':{},'bitwise_files':[p['file'] for p in self.prompts],
              'usage':{'baseline':{'native_draft_head':True}},'absolute_kl_max':None}
        with patch.object(old.subprocess,'run',side_effect=self.execute) as execute,\
             patch.object(old,'native') as native,patch.object(old,'clipped') as clipped:
            result=long_run.run(self.args,plan,self.root/'complete',{'native':True})
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(execute.call_count,3)
        native.assert_not_called();clipped.assert_not_called()
        self.assertEqual(result['sampled_kl']['status'],'PASS')
        self.assertEqual(result['production_final_kl']['status'],'PASS')
        self.assertEqual(len(result['captures']),12)
        self.assertTrue(all(row['candidate'] for row in result['boundary_invariance'].values()))

    def test_control_shards_keep_all_native_and_clipped_records(self):
        self.args.long_resident_shard='control'
        prepared=resident.prepare(self.args,self.plan)
        self.assertEqual(prepared['grants'],4)
        with patch.object(old.subprocess,'run',side_effect=self.execute) as execute:
            records=resident.collect(self.args,self.plan,self.out,prepared)
        self.assertEqual(execute.call_count,4)
        self.assertEqual(set(self.plan['resident_groups']),{'candidate','baseline','control-0','control-1'})
        self.assertTrue(all(len(record)==6 for record in records.values()))

    def test_all_roles_can_shard_without_losing_indices(self):
        self.args.long_resident_shard='all'
        prepared=resident.prepare(self.args,self.plan)
        with patch.object(old.subprocess,'run',side_effect=self.execute) as execute:
            records=resident.collect(self.args,self.plan,self.out,prepared)
        self.assertEqual(execute.call_count,6)
        self.assertEqual(prepared['grants'],6)
        self.assertEqual(set(records),{0,1})
        self.assertTrue(all(len(record)==6 for record in records.values()))

    def test_only_16k_control_has_explicit_600s_priority_two_exception(self):
        self.args.long_resident_shard='control'
        self.args.long_resident_16k_control_timeout=600
        prompt=self.prompts[1]
        Path(prompt['file']).write_text(','.join(['1']*16384)+'\n')
        self.plan['prompts']=[self.prompts[0],{**prompt,'tokens':16384,'sha256':g.digest(Path(prompt['file']))}]
        for mode,normal in (('control',3),('all',5)):
            with self.subTest(mode=mode):
                args=SimpleNamespace(**{**vars(self.args),'long_resident_shard':mode})
                plan={**self.plan,'captures':{},'resident_groups':{}}
                target=self.out/mode;target.mkdir()
                prepared=resident.prepare(args,plan)
                with patch.object(old.subprocess,'run',side_effect=self.execute) as execute:
                    records=resident.collect(args,plan,target,prepared)
                commands=[call.args[0] for call in execute.call_args_list]
                self.assertEqual([command[9] for command in commands],['600']*normal+['900'])  # each with the memory wait
                self.assertEqual([command[2] for command in commands],['1']*normal+['2'])
                group=plan['resident_groups']['control-1']
                self.assertEqual(group['timeout_exception']['prompt_tokens'],16384)
                self.assertEqual(group['timeout_exception']['prefill_rows'],0)
                self.assertTrue(all(len(record)==6 for record in records.values()))

    def test_invalid_shard_exception_modes_fail_before_queue(self):
        for mode,budget in (('bad',300),('none',600),('control',601),('control',True),('control',300.0)):
            with self.subTest(mode=mode,budget=budget):
                self.args.long_resident_shard=mode
                self.args.long_resident_16k_control_timeout=budget
                with self.assertRaises(ValueError):resident.prepare(self.args,self.plan)
        self.args.long_resident_shard='control'
        self.args.long_resident_16k_control_timeout=600
        with self.assertRaisesRegex(ValueError,'16K'):resident.prepare(self.args,self.plan)

    def test_execute_refuses_unbound_timeout_exception_before_queue(self):
        with patch.object(old.subprocess,'run') as process,self.assertRaises(ValueError):
            old.execute(['probe','ckpt','32768','0','manifest'],self.out/'bad-exception',self.args,
                        timeout_exception={'timeout_seconds':600})
        process.assert_not_called()

    def test_exception_rejects_small_or_multiple_requests_before_queue(self):
        args=SimpleNamespace(**{**vars(self.args),'priority':2})
        for indices in ([0],[0,1]):
            with self.subTest(indices=indices):
                path,_,_=resident.manifest(self.plan,self.out,'control',3,indices,f'test-{len(indices)}')
                command=['probe','ckpt','32768','0',str(path)]
                with patch.object(old.subprocess,'run') as process,self.assertRaises(ValueError):
                    old.execute(command,self.out/f'bad-{len(indices)}',args,timeout_exception=old.control_exception())
                process.assert_not_called()

    def test_direct_exception_validator_scope(self):
        prompt=self.root/'validator-16k.ids';prompt.write_text(','.join(['1']*16384)+'\n')
        valid_fields=[str(prompt),'points.ids','output','3']
        args=SimpleNamespace(**{**vars(self.args),'priority':2})
        path=self.out/'validator.tsv'
        path.write_text('\t'.join(valid_fields)+'\n')
        command=['probe','ckpt','32768','0',str(path)]
        old.validate_control_exception(command,args,old.control_exception())
        cases=[('priority',1,'0',valid_fields,1),('rows',2,'4096',valid_fields,1),
               ('points',2,'0',[valid_fields[0],'-',*valid_fields[2:]],1),
               ('multiple',2,'0',valid_fields,2),('short',2,'0',[self.prompts[0]['file'],*valid_fields[1:]],1)]
        for label,priority,rows,fields,count in cases:
            with self.subTest(case=label):
                path.write_text(('\t'.join(fields)+'\n')*count)
                candidate=SimpleNamespace(**{**vars(args),'priority':priority})
                with self.assertRaises(ValueError):
                    old.validate_control_exception([*command[:3],rows,str(path)],candidate,old.control_exception())
        with self.assertRaises(ValueError):old.validate_control_exception(command,args,{})

    def test_partial_options_and_budget_fail_before_queue(self):
        self.args.long_resident_baseline_probe=None
        with self.assertRaises(ValueError): resident.prepare(self.args,self.plan)
        self.args.long_resident_baseline_probe=self.probes[1]
        self.plan['prompts']=[self.prompts[0]]*8
        with self.assertRaises(ValueError): resident.prepare(self.args,self.plan)

    def test_failed_or_noop_process_never_passes(self):
        prepared=resident.prepare(self.args,self.plan)
        for index,code in enumerate((1,0)):
            target=self.root/f'failed{index}';target.mkdir()
            with patch.object(old.subprocess,'run',return_value=SimpleNamespace(returncode=code)):
                with self.assertRaises(ValueError):resident.collect(self.args,self.plan,target,prepared)

    def test_probe_and_manifest_mutation_fail_before_next_grant(self):
        for index,kind in enumerate(('probe','manifest')):
            target=self.root/f'mutated{index}';target.mkdir()
            prepared=resident.prepare(self.args,self.plan)
            def mutate(command,**kwargs):
                result=self.execute(command,**kwargs)
                path=self.probes[0] if kind=='probe' else Path(command[-1])
                path.write_bytes(path.read_bytes()+b'changed')
                return result
            with patch.object(old.subprocess,'run',side_effect=mutate) as execute:
                with self.assertRaises(ValueError):resident.collect(self.args,self.plan,target,prepared)
            self.assertEqual(execute.call_count,1)
            self.probes[0].write_bytes(b'probe'+bytes([0]))

    def test_source_binding_change_rejected(self):
        path=Path(str(self.probes[0])+'.build.json');record=json.loads(path.read_text())
        g.save(path,{**record,'bound_engine_binary_sha256':g.digest(self.baseline)})
        with self.assertRaises(ValueError):resident.prepare(self.args,self.plan)

if __name__=='__main__': unittest.main()
