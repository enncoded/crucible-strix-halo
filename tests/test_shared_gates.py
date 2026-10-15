"""CPU regressions for the shared gate runner. No GPU is used."""
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("fleet_gates", Path(__file__).parents[1] / "tools/gates.py")
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class GatesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # The disk preflight has its own tests; small test hosts must not trip it here.
        preflight = patch.object(g, "disk_preflight", return_value=0)
        preflight.start()
        self.addCleanup(preflight.stop)
        self.start = time.time_ns()
        # Mock writes need to cross the filesystem's coarse ctime tick.
        time.sleep(0.003)
        self.dump = self.root / "dump"
        self.dump.mkdir()
        self.payload = b"\x00\x00\x80\x3f" * g.WIDTH
        for name in g.NAMES:
            (self.dump / name).write_bytes(self.payload)

    def valid(self):
        return g.check_dumps(self.dump, 1, self.start)

    def test_complete(self):
        self.assertEqual(len(self.valid()), 49)

    def test_empty(self):
        for p in self.dump.iterdir():
            p.write_bytes(b"")
        with self.assertRaises(ValueError): self.valid()

    def test_stale(self):
        p = self.dump / g.NAMES[0]
        os.utime(p, ns=(self.start - 1000000000,) * 2)
        with self.assertRaises(ValueError): self.valid()

    def test_missing(self):
        (self.dump / g.NAMES[0]).unlink()
        with self.assertRaises(ValueError): self.valid()

    def test_truncated(self):
        (self.dump / g.NAMES[0]).write_bytes(self.payload[:-4])
        with self.assertRaises(ValueError): self.valid()

    def test_wrong_name(self):
        (self.dump / g.NAMES[0]).rename(self.dump / "h_0.f32")
        with self.assertRaises(ValueError): self.valid()

    def test_extra(self):
        (self.dump / "h_49.f32").write_bytes(self.payload)
        with self.assertRaises(ValueError): self.valid()

    def test_symlink(self):
        p = self.dump / g.NAMES[0]
        p.rename(self.root / "saved")
        p.symlink_to(self.root / "saved")
        with self.assertRaises(ValueError): self.valid()

    def test_content_comparison(self):
        other = self.root / "other"
        other.mkdir()
        for name in g.NAMES: (other / name).write_bytes(self.payload)
        g.compare_dumps(self.dump, other)
        (other / g.NAMES[-1]).write_bytes(b"x" * len(self.payload))
        with self.assertRaises(ValueError): g.compare_dumps(self.dump, other)

    def test_ids(self):
        self.assertEqual(g.parse_output("ids: 12 3\n", 2), [12, 3])
        for text in ("", "PASS", "ids:\n", "ids: 12\n", "ids: -1 3\n", "ids: 248320 3\n",
                     "ids: 12 3\nids: 12 3\n", "ids: 12 3 junk\n"):
            with self.subTest(text=text), self.assertRaises(ValueError): g.parse_output(text, 2)

    def test_id_files_empty_missing_stale(self):
        p = self.root / "ids.txt"
        with self.assertRaises(ValueError): g.check_ids(p, 2, self.start)
        p.write_text("")
        with self.assertRaises(ValueError): g.check_ids(p, 2, self.start)
        p.write_text("12,3\n")
        os.utime(p, ns=(self.start - 1000000000,) * 2)
        with self.assertRaises(ValueError): g.check_ids(p, 2, self.start)

    def test_noop_and_fake_summary(self):
        for text in ("", "PASS Gate 1\nPASS Gate 2\nPASS Gate 3\n", "ids: 1\nprefill 1 tok in 1 s\n"):
            def fake(command, **kwargs):
                kwargs["stdout"].write(text.encode())
                return type("Done", (), {"returncode": 0})()
            output = self.root / str(len(text))
            with patch.object(g.subprocess, "run", fake), self.assertRaises(ValueError):
                g.capture(Path('/bin/true'), self.root, self.root / 'prompt', output,
                          1, 1, 8, 0, 0, Path('/queue'))

    def test_failed_process(self):
        def fake(command, **kwargs):
            return type("Done", (), {"returncode": 7})()
        with patch.object(g.subprocess, "run", fake), self.assertRaises(ValueError):
            g.capture(Path('/bin/true'), self.root, self.root/'prompt', self.root/'failed',
                      1, 1, 8, 0, 0, Path('/queue'))

    def test_skipped_step(self):
        with self.assertRaises(ValueError): g.require_steps({'gate1': True, 'gate2': True})

    def test_oracle_points_and_thresholds(self):
        baseline = {(l, p): 1e-4 for l in range(48) for p in g.POSITIONS}
        g.check_oracle(baseline, baseline)
        for value in (float('nan'), float('inf'), 1.01e-3, 2.01e-4):
            points = {**baseline, (0, 0): value}
            with self.assertRaises(ValueError): g.check_oracle(points, baseline)
        with self.assertRaises(ValueError): g.check_oracle({(0, 0): 0}, baseline)
        with self.assertRaises(ValueError): g.check_oracle({k: v*1.11 for k,v in baseline.items()}, baseline)

    def arguments(self, name):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir(exist_ok=True)
        for file in ('config.json', 'quantization_config.json', 'model.safetensors.index.json'):
            (checkpoint / file).write_text('{}')
        prompt = self.root / 'prompt.ids'
        prompt.write_text('1\n')
        baseline = self.root/'baseline-engine'
        baseline.write_bytes(Path('/bin/true').read_bytes()+b'baseline-test-marker')
        return SimpleNamespace(output=self.root/name, engine=Path('/bin/true'),
                               checkpoint=checkpoint, ids=prompt, baseline_name='named-main',
                               baseline_engine=baseline, oracle_baseline=None,
                               tokens=3, prefill_rows=0, gpu_run=Path('/queue'), priority=3,
                               engine_build_record=None)

    def fake_engine(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        time.sleep(0.003)
        def option(flag): return command[command.index(flag) + 1]
        dump = Path(option('--dump'))
        n, spec = int(option('--n')), int(option('--spec'))
        rows = 1 + (n-1 if spec == 0 else 0)
        for name in g.NAMES: (dump/name).write_bytes(self.payload * rows)
        kwargs['stdout'].write(('ids:' + ' 7' * n + '\n').encode())
        return SimpleNamespace(returncode=0)

    def test_captures_run_from_pinned_copies_when_the_source_is_replaced(self):
        # Another process replaces both binaries after the start-time checks (the Oct 7 incident).
        args = self.arguments('pinned')
        sources = {}
        for name in ('engine', 'baseline'):
            path = self.root/('src-'+name)
            path.write_bytes(Path('/bin/true').read_bytes() + name.encode())
            path.chmod(0o755)
            sources[name] = (path, g.digest(path))
        args.engine, args.baseline_engine = sources['engine'][0], sources['baseline'][0]
        for field, name in (('engine_build_record', 'engine'), ('baseline_build_record', 'baseline')):
            record = self.root/(field+'.json')
            record.write_text(json.dumps({'binary_sha256': sources[name][1]}))
            setattr(args, field, record)
        executables, replaced = [], []
        def engine(command, **kwargs):
            if len(command) > 1:
                executables.append(Path(command[10]))
                if not replaced:
                    replaced.append(True)
                    for path, _ in sources.values(): path.write_bytes(b'replaced')
            return self.fake_engine(command, **kwargs)
        with patch.object(g.subprocess, 'run', engine): g.run(args)
        report = json.loads((args.output/'report.json').read_text())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['binary_sha256'], sources['engine'][1])
        self.assertEqual(report['baseline_binary_sha256'], sources['baseline'][1])
        self.assertEqual({p.parent for p in executables}, {(args.output/'bin').resolve()})
        for path in executables:
            self.assertIn(g.digest(path), (sources['engine'][1], sources['baseline'][1]))
            self.assertFalse(path.stat().st_mode & 0o222)
        self.assertEqual(report['pinned_binaries']['candidate']['sha256'], sources['engine'][1])
        self.assertEqual(report['pinned_binaries']['baseline']['source'], str(sources['baseline'][0]))

    def test_pinning_refuses_a_record_mismatch(self):
        args = self.arguments('pinned-mismatch')
        record = self.root/'record.json'
        record.write_text(json.dumps({'binary_sha256': '0'*64}))
        args.baseline_build_record = record
        with patch.object(g.subprocess, 'run', self.fake_engine):
            with self.assertRaisesRegex(ValueError, 'does not match the build record'): g.run(args)

    def test_full_run_all_gates_and_provenance(self):
        args = self.arguments('complete')
        with patch.object(g.subprocess, 'run', self.fake_engine): g.run(args)
        import json
        report = json.loads((args.output/'report.json').read_text())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(set(report['steps']), {'gate1', 'gate2', 'gate3'})
        self.assertEqual(len(report['captures']), 6)
        self.assertEqual(report['binary_sha256'], g.digest(Path('/bin/true')))
        self.assertEqual(report['baseline_name'], 'named-main')
        for capture in report['captures'].values():
            self.assertEqual(capture['queued_command'][:10], ['/queue', '-p', '3', '-w', '7200', '--',
                                                           'timeout','-k','10','300'])
            self.assertEqual(capture['returncode'], 0)
            self.assertEqual(len(capture['dumps']), 49)
            self.assertNotIn('--mtp-prefix', capture['command'])
            self.assertIsNone(capture['draft_policy']['mtp_prefix'])
        with self.assertRaises(FileExistsError): g.run(args)

    def batch_arguments(self,name):
        args=self.arguments(name)
        for field in ('engine_build_record','batch_gates_binary','batch_build_record','batch_prompt_list'):
            path=self.root/field
            # The engine record must hold the engine's real sha: a record is verified for native engines too.
            path.write_text(json.dumps({'binary_sha256':g.digest(args.engine)}) if field=='engine_build_record' else 'batch-input')
            setattr(args,field,path)
        return args

    def test_batch_hook_after_primary_gates(self):
        import json,sys
        from tools import batch_gate_report as batch
        args=self.batch_arguments('batch-good')
        calls=[]
        def engine(command,**kwargs):
            if '--dump' in command: calls.append(command)
            return self.fake_engine(command,**kwargs)
        def supplement(**kwargs):
            self.assertEqual(len(calls),6)
            self.assertEqual(kwargs["engine"].parent,(args.output/"bin").resolve())
            self.assertEqual(kwargs['primary_ids'],args.ids.resolve())
            self.assertEqual(kwargs['baseline_dump'],args.output/'candidate/dump')
            self.assertEqual(kwargs['probe_build_record'],args.batch_build_record.resolve())
            self.assertEqual(kwargs['priority'],3)
            kwargs['output'].mkdir()
            result={'status':'PASS','supplemental':True}
            g.save(kwargs['output']/'report.json',result)
            return result
        with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(batch,'run',side_effect=supplement),\
             patch.object(g.subprocess,'run',side_effect=engine): g.run(args)
        report=json.loads((args.output/'report.json').read_text())
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['batch']['status'],'PASS')
        self.assertIn(str(Path(batch.__file__).resolve()),report['batch']['inputs'])

    def test_batch_hook_partial_options_and_oracle_only_reject(self):
        import sys
        from tools import batch_gate_report as batch
        for missing in ('engine_build_record','batch_gates_binary','batch_build_record','batch_prompt_list','baseline_engine'):
            args=self.batch_arguments('batch-missing-'+missing)
            setattr(args,missing,None)
            with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(g.subprocess,'run',side_effect=self.fake_engine) as execute:
                with self.assertRaisesRegex(ValueError,'batch'): g.run(args)
            self.assertTrue(all(len(call.args[0])==1 for call in execute.call_args_list))

    def test_batch_hook_primary_failure_skips_supplement(self):
        import sys
        from tools import batch_gate_report as batch
        args=self.batch_arguments('batch-primary-failure')
        def engine(command,**kwargs):
            done=self.fake_engine(command,**kwargs)
            if '--dump' in command:
                dump=Path(command[command.index('--dump')+1])
                if dump.parent.name=='baseline': (dump/g.NAMES[0]).write_bytes(b'x'*len(self.payload))
            return done
        with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(batch,'run') as supplement,\
             patch.object(g.subprocess,'run',side_effect=engine):
            with self.assertRaises(ValueError): g.run(args)
        supplement.assert_not_called()

    def test_batch_hook_failure_never_passes(self):
        import json,sys
        from tools import batch_gate_report as batch
        for index,result in enumerate(({'status':'FAIL','supplemental':True},'PASS',{'status':'PASS'})):
            args=self.batch_arguments('batch-failed-'+str(index))
            with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(batch,'run',return_value=result),\
                 patch.object(g.subprocess,'run',side_effect=self.fake_engine):
                with self.assertRaisesRegex(ValueError,'batch'): g.run(args)
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['status'],'FAIL')
            self.assertTrue(all(report['steps'].values()))

    def test_batch_hook_rechecks_inputs_before_supplement(self):
        import sys
        from tools import batch_gate_report as batch
        args=self.batch_arguments('batch-mutated')
        def engine(command,**kwargs):
            done=self.fake_engine(command,**kwargs)
            if '--dump' in command and Path(command[command.index('--dump')+1]).parent.name=='spec2':
                args.batch_build_record.write_text('changed-after-primary')
            return done
        with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(batch,'run') as supplement,\
             patch.object(g.subprocess,'run',side_effect=engine):
            with self.assertRaises(ValueError): g.run(args)
        supplement.assert_not_called()

    def test_batch_hook_rechecks_inputs_after_supplement(self):
        import json,sys
        from tools import batch_gate_report as batch
        args=self.batch_arguments('batch-mutated-after')
        def supplement(**kwargs):
            args.batch_build_record.write_text('changed-in-supplement')
            return {'status':'PASS','supplemental':True}
        with patch.dict(sys.modules,{'batch_gate_report':batch}),patch.object(batch,'run',side_effect=supplement),\
             patch.object(g.subprocess,'run',side_effect=self.fake_engine):
            with self.assertRaises(ValueError): g.run(args)
        report=json.loads((args.output/'report.json').read_text())
        self.assertEqual(report['status'],'FAIL')

    def test_batch_hook_cli_options(self):
        import sys
        argv=['gates','checkpoint','--engine','engine','--ids','ids','--output','output',
              '--baseline-name','main','--baseline-engine','baseline','--engine-build-record','engine.json',
              '--batch-gates-binary','probe','--batch-build-record','probe.json','--batch-prompt-list','prompts']
        with patch.object(sys,'argv',argv),patch.object(g,'run') as execute: self.assertEqual(g.main(),0)
        args=execute.call_args.args[0]
        self.assertEqual(args.batch_gates_binary,Path('probe'))
        self.assertEqual(args.batch_build_record,Path('probe.json'))
        self.assertEqual(args.batch_prompt_list,Path('prompts'))

    def test_full_run_failures_never_pass(self):
        for bad in ('baseline', 'chunk1', 'spec2'):
            args = self.arguments('bad-' + bad)
            def fake(command, **kwargs):
                done = self.fake_engine(command, **kwargs)
                if len(command) == 1: return done
                directory = Path(command[command.index('--dump') + 1])
                if directory.parent.name == bad:
                    if bad == 'spec2': kwargs['stdout'].write(b'ids: 8 8 8\n')
                    else: (directory/g.NAMES[0]).write_bytes(b'x' * len(self.payload))
                return done
            with patch.object(g.subprocess, 'run', fake), self.assertRaises(ValueError): g.run(args)
            import json
            self.assertEqual(json.loads((args.output/'report.json').read_text())['status'], 'FAIL')

    def test_native_policy_is_explicit(self):
        args = self.arguments('native')
        with patch.object(g.subprocess, 'run', self.fake_engine): g.run(args)
        import json
        report = json.loads((args.output/'report.json').read_text())
        self.assertTrue(report['full_head_capability']['native'])
        for name, capture in report['captures'].items():
            cmd = capture['command']
            if name == 'baseline':
                self.assertNotIn('--draft-head', cmd)
                continue
            self.assertEqual(cmd[cmd.index('--draft-head')+1], 'full')
            self.assertEqual(cmd[cmd.index('--draft-head-auto')+1], '0')

    def test_legacy_policy_requires_audited_hashes(self):
        path = self.root/'build.json'
        import json
        source = next(iter(g.LEGACY_FULL_HEAD_SOURCES))
        record = {'binary_sha256': g.digest(Path('/bin/true')), 'source_hashes': {'engine/cli/gen_cli.h': source}}
        path.write_text(json.dumps(record))
        probe = SimpleNamespace(returncode=2, stdout='old usage', stderr='')
        with patch.object(g.subprocess, 'run', return_value=probe):
            with self.assertRaises(ValueError): g.full_head_policy(Path('/bin/true'))
            self.assertFalse(g.full_head_policy(Path('/bin/true'), path)['native'])
            path.write_text(json.dumps({**record, 'binary_sha256': 'wrong'}))
            with self.assertRaises(ValueError): g.full_head_policy(Path('/bin/true'), path)
            path.write_text(json.dumps({**record, 'source_hashes': {'engine/cli/gen_cli.h': 'unknown'}}))
            with self.assertRaises(ValueError): g.full_head_policy(Path('/bin/true'), path)

    def test_legacy_record_keys_from_either_layout(self):
        # records from before the layout move keep their keys.
        import json
        row = g.layout.Layout(g.layout.load_map()).row('engine/cli/gen_cli.h')
        source = next(iter(g.LEGACY_FULL_HEAD_SOURCES))
        probe = SimpleNamespace(returncode=2, stdout='old usage', stderr='')
        for key in (row.old, row.new):
            path = self.root/'build.json'
            path.write_text(json.dumps({'binary_sha256': g.digest(Path('/bin/true')), 'source_hashes': {key: source}}))
            with patch.object(g.subprocess, 'run', return_value=probe):
                self.assertFalse(g.full_head_policy(Path('/bin/true'), path)['native'], key)

    def test_live_bulk_gate3_and_integration_priority(self):
        args = self.arguments('bulk')
        args.prefill_rows, args.priority = 24, 0
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            done = self.fake_engine(command, **kwargs)
            if ('--prefill' in command and int(command[command.index('--prefill')+1]) > 0) or ('--dump' in command and
                    Path(command[command.index('--dump')+1]).parent.name == 'baseline'):
                dump = Path(command[command.index('--dump')+1])
                for name in g.NAMES:
                    path = dump/name
                    path.write_bytes(b'\x00\x00\x00\x40' * (path.stat().st_size // 4))
            return done
        with patch.object(g.subprocess, 'run', fake): g.run(args)
        for command in calls:
            if '--dump' not in command: continue
            self.assertEqual(command[2], '0')
            name = Path(command[command.index('--dump')+1]).parent.name
            expected = 24 if name in ('candidate','serial','spec2') else 0
            self.assertEqual(command[command.index('--prefill')+1],str(expected))

    def test_zero_prefill_is_explicit_for_every_role(self):
        args = self.arguments('explicit-zero')
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            return self.fake_engine(command,**kwargs)
        with patch.object(g.subprocess,'run',fake): g.run(args)
        for command in calls:
            if '--dump' not in command: continue
            self.assertIn('--prefill',command)
            self.assertEqual(command[command.index('--prefill')+1],'0')

    def test_named_bulk_baseline(self):
        import json
        args = self.arguments('named-bulk')
        args.prefill_rows = args.baseline_prefill_rows = 24
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            done = self.fake_engine(command, **kwargs)
            if '--prefill' in command and int(command[command.index('--prefill')+1]) > 0:
                dump = Path(command[command.index('--dump')+1])
                for name in g.NAMES:
                    path = dump/name
                    path.write_bytes(b'\x00\x00\x00\x40'*(path.stat().st_size//4))
            return done
        with patch.object(g.subprocess,'run',fake): g.run(args)
        baseline = next(command for command in calls if '--dump' in command and
                        Path(command[command.index('--dump')+1]).parent.name == 'baseline')
        self.assertEqual(baseline[baseline.index('--prefill')+1],'24')
        self.assertEqual(json.loads((args.output/'report.json').read_text())['baseline_prefill_rows'],24)

    def test_invalid_named_bulk_baseline_fails_before_gpu(self):
        for value, with_binary in ((-1,True),(True,True),(24,False)):
            args = self.arguments(f'bad-baseline-bulk-{value}-{with_binary}')
            args.baseline_prefill_rows = value
            if not with_binary:
                args.baseline_engine = None
                args.oracle_baseline = self.root/'missing-oracle.md'
            calls = []
            def fake(command, **kwargs):
                calls.append(command)
                return self.fake_engine(command, **kwargs)
            with patch.object(g.subprocess,'run',fake), self.assertRaises(ValueError): g.run(args)
            self.assertTrue(all(len(command)==1 for command in calls))

    def test_missing_live_bulk_rows_fail(self):
        args = self.arguments('broken-live-bulk')
        args.prefill_rows = 24
        def fake(command, **kwargs):
            done = self.fake_engine(command, **kwargs)
            if '--dump' in command:
                dump = Path(command[command.index('--dump')+1])
                if dump.parent.name == 'serial':
                    for name in g.NAMES: (dump/name).write_bytes(self.payload)
            return done
        with patch.object(g.subprocess, 'run', fake), self.assertRaises(ValueError): g.run(args)

    def test_oracle_baseline_table(self):
        path = self.root/'baseline.md'
        path.write_text(''.join(f'| {l} | 1e-4 | 1e-4 | 1e-4 |\n' for l in range(48)))
        self.assertEqual(len(g.baseline_points(path)), 144)
        for text in ('', '| 0 | 1e-4 | 1e-4 | 1e-4 |\n', path.read_text()*2):
            path.write_text(text)
            with self.assertRaises(ValueError): g.baseline_points(path)

    def test_published_baseline_tables(self):
        for file, mean in (('oracle-baseline.md', 1.560679514e-4),
                           ('oracle-baseline-405.md', 1.564899931e-4)):
            points = g.baseline_points(g.ROOT/'docs/oracle'/file)
            self.assertEqual(len(points), 144)
            self.assertAlmostEqual(sum(points.values())/144, mean, places=12)

    def test_live_oracle_invocations(self):
        path = self.root/'oracle-baseline.md'
        path.write_text(''.join(f'| {l} | 1e-4 | 1e-4 | 1e-4 |\n' for l in range(48)))
        commands = []
        def fake(command, **kwargs):
            commands.append(command)
            time.sleep(0.003)
            self.assertEqual(command[:4], ['flock','-w','7200','/tmp/crucible-oracle.lock'])
            layer = int(command[command.index('--pos')-1])
            text = ''.join(f'/dump: layer {layer} pos {p}: relL2 vs fp64 1e-4 (layer update)\n' for p in g.POSITIONS)
            kwargs['stdout'].write(text.encode())
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '8')
            return SimpleNamespace(returncode=0)
        with patch.object(g.subprocess, 'run', fake):
            evidence = g.oracle(self.root, self.dump, self.root/'prompt', path, Path('/python'), self.root)
        self.assertEqual(len(commands), 48)
        self.assertEqual(len(evidence['points']), 144)
        self.assertEqual(evidence['lock_path'], '/tmp/crucible-oracle.lock')
        self.assertAlmostEqual(evidence['mean_relL2'], 1e-4)
        with patch.object(g.subprocess, 'run', return_value=SimpleNamespace(returncode=7)), self.assertRaises(ValueError):
            g.oracle(self.root, self.dump, self.root/'prompt', path, Path('/python'), self.root)

    def test_mtp_prefix_candidate_role_only(self):
        import json
        for cap in (0, 8, 262144):
            args = self.arguments(f'cap-{cap}')
            args.mtp_prefix = cap
            def fake(command, **kwargs):
                if len(command) == 1:
                    return SimpleNamespace(returncode=2, stdout='--draft-head full|64k --mtp-prefix N', stderr='')
                return self.fake_engine(command, **kwargs)
            with patch.object(g.subprocess, 'run', fake): g.run(args)
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['mtp_prefix'], cap)
            self.assertTrue(report['full_head_capability']['mtp_prefix_capable'])
            for name, capture in report['captures'].items():
                command = capture['command']
                if name == 'baseline':
                    self.assertEqual(command[command.index('--mtp-prefix')+1],'0')
                    self.assertEqual(capture['draft_policy']['mtp_prefix'],0)
                else:
                    self.assertEqual(command[command.index('--mtp-prefix')+1], str(cap))
                    self.assertEqual(capture['draft_policy']['mtp_prefix'], cap)
            self.assertEqual(report['status'], 'PASS')

    def test_gate_tool_change_fails_before_pass(self):
        args=self.arguments('changed-gate-tool')
        digest=g.digest
        changed=False
        def fake_digest(path):
            return 'changed-tool' if changed and path==Path(g.__file__) else digest(path)
        def process(command,**kwargs):
            nonlocal changed
            result=self.fake_engine(command,**kwargs)
            if '--spec' in command and command[command.index('--spec')+1]=='2': changed=True
            return result
        with patch.object(g,'digest',side_effect=fake_digest),patch.object(g.subprocess,'run',process),\
                self.assertRaisesRegex(ValueError,'gate tool changed'): g.run(args)

    def test_runtime_environment_records_only_audited_knobs(self):
        args=self.arguments('env-provenance')
        with patch.dict(g.os.environ,{'CRUCIBLE_PF_PAD_H':'1','CRUCIBLE_PF_GDN_MICRO':'1',
                                       'HSA_USE_SVM':'0','GPU_PINNED_MIN_XFER_SIZE':'4096',
                                       'CRUCIBLE_STATE_CACHE':'1',
                                       'CRUCIBLE_PF_TILES':'1','CRUCIBLE_FUSE':'1',
                                       'CRUCIBLE_DECODE_RING':'1','CRUCIBLE_WARMUP':'1',
                                       'CRUCIBLE_PERSISTENT_GDN':'1',
                                       'CRUCIBLE_PF_EXACT_SPLIT':'1','CRUCIBLE_API_KEY':'test-secret-placeholder'},clear=True),\
                patch.object(g.subprocess,'run',self.fake_engine): g.run(args)
        import json
        report=json.loads((args.output/'report.json').read_text())
        for record in report['captures'].values():
            self.assertEqual(record['environment']['CRUCIBLE_PF_PAD_H'],'1')
            self.assertEqual(record['environment']['CRUCIBLE_PF_GDN_MICRO'],'1')
            self.assertEqual(record['environment']['HSA_USE_SVM'],'0')
            self.assertEqual(record['environment']['GPU_PINNED_MIN_XFER_SIZE'],'4096')
            self.assertEqual(record['environment']['CRUCIBLE_STATE_CACHE'],'1')
            for name in ('CRUCIBLE_PF_TILES','CRUCIBLE_FUSE','CRUCIBLE_DECODE_RING',
                         'CRUCIBLE_WARMUP','CRUCIBLE_PERSISTENT_GDN',
                         'CRUCIBLE_PF_EXACT_SPLIT'):
                self.assertEqual(record['environment'][name],'1')
            self.assertEqual(record['environment']['HSA_USERPTR_FOR_PAGED_MEM'],'0')
            self.assertNotIn('CRUCIBLE_API_KEY',record['environment'])
            self.assertIsNone(record['environment']['CRUCIBLE_TUNE_ROWS'])

    def test_train_runtime_environment_is_bound_to_primary_captures(self):
        names = ('CRUCIBLE_COMPACT_STATE',
                 'CRUCIBLE_QSA_CAPTURE_INDEX_ONLY',
                 'CRUCIBLE_HC_WMMA',
                 'CRUCIBLE_PF_PRUNE')
        args = self.arguments('train-env-provenance')
        with patch.dict(g.os.environ, dict.fromkeys(names, '1'), clear=True), \
                patch.object(g.subprocess, 'run', self.fake_engine):
            g.run(args)
        import json
        report = json.loads((args.output/'report.json').read_text())
        for role, record in report['captures'].items():
            for name in names:
                expected = '0' if role == 'baseline' and name == 'CRUCIBLE_COMPACT_STATE' else '1'
                self.assertEqual(record['environment'][name], expected)
        for name in names:
            self.assertIsNone(g.runtime_environment({})[name])

    def test_compact_baseline_process_uses_full_storage(self):
        args = self.arguments('compact-baseline-env')
        def process(command, **kwargs):
            if len(command) > 1:
                baseline = str((args.output / 'bin' / 'baseline').resolve()) in command
                self.assertEqual(kwargs['env']['CRUCIBLE_COMPACT_STATE'], '0' if baseline else '1')
                self.assertEqual(kwargs['env']['CRUCIBLE_PF_EXACT_SPLIT'], '1')
            return self.fake_engine(command, **kwargs)
        with patch.dict(g.os.environ, {'CRUCIBLE_COMPACT_STATE': '1', 'CRUCIBLE_PF_EXACT_SPLIT': '1'}, clear=True), \
                patch.object(g.subprocess, 'run', process):
            g.run(args)
            self.assertEqual(g.os.environ['CRUCIBLE_COMPACT_STATE'], '1')

    def test_implicit_mtp_cap_is_explicit_uncapped(self):
        import json
        args = self.arguments('implicit-mtp')
        def fake(command, **kwargs):
            if len(command)==1:
                return SimpleNamespace(returncode=2,stdout='--draft-head full|64k --mtp-prefix N',stderr='')
            return self.fake_engine(command,**kwargs)
        with patch.object(g.subprocess,'run',fake): g.run(args)
        report=json.loads((args.output/'report.json').read_text())
        for value in report['captures'].values():
            self.assertEqual(value['command'][value['command'].index('--mtp-prefix')+1],'0')
            self.assertEqual(value['draft_policy']['mtp_prefix'],0)

    def test_long_failure_keeps_primary_report_failed(self):
        import json
        import sys
        sys.path.insert(0,str(g.ROOT/'tools'))
        import long_capture,long_run
        args = self.arguments('long-failure')
        args.long_bitwise_ids_files = [self.root/'long.ids']
        def fail(*unused): raise ValueError('long failed')
        with patch.object(g.subprocess,'run',self.fake_engine),\
                patch.object(long_capture,'prepare',return_value={'status':'FAIL'}),\
                patch.object(long_run,'run',side_effect=fail),self.assertRaises(ValueError): g.run(args)
        report = json.loads((args.output/'report.json').read_text())
        self.assertEqual(report['status'],'FAIL')
        self.assertEqual(report['steps'],{'gate1':True,'gate2':True,'gate3':True})
        self.assertEqual(report['long']['status'],'FAIL')

    def test_resident_shard_flags_without_prompts_fail_before_gpu(self):
        for index,flags in enumerate(({'long_resident_shard':'control'},
                                      {'long_resident_shard':'all'},
                                      {'long_resident_16k_control_timeout':600})):
            args=self.arguments(f'shard-missing-input-{index}')
            args=SimpleNamespace(**{**vars(args),**flags})
            with patch.object(g.subprocess,'run',side_effect=self.fake_engine) as process, \
                    self.assertRaisesRegex(ValueError,'explicit.*prompt'):
                g.run(args)
            self.assertTrue(all(len(call.args[0])==1 for call in process.call_args_list))

    def test_resident_shard_with_prompts_but_no_probes_fails_before_gpu(self):
        args=self.arguments('shard-missing-probes')
        args.long_resident_shard='control'
        args.long_bitwise_ids_files=[g.ROOT/f'bench/suite/kl/code-{n}.ids' for n in (1024,4096,16384)]
        with patch.object(g.subprocess,'run',side_effect=self.fake_engine) as process, \
                self.assertRaisesRegex(ValueError,'both source-bound probes'):
            g.run(args)
        self.assertTrue(all(len(call.args[0])==1 for call in process.call_args_list))

    def test_candidate_cannot_be_its_own_baseline(self):
        import json
        for copied in (False, True):
            args = self.arguments(f'self-baseline-{copied}')
            if copied:
                args.baseline_engine.write_bytes(args.engine.read_bytes())
            else:
                args.baseline_engine = args.engine
            calls = []
            def fake(command, **kwargs):
                calls.append(command)
                return self.fake_engine(command, **kwargs)
            with patch.object(g.subprocess,'run',fake), self.assertRaises(ValueError): g.run(args)
            self.assertTrue(all(len(command)==1 for command in calls))
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['status'],'FAIL')
            self.assertEqual(report['captures'],{})

    def test_mtp_prefix_unsupported_and_invalid_fail_before_gpu(self):
        import json
        for number, value in enumerate((8, -1, 262145, 1.5, True)):
            args = self.arguments(f'invalid-cap-{number}')
            args.mtp_prefix = value
            calls = []
            def fake(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=2, stdout='--draft-head full|64k --mtp-prefix-internal N', stderr='')
            with patch.object(g.subprocess, 'run', fake), self.assertRaises(ValueError): g.run(args)
            self.assertTrue(all(len(command)==1 for command in calls))
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['captures'], {})

    def test_cli_mtp_prefix_bounds(self):
        args = self.arguments('cap-cli')
        base = [g.sys.executable, str(Path(g.__file__)), str(args.checkpoint),
                '--engine','/bin/true','--baseline-engine','/bin/true','--baseline-name','noop',
                '--ids',str(args.ids),'--gpu-run','/bin/true']
        for number in ('-1','262145','not-an-int'):
            done = g.subprocess.run([*base,'--mtp-prefix',number,'--output',str(args.output/number)],capture_output=True)
            self.assertNotEqual(done.returncode,0)
            self.assertNotIn(b'PASS',done.stdout)

    def test_cli_noop_is_nonzero(self):
        args = self.arguments('cli-noop')
        done = g.subprocess.run([g.sys.executable, str(Path(g.__file__)), str(args.checkpoint),
                                 '--engine', '/bin/true', '--baseline-engine', '/bin/true',
                                 '--baseline-name', 'noop', '--ids', str(args.ids),
                                 '--gpu-run', '/bin/true', '--output', str(args.output)], capture_output=True)
        self.assertEqual(done.returncode, 1)
        self.assertNotIn(b'PASS', done.stdout)
        import json
        self.assertEqual(json.loads((args.output/'report.json').read_text())['status'], 'FAIL')

    def test_oracle_parser(self):
        text = ''.join(f'/dump: layer 3 pos {p}: relL2 vs fp64 1.000e-04 (layer update)\n' for p in g.POSITIONS)
        self.assertEqual(len(g.parse_oracle(text, 3)), 3)
        for bad in ('PASS', text + text, text.replace('1.000e-04', 'nan'), text.replace('layer 3', 'layer 2')):
            with self.assertRaises(ValueError): g.parse_oracle(bad, 3)


if __name__ == '__main__': unittest.main()
