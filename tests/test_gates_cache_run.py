"""CPU tests: gates.py with --baseline-cache fills, reuses, refuses and rechecks baseline captures."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import baseline_cache as bc
import gates as g

PLATFORM = {'rocm_version': '7.2.4', 'kernel': 'test', 'firmware': {}, 'libraries': {}}


class CachedGateRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.writable)
        self.root = Path(self.temp.name)
        for patcher in (patch.object(g, 'disk_preflight', return_value=0),
                        patch.object(bc, 'platform_identity', return_value=PLATFORM)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.payload = b'\x00\x00\x80\x3f' * g.WIDTH
        self.cache = self.root / 'cache'
        self.baseline_payload = self.payload
        self.calls = []

    def writable(self):
        for path in self.root.rglob('*'):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)

    def arguments(self, name, **extra):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir(exist_ok=True)
        for file in ('config.json', 'quantization_config.json', 'model.safetensors.index.json'):
            (checkpoint / file).write_text('{}')
        prompt = self.root / 'prompt.ids'
        prompt.write_text('1\n')
        baseline = self.root / 'baseline-engine'
        if not baseline.exists():
            baseline.write_bytes(Path('/bin/true').read_bytes() + b'baseline-test-marker')
        values = dict(output=self.root / name, engine=Path('/bin/true'), checkpoint=checkpoint, ids=prompt,
                      baseline_name='named-main', baseline_engine=baseline, oracle_baseline=None, tokens=3,
                      prefill_rows=0, gpu_run=Path('/queue'), priority=3, engine_build_record=None,
                      baseline_cache=self.cache, baseline_recheck=0)
        return SimpleNamespace(**{**values, **extra})

    def fake_engine(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        time.sleep(0.003)
        def option(flag): return command[command.index(flag) + 1]
        dump = Path(option('--dump'))
        self.calls.append(dump.parent.name)
        n, spec = int(option('--n')), int(option('--spec'))
        rows = 1 + (n - 1 if spec == 0 else 0)
        payload = self.baseline_payload if dump.parent.name == 'baseline' else self.payload
        for name in g.NAMES:
            (dump / name).write_bytes(payload * rows)
        kwargs['stdout'].write(('ids:' + ' 7' * n + '\n').encode())
        return SimpleNamespace(returncode=0)

    def gate(self, name, **extra):
        args = self.arguments(name, **extra)
        self.calls = []
        with patch.object(g.subprocess, 'run', self.fake_engine):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def failed_gate(self, name, pattern, **extra):
        args = self.arguments(name, **extra)
        self.calls = []
        with patch.object(g.subprocess, 'run', self.fake_engine), self.assertRaisesRegex(ValueError, pattern):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def test_first_run_fills_and_second_reuses_with_the_same_verdict(self):
        first = self.gate('first')
        self.assertIn('baseline', self.calls)
        self.assertEqual(first['status'], 'PASS')
        self.assertEqual(first['baseline_source'], 'fresh')
        self.assertTrue(first['baseline_cache']['stored'])
        key = first['baseline_cache']['key']
        # The key holds the sha of the pinned copy that actually ran.
        self.assertEqual(first['baseline_cache']['inputs']['binary_sha256'],
                         first['pinned_binaries']['baseline']['sha256'])
        self.assertEqual(first['captures']['baseline']['command'][0], first['pinned_binaries']['baseline']['path'])
        self.assertTrue((self.cache / 'entries' / key / 'manifest.json').is_file())
        second = self.gate('second')
        self.assertNotIn('baseline', self.calls)
        self.assertEqual(second['status'], 'PASS')
        self.assertEqual(second['steps'], first['steps'])
        self.assertEqual(second['baseline_source'], 'cache')
        self.assertEqual(second['baseline_cache']['key'], key)
        self.assertNotIn('baseline', second['captures'])
        self.assertEqual(second['baseline_cache']['provenance']['gate_output'], str(self.root / 'first'))
        self.assertEqual(second['baseline_cache']['provenance']['started_ns'],
                         first['captures']['baseline']['started_ns'])
        sha = lambda report, name: {n: v['sha256'] for n, v in report['captures'][name]['dumps'].items()}
        for name in ('candidate', 'chunk1', 'chunk8', 'serial', 'spec2'):
            self.assertEqual(sha(second, name), sha(first, name))
            self.assertEqual(second['captures'][name]['output_ids'], first['captures'][name]['output_ids'])
        self.assertEqual(second['baseline_cache']['dumps'], sha(first, 'baseline'))

    def test_default_run_has_no_cache_fields(self):
        report = self.gate('plain', baseline_cache=None)
        self.assertNotIn('baseline_source', report)
        self.assertNotIn('baseline_cache', report)
        self.assertFalse(self.cache.exists())

    def test_corrupt_entry_is_refused_and_captured_fresh(self):
        key = self.gate('fill')['baseline_cache']['key']
        self.writable()
        path = self.cache / 'entries' / key / 'dump' / g.NAMES[5]
        path.chmod(0o644)
        path.write_bytes(b'\x00' * len(self.payload))
        report = self.gate('after-corruption')
        self.assertIn('baseline', self.calls)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['baseline_source'], 'fresh')
        self.assertIn('h_05.f32', report['baseline_cache']['refused'])
        self.assertTrue(report['baseline_cache']['stored'])
        self.assertEqual(self.gate('refilled')['baseline_source'], 'cache')

    def test_a_cached_baseline_still_fails_a_different_candidate(self):
        self.gate('fill')
        self.payload = b'\x00\x00\x00\x40' * g.WIDTH
        report = self.failed_gate('changed-candidate', 'differs')
        self.assertEqual(report['baseline_source'], 'cache')
        self.assertNotIn('gate1', report['steps'])

    def test_changed_environment_misses(self):
        key = self.gate('fill')['baseline_cache']['key']
        with patch.dict(g.os.environ, {'CRUCIBLE_FUSE': '7'}):
            report = self.gate('other-env')
        self.assertEqual(report['baseline_source'], 'fresh')
        self.assertNotEqual(report['baseline_cache']['key'], key)

    def test_recheck_match_passes_and_records_it(self):
        self.gate('fill')
        report = self.gate('recheck', baseline_recheck=1)
        self.assertIn('baseline', self.calls)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['baseline_source'], 'cache')
        self.assertEqual(report['baseline_cache']['recheck'], {'every': 1, 'index': 0, 'due': True, 'match': True})

    def test_recheck_drift_fails_loudly_and_retires_the_entry(self):
        key = self.gate('fill')['baseline_cache']['key']
        self.baseline_payload = b'\x00\x00\x00\x40' * g.WIDTH
        report = self.failed_gate('drift', 'baseline cache drift', baseline_recheck=1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertFalse(report['baseline_cache']['recheck']['match'])
        self.assertFalse((self.cache / 'entries' / key).exists())

    def test_cli_flags(self):
        argv = ['gates', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', 'o', '--baseline-name', 'b',
                '--baseline-engine', 'x']
        for extra, cache, every in (([], None, 0), (['--baseline-cache'], g.BASELINE_CACHE_DIR, 0),
                                    (['--baseline-cache', '/c', '--baseline-recheck', '5'], Path('/c'), 5)):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run') as execute, \
                    patch.object(g, 'prune_dumps', return_value={'files': 0, 'bytes': 0}):
                self.assertEqual(g.main(), 0)
            self.assertEqual(execute.call_args.args[0].baseline_cache, cache)
            self.assertEqual(execute.call_args.args[0].baseline_recheck, every)

    def test_cache_needs_a_named_baseline(self):
        args = self.arguments('oracle', baseline_engine=None, oracle_baseline=self.root / 'o.md')
        (self.root / 'o.md').write_text('x')
        with patch.object(g.subprocess, 'run', self.fake_engine), \
                self.assertRaisesRegex(ValueError, 'baseline cache needs a named baseline binary'):
            g.run(args)


if __name__ == '__main__':
    unittest.main()
