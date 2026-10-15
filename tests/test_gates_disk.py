"""CPU tests for the gate disk preflight and the dump prune after PASS."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g

GB = 10**9


def usage(free):
    return SimpleNamespace(total=2000 * GB, used=2000 * GB - free, free=free)


class DiskPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_low_space_fails_closed_with_clear_message(self):
        with patch.object(g.shutil, 'disk_usage', return_value=usage(149 * GB)) as du:
            with self.assertRaisesRegex(ValueError, r'149\.0 GB free.*150 GB'):
                g.disk_preflight(self.root / 'new' / 'out')
        # The check uses the nearest existing ancestor of a new output path.
        self.assertEqual(Path(du.call_args.args[0]), self.root)

    def test_enough_space_passes(self):
        with patch.object(g.shutil, 'disk_usage', return_value=usage(150 * GB)):
            self.assertEqual(g.disk_preflight(self.root / 'out'), 150 * GB)

    def test_run_stops_before_any_capture(self):
        args = SimpleNamespace(output=self.root / 'gate')
        with patch.object(g.shutil, 'disk_usage', return_value=usage(10 * GB)):
            with self.assertRaisesRegex(ValueError, 'free'):
                g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('disk preflight', report['error'])
        self.assertEqual(sorted(p.name for p in args.output.iterdir()), ['report.json'])


class PruneDumpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'gate'

    def make(self, status='PASS'):
        captures = {}
        for cap in ('candidate', 'baseline'):
            d = self.out / cap / 'dump'
            d.mkdir(parents=True)
            dumps = {}
            for name in ('h_00.f32', 'h_01.f32'):
                p = d / name
                p.write_bytes((cap + name).encode() * 4)
                info = p.stat()
                dumps[name] = {'bytes': info.st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                               'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}
            captures[cap] = {'dumps': dumps}
        (self.out / 'report.json').write_text(json.dumps({'status': status, 'captures': captures}))

    def files(self):
        return sorted(p.name for p in self.out.rglob('*.f32'))

    def test_pass_deletes_recorded_dumps_and_records_it(self):
        self.make()
        g.prune_dumps(self.out)
        self.assertEqual(self.files(), [])
        report = json.loads((self.out / 'report.json').read_text())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['dumps_pruned']['files'], 4)
        self.assertGreater(report['dumps_pruned']['bytes'], 0)
        # The hashes stay in the report after the files go.
        self.assertEqual(len(report['captures']['candidate']['dumps']['h_00.f32']['sha256']), 64)

    def test_fail_keeps_dumps(self):
        self.make(status='FAIL')
        with self.assertRaisesRegex(ValueError, 'PASS'):
            g.prune_dumps(self.out)
        self.assertEqual(len(self.files()), 4)

    def test_changed_file_keeps_every_dump(self):
        self.make()
        (self.out / 'baseline/dump/h_01.f32').write_bytes(b'changed after the report')
        with self.assertRaisesRegex(ValueError, 'changed'):
            g.prune_dumps(self.out)
        self.assertEqual(len(self.files()), 4)

    def test_main_prunes_on_pass_unless_kept(self):
        argv = ['gates.py', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', str(self.out),
                '--baseline-name', 'b', '--baseline-engine', 'x']
        for extra, calls in (([], 1), (['--keep-dumps'], 0)):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run'), \
                    patch.object(g, 'prune_dumps', return_value={'files': 4, 'bytes': 8}) as prune:
                self.assertEqual(g.main(), 0)
            self.assertEqual(prune.call_count, calls)

    def test_main_keeps_dumps_on_fail(self):
        argv = ['gates.py', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', str(self.out),
                '--baseline-name', 'b', '--baseline-engine', 'x']
        with patch.object(sys, 'argv', argv), patch.object(g, 'run', side_effect=ValueError('bad')), \
                patch.object(g, 'prune_dumps') as prune:
            self.assertEqual(g.main(), 1)
        prune.assert_not_called()


if __name__ == '__main__':
    unittest.main()
