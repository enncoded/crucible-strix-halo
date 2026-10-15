"""CPU tests: --lease-batch fits a bulk gate in one lease, and a checked fresh baseline survives a failed lease."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools'))
import baseline_cache as bc
import gates as g
import lease_runner as lr
import test_gates_lease as lease

# Process seconds of one bulk305 gate in separate mode (the test machine, Oct 7, capture-once/results/g/bulk305).
BULK305 = {'candidate': 37, 'baseline': 37, 'chunk1': 76, 'chunk8': 34, 'serial': 44, 'spec2': 45}
# The same for bulk405 (capture-once/results/g/bulk405 and bulk405-jobs).
BULK405 = {'candidate': 66, 'baseline': 75, 'chunk1': 100, 'chunk8': 75, 'serial': 80, 'spec2': 75}
PLATFORM = {'rocm_version': 'test', 'kernel': 'test', 'firmware': {}, 'libraries': {}}


class LeaseFitTests(unittest.TestCase):
    writable, arguments = lease.LeaseGateTests.writable, lease.LeaseGateTests.arguments

    def setUp(self):
        lease.LeaseGateTests.setUp(self)
        self.seconds = {}
        self.limits = []
        self.expire = set()  # lease indexes whose queue wait expires (gpu-run exit 75)
        patcher = patch.object(bc, 'platform_identity', return_value=PLATFORM)
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self, job, env, limit):
        name = Path(job['stdout']).parent.name
        self.job_seconds = self.seconds.get(name, 10)
        return lease.LeaseGateTests.engine(self, job, env, limit)

    def subprocess_run(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        runner = command.index(str(g.ROOT / 'tools/lease_runner.py'))
        self.limits.append(command[runner - 2])
        plan = json.loads(Path(command[runner + 1]).read_text())
        if len(self.leases) in self.expire:
            self.leases.append([])
            return SimpleNamespace(returncode=75)
        self.leases.append([job['name'] for job in plan['jobs']])
        self.clock = lease.Clock()
        result = lr.run_plan(plan, Path(command[runner + 2]), self.engine, self.clock)
        self.leases[-1] = [job['name'] for job in result['jobs']]
        return SimpleNamespace(returncode=0)

    def gate(self, name, **extra):
        args = self.arguments(name, **extra)
        with patch.object(g.subprocess, 'run', self.subprocess_run):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def test_a_bulk305_gate_fits_one_lease(self):
        # chunk1 runs a 1024-row prompt one row at a time: about 1.7 times the other captures. It runs
        # last, with an estimate of 45 x 1.25 x 1.4 + 5 = 84 s, and 88 s remain after the other five.
        self.seconds = BULK305
        report = self.gate('bulk305')
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(self.leases, [['candidate', 'baseline', 'chunk8', 'serial', 'spec2', 'chunk1']])
        # The checks keep the gate order.
        self.assertEqual(list(report['captures']), ['candidate', 'baseline', 'chunk1', 'chunk8', 'serial', 'spec2'])

    def test_a_bulk305_gate_with_a_cached_baseline_fits_one_lease(self):
        self.seconds = BULK305
        self.gate('fill', baseline_cache=self.root / 'cache', baseline_recheck=0)
        self.leases = []
        report = self.gate('bulk305', baseline_cache=self.root / 'cache', baseline_recheck=0)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(self.leases, [['candidate', 'chunk8', 'serial', 'spec2', 'chunk1']])

    def test_lease_seconds_lets_a_bulk405_gate_fit_one_lease(self):
        self.seconds = BULK405
        report = self.gate('bulk405-300')
        # With 300 s: candidate, baseline, chunk8 | serial, spec2 | chunk1 (its estimate is 145 s, 130 s remain).
        self.assertEqual(len(self.leases), 3)
        self.assertEqual(report['lease_seconds'], 300)
        self.leases, self.limits = [], []
        report = self.gate('bulk405-600', lease_seconds=600)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(len(self.leases), 1)
        self.assertEqual(self.limits, ['900'])  # the 600 s lease and the 300 s memory wait
        self.assertEqual(report['lease_seconds'], 600)
        self.assertEqual(json.loads((self.root / 'bulk405-600/leases/0.plan.json').read_text())['budget_s'], 585)

    def test_chunk1_carries_a_scaled_estimate(self):
        self.seconds = BULK305
        self.gate('plan')
        plan = json.loads((self.root / 'plan/leases/0.plan.json').read_text())
        self.assertEqual({job['name']: job.get('estimate_scale', 1) for job in plan['jobs']},
                         {'candidate': 1, 'baseline': 1, 'chunk8': 1, 'serial': 1, 'spec2': 1, 'chunk1': 1.4})

    def test_a_slower_lease_defers_chunk1_instead_of_cutting_it(self):
        # 10% slower than BULK305 with a fresh baseline: 68 s remain, and chunk1 needs 84 s. Without the
        # scale its estimate (67.5 s) would start it, and the lease budget would cut it.
        self.seconds = {name: round(value * 1.1) for name, value in BULK305.items()}
        report = self.gate('slower')
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(self.leases, [['candidate', 'baseline', 'chunk8', 'serial', 'spec2'], ['chunk1']])
        self.assertTrue(all(job['status'] == 'done' for lease in report['leases'] for job in lease['jobs']))

    def test_lease_seconds_need_lease_batch(self):
        args = self.arguments('no-batch', lease_seconds=600, lease_batch=False)
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, '--lease-batch'):
            g.run(args)
        self.assertEqual(self.leases, [])

    def test_lease_seconds_stay_within_the_lease_rule(self):
        for value in (299, 901, 600.5, '600'):
            args = self.arguments(f'bad-{value}', lease_seconds=value)
            with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'lease-seconds'):
                g.run(args)
        self.assertEqual(self.leases, [])

    def test_a_checked_fresh_baseline_is_stored_when_a_later_lease_fails(self):
        cache = self.root / 'cache'
        self.seconds = {'candidate': 100, 'baseline': 100}
        self.expire = {1}
        args = self.arguments('expired', baseline_cache=cache, baseline_recheck=0)
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'lease 1 exited 75'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['baseline_cache']['stored'], True)
        # The same lease record as a capture that take() finishes.
        self.assertEqual(set(report['captures']['baseline']['lease']), {'index', 'timeout_s', 'process_started_ns', 'seconds'})
        self.assertEqual(self.leases[0], ['candidate', 'baseline'])
        # The next gate reuses it and needs no baseline capture.
        self.expire, self.seconds, self.leases = set(), {}, []
        report = self.gate('reuse', baseline_cache=cache, baseline_recheck=0)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(report['baseline_source'], 'cache')
        self.assertNotIn('baseline', self.leases[0])

    def test_a_failed_baseline_is_not_stored(self):
        cache = self.root / 'cache'
        self.seconds = {'candidate': 100, 'baseline': 100}
        self.expire = {1}
        self.failing = {'baseline': 3}
        args = self.arguments('baseline-failed', baseline_cache=cache, baseline_recheck=0)
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaises(ValueError):
            g.run(args)
        self.assertFalse(any((cache / 'entries').glob('[0-9a-f]*')) if (cache / 'entries').exists() else False)

    def test_cli_flag(self):
        argv = ['gates', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', 'o', '--baseline-name', 'b',
                '--baseline-engine', 'x']
        for extra, value in (([], 300), (['--lease-seconds', '600'], 600)):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run') as execute, \
                    patch.object(g, 'prune_dumps', return_value={'files': 0, 'bytes': 0}):
                self.assertEqual(g.main(), 0)
            self.assertEqual(execute.call_args.args[0].lease_seconds, value)


if __name__ == '__main__':
    unittest.main()
