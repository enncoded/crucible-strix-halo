"""CPU tests: the memory floor at the start of each GPU lease.
A lease that starts right after another one can see 74-118 GB free for a while. lease_runner.py and lease_steps.sh
poll MemAvailable (tools/mem_floor.sh) before the first engine starts, and exit 76 when the floor does not come."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g
import test_gates_lease as lease


def meminfo(path, kb):
    path.write_text(f'MemTotal: 131072000 kB\nMemAvailable: {kb} kB\n')


class LeaseRunnerMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mem = self.root / 'meminfo'
        self.marker = self.root / 'ran'

    def plan(self, **extra):
        plan = {'budget_s': 60, 'jobs': [{'name': 'a', 'cmd': ['touch', str(self.marker)], 'timeout_s': 30,
                                          'stdout': str(self.root / 'a.out'), 'stderr': str(self.root / 'a.err')}],
                **extra}
        path = self.root / 'plan.json'
        path.write_text(json.dumps(plan))
        return path

    def run_runner(self, plan):
        env = {**os.environ, 'MEM_FLOOR_MEMINFO': str(self.mem)}
        return subprocess.run([sys.executable, str(ROOT / 'tools/lease_runner.py'), str(plan), str(self.root / 'r.json')],
                              env=env, capture_output=True, text=True, timeout=60)

    def test_below_the_floor_no_engine_starts_and_the_lease_exits_76(self):
        meminfo(self.mem, 74_000_000)
        run = self.run_runner(self.plan(mem_floor_gb=126, mem_wait_s=1, mem_poll_s=0.2))
        self.assertEqual(run.returncode, 76, run)
        self.assertFalse(self.marker.exists())
        result = json.loads((self.root / 'r.json').read_text())
        self.assertEqual(result['jobs'], [])
        self.assertFalse(result['memory']['ok'])
        self.assertIn('below: no start', result['memory']['log'])

    def test_memory_that_comes_inside_the_wait_runs_the_jobs(self):
        meminfo(self.mem, 109_922_876)
        timer = threading.Timer(0.6, meminfo, (self.mem, 128_400_000))
        timer.start()
        run = self.run_runner(self.plan(mem_floor_gb=126, mem_wait_s=30, mem_poll_s=0.1))
        timer.join()
        self.assertEqual(run.returncode, 0, run)
        self.assertTrue(self.marker.exists())
        result = json.loads((self.root / 'r.json').read_text())
        self.assertTrue(result['memory']['ok'])
        self.assertEqual(result['jobs'][0]['status'], 'done')

    def test_a_plan_without_a_floor_runs_at_once(self):
        meminfo(self.mem, 100)
        run = self.run_runner(self.plan())
        self.assertEqual(run.returncode, 0, run)
        self.assertTrue(self.marker.exists())
        self.assertNotIn('memory', json.loads((self.root / 'r.json').read_text()))


class GatesMemoryRetryTests(unittest.TestCase):
    """gates.py puts the floor into each lease plan, adds the wait to the lease timeout and runs a lease that
    ended with 76 (no capture ran) again, until its queue wait has passed."""
    writable, engine, arguments = lease.LeaseGateTests.writable, lease.LeaseGateTests.engine, \
        lease.LeaseGateTests.arguments

    def setUp(self):
        lease.LeaseGateTests.setUp(self)
        self.low_leases = 1
        self.plans = []
        self.commands = []

    def subprocess_run(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        runner = command.index(str(g.ROOT / 'tools/lease_runner.py'))
        self.commands.append(command[:runner])
        plan = json.loads(Path(command[runner + 1]).read_text())
        self.plans.append(plan)
        result_path = Path(command[runner + 2])
        if self.low_leases:
            self.low_leases -= 1
            result_path.write_text(json.dumps({'jobs': [], 'memory': {'ok': False, 'log': 'below: no start'}}))
            return SimpleNamespace(returncode=76)
        self.clock = lease.Clock()
        lease.lr.run_plan(plan, result_path, self.engine, self.clock)
        return SimpleNamespace(returncode=0)

    def gate(self, name, **extra):
        args = self.arguments(name, **extra)
        with patch.object(g.subprocess, 'run', self.subprocess_run):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def test_each_lease_waits_for_the_floor_inside_and_its_timeout_holds_the_wait(self):
        self.low_leases = 0
        report = self.gate('floor')
        self.assertEqual(report['status'], 'PASS')
        for plan in self.plans:
            self.assertEqual((plan['mem_floor_gb'], plan['mem_wait_s'], plan['mem_poll_s']), (126, 300, 5))
        for command in self.commands:
            self.assertEqual(command[-5:-1], ['timeout', '-k', '10', str(g.LEASE_LIMIT_S + g.LEASE_MEM_WAIT_S)])

    def test_a_lease_below_the_floor_runs_again(self):
        report = self.gate('retry')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['leases'][0]['returncode'], 76)
        self.assertEqual(len(report['leases']), 2)

    def test_the_retries_end_at_the_queue_wait(self):
        self.low_leases = 10 ** 6
        args = self.arguments('starve', queue_wait=60)
        clock = itertools.count(0, 40 * 10 ** 9)  # each call 40 s later
        with patch.object(g.subprocess, 'run', self.subprocess_run), patch.object(g.time, 'monotonic_ns',
                                                                                   lambda: next(clock)):
            with self.assertRaisesRegex(ValueError, 'memory floor'):
                g.run(args)
        self.assertLessEqual(len(self.plans), 3)


if __name__ == '__main__':
    unittest.main()
