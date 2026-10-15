"""CPU tests: gates.py --queue-wait reaches every gpu-run ticket and report.json; the default stays 7200 s."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g
import test_gates_lease as lease


class QueueWaitTests(unittest.TestCase):
    # The lease tests' fixture: fake checkpoint, prompt, baseline binary and engine.
    writable, engine, arguments = lease.LeaseGateTests.writable, lease.LeaseGateTests.engine, lease.LeaseGateTests.arguments

    def setUp(self):
        lease.LeaseGateTests.setUp(self)
        self.tickets = []

    def subprocess_run(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        self.tickets.append(command[:5])
        runner = str(g.ROOT / 'tools/lease_runner.py')
        if runner in command:
            index = command.index(runner)
            plan = json.loads(Path(command[index + 1]).read_text())
            self.clock = lease.Clock()
            lease.lr.run_plan(plan, Path(command[index + 2]), self.engine, self.clock)
            return SimpleNamespace(returncode=0)
        # One ticket per capture: the engine command follows the queue prefix.
        job = {'cmd': command[10:], 'stdout': kwargs['stdout'].name, 'stderr': kwargs['stderr'].name}
        self.clock = lease.Clock()
        return SimpleNamespace(returncode=self.engine(job, {}, 300))

    def gate(self, name, **extra):
        args = self.arguments(name, **extra)
        with patch.object(g.subprocess, 'run', self.subprocess_run):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def test_lease_batch_tickets_use_the_wait(self):
        report = self.gate('leases', queue_wait=21600)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(report['queue_wait_s'], 21600)
        self.assertEqual(self.tickets, [['/queue', '-p', '1', '-w', '21600']])

    def test_one_ticket_per_capture_uses_the_wait(self):
        report = self.gate('single', queue_wait=21600, lease_batch=False)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(len(self.tickets), 6)
        self.assertTrue(all(ticket == ['/queue', '-p', '1', '-w', '21600'] for ticket in self.tickets))
        for capture in report['captures'].values():
            self.assertEqual(capture['queued_command'][3:5], ['-w', '21600'])

    def test_default_stays_7200(self):
        report = self.gate('default')
        self.assertEqual(report['queue_wait_s'], 7200)
        self.assertEqual(self.tickets, [['/queue', '-p', '1', '-w', '7200']])

    def test_invalid_wait_fails_before_gpu_work(self):
        for value in (0, -5, 1.5, '600'):
            args = self.arguments(f'bad-{value}', queue_wait=value)
            with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'queue-wait'):
                g.run(args)
        self.assertEqual(self.tickets, [])

    def test_cli_flag(self):
        argv = ['gates', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', 'o', '--baseline-name', 'b',
                '--baseline-engine', 'x']
        for extra, value in (([], 7200), (['--queue-wait', '21600'], 21600)):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run') as execute, \
                    patch.object(g, 'prune_dumps', return_value={'files': 0, 'bytes': 0}):
                self.assertEqual(g.main(), 0)
            self.assertEqual(execute.call_args.args[0].queue_wait, value)


if __name__ == '__main__':
    unittest.main()
