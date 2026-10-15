"""CPU tests: gates.py job mode (gen CKPT --jobs FILE): one engine load per --ctx and --prefill, with canaries."""
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

USAGE = ('usage: crucible gen CKPT --ids a,b,c [--draft-head full|64k] [--mtp-prefix N]\n'
         '       crucible gen CKPT --jobs FILE   (one load)\n')


class JobModeGateTests(unittest.TestCase):
    # The lease tests' fixture: fake checkpoint, prompt, baseline binary and single-capture engine.
    writable, engine, gate = lease.LeaseGateTests.writable, lease.LeaseGateTests.engine, lease.LeaseGateTests.gate

    def setUp(self):
        lease.LeaseGateTests.setUp(self)
        self.usage = USAGE
        self.processes = []
        self.leak = None  # a capture name whose second run in a process writes other values
        self.stop_after = None  # (process index, job index): the engine exits 1 after that job
        self.queue_wait = '7200'
        self.lease_limit = '600'  # 300 s and the memory wait (gates.LEASE_MEM_WAIT_S)

    def arguments(self, name, **extra):
        return lease.LeaseGateTests.arguments(self, name, **{'lease_batch': False, 'baseline_recheck': 0, **extra})

    def jobs_process(self, job, env, limit):
        command = job['cmd']
        self.assertEqual(command[1:3], ['gen', command[2]])
        self.assertEqual(command[3], '--jobs')
        self.assertEqual(len(command), 5)
        lines = [json.loads(line) for line in Path(command[4]).read_text().splitlines()]
        process = len(self.processes)
        self.processes.append([Path(line['stdout']).parent.name for line in lines])
        seen, out = set(), []
        for index, line in enumerate(lines):
            key = tuple(a for a in line['args'] if not a.endswith('/dump'))
            single = {'cmd': [command[0], 'gen', command[2], *line['args']], 'stdout': line['stdout'],
                      'stderr': line['stderr']}
            self.clock.now -= self.job_seconds  # one load per process: count the time once below
            code = self.engine(single, env, limit)
            if self.leak and key in seen and self.leak in line['stdout']:
                dump = Path(line['args'][line['args'].index('--dump') + 1])
                (dump / 'h_07.f32').write_bytes(b'\x01' * (dump / 'h_07.f32').stat().st_size)
            seen.add(key)
            if code:
                return code
            out.append(f'job {index} done 0.100 s\n')
            Path(job['stdout']).write_text(''.join(out))
            if self.stop_after == (process, index):
                return 1
        self.clock.now += self.job_seconds
        return 0

    def run_unit(self, job, env, limit):
        if '--jobs' in job['cmd']:
            return self.jobs_process(job, env, limit)
        return self.engine(job, env, limit)

    def subprocess_run(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout=self.usage, stderr='')
        if '--dump' in command or '--jobs' in command:
            raise AssertionError('job mode must not queue an engine process by itself')
        runner = command.index(str(g.ROOT / 'tools/lease_runner.py'))
        self.assertEqual(command[:runner - 1], ['/queue', '-p', '1', '-w', self.queue_wait, '--', 'timeout', '-k', '10',
                                                self.lease_limit])
        plan = json.loads(Path(command[runner + 1]).read_text())
        self.leases.append([job['name'] for job in plan['jobs']])
        self.clock = lease.Clock()
        lr.run_plan(plan, Path(command[runner + 2]), self.run_unit, self.clock)
        return SimpleNamespace(returncode=0)

    def test_one_process_per_load_with_a_canary_first_and_last(self):
        report = self.gate('dec')
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(self.leases, [['jobs0', 'baseline']])
        # The Gate 1 capture runs first on a fresh load, never after another job on the same prompt.
        self.assertEqual(self.processes, [['candidate', 'chunk1', 'chunk8', 'serial', 'spec2', 'canary_candidate']])
        mode = report['capture_mode']
        self.assertEqual((mode['requested'], mode['used'], mode['candidate_jobs_capable']), ('auto', 'jobs', True))
        self.assertEqual(mode['processes'][0]['load'], {'ctx': '32768', 'prefill': '0'})
        self.assertEqual(mode['processes'][0]['returncode'], 0)
        self.assertEqual(mode['canaries'], [{'process': 'jobs0', 'source': 'candidate', 'canary': 'canary_candidate',
                                             'match': True}])
        for position, name in enumerate(mode['processes'][0]['jobs']):
            process = report['captures'][name]['process']
            self.assertEqual((process['name'], process['job_index'], process['job_done']), ('jobs0', position, True))
        self.assertNotIn('process', report['captures']['baseline'])
        # Each job line carries the same options as the separate command that the report records.
        lines = [json.loads(line) for line in Path(mode['processes'][0]['jobs_file']).read_text().splitlines()]
        self.assertEqual(lines[0]['args'], report['captures']['candidate']['command'][3:])
        self.assertEqual(g.digest(Path(mode['processes'][0]['jobs_file'])), mode['processes'][0]['jobs_sha256'])

    def test_the_candidate_is_the_first_job_of_its_process_in_every_gate(self):
        # A kernel that reads scratch before it writes it could pass on values from an earlier job.
        # Gate 1 compares the candidate capture with the baseline, so it must run on a fresh load.
        for name, extra in (('dec', {}), ('bulk', {'prefill_rows': 4096, 'baseline_prefill_rows': 4096})):
            self.processes = []
            self.gate(f'first-{name}', **extra)
            with_candidate = [order for order in self.processes if 'candidate' in order]
            self.assertEqual(len(with_candidate), 1)
            self.assertEqual((with_candidate[0][0], with_candidate[0][-1]), ('candidate', 'canary_candidate'))

    def test_bulk_prefill_splits_the_candidate_into_two_loads(self):
        report = self.gate('bulk', prefill_rows=4096, baseline_prefill_rows=4096)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(self.processes, [['candidate', 'serial', 'spec2', 'canary_candidate'],
                                          ['chunk8', 'chunk1', 'canary_chunk8']])
        self.assertEqual([p['load']['prefill'] for p in report['capture_mode']['processes']], ['4096', '0'])
        self.assertEqual(len(report['capture_mode']['canaries']), 2)

    def test_a_canary_that_differs_fails_the_gate(self):
        self.leak = 'canary_candidate'
        args = self.arguments('leak')
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'state leaked'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')

    def test_an_engine_process_failure_fails_the_gate(self):
        self.stop_after = (0, 2)
        args = self.arguments('stop')
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'engine failed'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['capture_mode']['processes'][0]['returncode'], 1)

    def test_separate_mode_keeps_one_process_per_capture(self):
        report = self.gate('separate', capture_mode='separate', lease_batch=True)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(self.processes, [])
        self.assertEqual(report['capture_mode']['used'], 'separate')
        self.assertEqual(self.leases, [['candidate', 'baseline', 'chunk8', 'serial', 'spec2', 'chunk1']])

    def test_an_engine_without_jobs_keeps_separate_processes(self):
        self.usage = USAGE.splitlines()[0]
        report = self.gate('old', lease_batch=True)
        self.assertEqual(report['capture_mode'], {'requested': 'auto', 'used': 'separate',
                                                  'candidate_jobs_capable': False})
        self.assertEqual(self.processes, [])
        args = self.arguments('forced', capture_mode='jobs')
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, '--jobs'):
            g.run(args)

    def test_cache_hit_runs_only_the_candidate_processes(self):
        cache = self.root / 'cache'
        patcher = patch.object(bc, 'platform_identity',
                               return_value={'rocm_version': 'test', 'kernel': 'test', 'firmware': {}, 'libraries': {}})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gate('fill', baseline_cache=cache)
        report = self.gate('hit', baseline_cache=cache)
        self.assertEqual(report['baseline_source'], 'cache')
        self.assertEqual(self.leases[-1], ['jobs0'])
        self.assertEqual(report['status'], 'PASS')

    def test_budget_cut_reruns_the_whole_process_in_fresh_directories(self):
        self.job_seconds = 200
        self.failing = {'chunk8': 124}
        report = self.gate('retry')
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.leases[1][0], 'jobs0')

    def test_ablated_self_repeat_keeps_its_own_process(self):
        with patch.dict('os.environ', {'CRUCIBLE_ABLATE': 'x'}):
            report = self.gate('ablated', baseline_engine=None, ablated_numerics=True)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertIn('candidate_repeat', self.leases[0])
        self.assertNotIn('candidate_repeat', self.processes[0])

    def test_repeat_candidate_and_twin_check_in_job_mode(self):
        # the repeat runs on its own load (a run-to-run check across loads); the twin check
        # compares the candidate with chunk8 of the same load.
        report = self.gate('repeat', repeat_candidate=True)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertTrue(report['repeat_candidate'])
        self.assertTrue(report['twin_check'])
        self.assertIn('candidate_repeat', self.leases[0])
        self.assertTrue(all('candidate_repeat' not in order for order in self.processes))
        self.assertIn('candidate_repeat', report['captures'])

    def test_repeat_candidate_rejects_a_changed_load_in_job_mode(self):
        real = self.run_unit

        def drift(job, env, limit):
            code = real(job, env, limit)
            if job['name'] == 'candidate_repeat':
                path = Path(job['stdout']).parent / 'dump' / 'h_20.f32'
                data = bytearray(path.read_bytes())
                data[3] ^= 0x01
                path.write_bytes(bytes(data))
            return code
        self.run_unit = drift
        args = self.arguments('repeat-bad', repeat_candidate=True)
        with patch.object(g.subprocess, 'run', self.subprocess_run), \
                self.assertRaisesRegex(ValueError, 'dump content differs: h_20'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertNotIn('gate1', report['steps'])

    def test_queue_wait_reaches_every_ticket(self):
        self.queue_wait = '21600'
        report = self.gate('wait', queue_wait=21600)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(report['queue_wait_s'], 21600)
        with patch.object(g.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(returncode=0)
            job = g.prepare_capture(Path('/e'), Path('/c'), Path('/i'), self.root / 'single', 1, 1, 8, 0, 0,
                                    Path('/queue'), 1, queue_wait=21600)
        self.assertEqual(job['record']['queued_command'][3:5], ['-w', '21600'])

    def test_lease_seconds_apply_to_job_mode(self):
        self.lease_limit = '900'
        report = self.gate('long-lease', lease_seconds=600)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        self.assertEqual(report['lease_seconds'], 600)
        plan = json.loads((self.root / 'long-lease/leases/0.plan.json').read_text())
        self.assertEqual(plan['budget_s'], 585)

    def test_cli_flag(self):
        argv = ['gates', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', 'o', '--baseline-name', 'b',
                '--baseline-engine', 'x']
        for extra, value in (([], 'auto'), (['--capture-mode', 'separate'], 'separate'),
                             (['--capture-mode', 'jobs', '--queue-wait', '600'], 'jobs')):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run') as execute, \
                    patch.object(g, 'prune_dumps', return_value={'files': 0, 'bytes': 0}):
                self.assertEqual(g.main(), 0)
            self.assertEqual(execute.call_args.args[0].capture_mode, value)
            self.assertEqual(execute.call_args.args[0].queue_wait, 600 if value == 'jobs' else 7200)


if __name__ == '__main__':
    unittest.main()
