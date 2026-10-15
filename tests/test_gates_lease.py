"""CPU tests: gate captures batched into few GPU leases (gates.py --lease-batch, tools/lease_runner.py)."""
import json
import os
import signal
import subprocess
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
import lease_runner as lr


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def jobs(self, count):
        return [{'name': f'j{i}', 'cmd': ['engine', str(i)], 'stdout': str(self.root / f'{i}.out'),
                 'stderr': str(self.root / f'{i}.err'), 'timeout_s': 300,
                 'env_set': {'HSA_USERPTR_FOR_PAGED_MEM': '0'}, 'env_unset': ['CRUCIBLE_ABLATE']}
                for i in range(count)]

    def test_runs_jobs_in_order_while_the_budget_fits_them(self):
        clock, seen = Clock(), []
        def run(job, env, limit):
            seen.append((job['name'], limit, env.get('HSA_USERPTR_FOR_PAGED_MEM'), 'CRUCIBLE_ABLATE' in env))
            clock.now += 50
            return 0
        result = lr.run_plan({'budget_s': 285, 'jobs': self.jobs(6)}, self.root / 'r.json', run, clock,
                             base_env={'CRUCIBLE_ABLATE': 'x', 'PATH': '/bin'})
        # 50 s jobs: after 4 jobs 85 s remain, which fits 50 * 1.25 + 5; after 5 only 35 s remain.
        self.assertEqual([j['name'] for j in result['jobs']], ['j0', 'j1', 'j2', 'j3', 'j4'])
        self.assertEqual(seen[0], ('j0', 285, '0', False))
        self.assertEqual(seen[4][1], 85)
        self.assertEqual(json.loads((self.root / 'r.json').read_text())['jobs'], result['jobs'])
        self.assertTrue(all(j['status'] == 'done' for j in result['jobs']))

    def test_a_scaled_job_waits_for_a_lease_that_fits_its_estimate(self):
        # 50 s jobs; the last one has twice the estimate (gates.py gives chunk1 a scale of 2).
        clock = Clock()
        def run(job, env, limit):
            clock.now += 50
            return 0
        jobs = self.jobs(4)
        jobs[-1]['estimate_scale'] = 2
        result = lr.run_plan({'budget_s': 275, 'jobs': jobs}, self.root / 'r.json', run, clock)
        # After 3 jobs 125 s remain: enough for 50 * 1.25 + 5, not for 50 * 1.25 * 2 + 5.
        self.assertEqual([j['name'] for j in result['jobs']], ['j0', 'j1', 'j2'])
        clock = Clock()
        result = lr.run_plan({'budget_s': 300, 'jobs': jobs}, self.root / 'r.json', run, clock)
        self.assertEqual([j['name'] for j in result['jobs']], ['j0', 'j1', 'j2', 'j3'])

    def test_first_job_always_runs_and_a_budget_cut_is_marked(self):
        clock = Clock()
        def run(job, env, limit):
            clock.now += limit
            return 124
        result = lr.run_plan({'budget_s': 285, 'jobs': self.jobs(3)}, self.root / 'r.json', run, clock)
        self.assertEqual(len(result['jobs']), 1)
        self.assertEqual(result['jobs'][0]['status'], 'budget')

    def test_own_timeout_and_failures_are_plain_results(self):
        clock = Clock()
        def run(job, env, limit):
            clock.now += 10
            return 7
        result = lr.run_plan({'budget_s': 600, 'jobs': self.jobs(2)}, self.root / 'r.json', run, clock)
        self.assertEqual([(j['returncode'], j['status'], j['timeout_s']) for j in result['jobs']],
                         [(7, 'done', 300), (7, 'done', 300)])

    def test_real_processes_write_their_logs(self):
        jobs = [{'name': 'echo', 'cmd': ['sh', '-c', 'echo "$X out"; echo err >&2'], 'stdout': str(self.root / 'o'),
                 'stderr': str(self.root / 'e'), 'timeout_s': 30, 'env_set': {'X': 'lease'}}]
        result = lr.run_plan({'budget_s': 60, 'jobs': jobs}, self.root / 'r.json')
        self.assertEqual(result['jobs'][0]['returncode'], 0)
        self.assertEqual((self.root / 'o').read_text(), 'lease out\n')
        self.assertEqual((self.root / 'e').read_text(), 'err\n')


    def test_the_running_capture_is_recorded_before_it_starts(self):
        seen = []
        def run(job, env, limit):
            seen.append(json.loads((self.root / 'r.json').read_text())['jobs'][-1])
            return 0
        lr.run_plan({'budget_s': 600, 'jobs': self.jobs(2)}, self.root / 'r.json', run, Clock())
        self.assertEqual([(j['name'], j['status']) for j in seen], [('j0', 'running'), ('j1', 'running')])

    def test_termination_kills_the_capture_process_group(self):
        pid_file = self.root / 'engine.pid'
        jobs = [{'name': 'slow', 'cmd': ['sh', '-c', f'echo $$ > {pid_file}; exec sleep 60'],
                 'stdout': str(self.root / 'o'), 'stderr': str(self.root / 'e'), 'timeout_s': 120}]
        plan, results = self.root / 'plan.json', self.root / 'r.json'
        plan.write_text(json.dumps({'budget_s': 200, 'jobs': jobs}))
        # The runner leads its own group here, as the outer `timeout` of a gpu-run lease does.
        runner = subprocess.Popen([sys.executable, str(ROOT / 'tools/lease_runner.py'), str(plan), str(results)],
                                  start_new_session=True)
        deadline = time.time() + 20
        while not pid_file.exists() or not pid_file.read_text().strip():
            self.assertLess(time.time(), deadline)
            time.sleep(0.05)
        engine = int(pid_file.read_text())
        runner.send_signal(signal.SIGTERM)
        self.assertNotEqual(runner.wait(timeout=20), 0)
        # `timeout` runs the capture in its own process group; the runner must not leave it behind.
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.kill(engine, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(engine, signal.SIGKILL)
            self.fail('the capture process outlived its lease runner')
        last = json.loads(results.read_text())['jobs'][-1]
        self.assertEqual((last['name'], last['status']), ('slow', 'killed'))


    def test_a_killed_runner_leaves_its_capture_to_the_lease_group_kill(self):
        pid_file = self.root / 'engine.pid'
        jobs = [{'name': 'slow', 'cmd': ['sh', '-c', f'echo $$ > {pid_file}; exec sleep 60'],
                 'stdout': str(self.root / 'o'), 'stderr': str(self.root / 'e'), 'timeout_s': 120}]
        plan, results = self.root / 'plan.json', self.root / 'r.json'
        plan.write_text(json.dumps({'budget_s': 200, 'jobs': jobs}))
        runner = subprocess.Popen([sys.executable, str(ROOT / 'tools/lease_runner.py'), str(plan), str(results)],
                                  start_new_session=True)
        deadline = time.time() + 20
        while not pid_file.exists() or not pid_file.read_text().strip():
            self.assertLess(time.time(), deadline)
            time.sleep(0.05)
        engine = int(pid_file.read_text())
        # SIGKILL gives the runner no chance to act; gpu-run then kills the lease's process group.
        runner.kill()
        runner.wait(timeout=10)
        try:
            os.killpg(runner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.kill(engine, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(engine, signal.SIGKILL)
            self.fail('the capture left the lease process group')


class LeaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.writable)
        self.root = Path(self.temp.name)
        patcher = patch.object(g, 'disk_preflight', return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = b'\x00\x00\x80\x3f' * g.WIDTH
        self.leases = []
        self.job_seconds = 10
        self.failing = {}
        self.dying_leases = []

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
                      prefill_rows=0, gpu_run=Path('/queue'), priority=1, engine_build_record=None,
                      lease_batch=True)
        return SimpleNamespace(**{**values, **extra})

    def engine(self, job, env, limit):
        command = job['cmd']
        def option(flag): return command[command.index(flag) + 1]
        dump = Path(option('--dump'))
        # The real runner opens both logs before the engine starts.
        Path(job['stderr']).write_text('')
        time.sleep(0.003)
        self.clock.now += self.job_seconds
        if self.failing.get(dump.parent.name):
            return self.failing.pop(dump.parent.name)
        n, spec = int(option('--n')), int(option('--spec'))
        rows = 1 + (n - 1 if spec == 0 else 0)
        for name in g.NAMES:
            (dump / name).write_bytes(self.payload * rows)
        Path(job['stdout']).write_text('ids:' + ' 7' * n + '\n')
        return 0

    def subprocess_run(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        if '--dump' in command:
            raise AssertionError('lease mode must not queue a capture by itself')
        runner = command.index(str(g.ROOT / 'tools/lease_runner.py'))
        self.assertEqual(command[:runner - 1], ['/queue', '-p', '1', '-w', '7200', '--', 'timeout', '-k', '10', '600'])
        plan = json.loads(Path(command[runner + 1]).read_text())
        self.leases.append([job['name'] for job in plan['jobs']])
        self.clock = Clock()
        if self.dying_leases:
            return self.dying_leases.pop(0)(plan, Path(command[runner + 2]))
        result = lr.run_plan(plan, Path(command[runner + 2]), self.engine, self.clock)
        return SimpleNamespace(returncode=0)

    def gate(self, name, **extra):
        args = self.arguments(name, **extra)
        with patch.object(g.subprocess, 'run', self.subprocess_run):
            g.run(args)
        return json.loads((args.output / 'report.json').read_text())

    def test_one_lease_runs_every_capture_and_the_checks_pass(self):
        report = self.gate('one')
        self.assertEqual(report['status'], 'PASS')
        # chunk1 runs last in a lease (gates.LEASE_LAST).
        self.assertEqual(self.leases, [['candidate', 'baseline', 'chunk8', 'serial', 'spec2', 'chunk1']])
        self.assertEqual(set(report['captures']), {'candidate', 'baseline', 'chunk1', 'chunk8', 'serial', 'spec2'})
        for capture in report['captures'].values():
            self.assertEqual(capture['returncode'], 0)
            self.assertEqual(len(capture['dumps']), 49)
            self.assertEqual(capture['output_ids'][:1], [7])
            self.assertEqual(capture['command'][0], str(Path(capture['command'][0])))
        self.assertEqual(len(report['leases']), 1)
        self.assertEqual(len(report['leases'][0]['jobs']), 6)

    def test_slow_captures_split_into_more_leases(self):
        self.job_seconds = 100
        report = self.gate('split')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(self.leases[0][:2], ['candidate', 'baseline'])
        self.assertEqual(sum(len(r['jobs']) for r in report['leases']), 6)
        # Two captures in each lease; chunk1 (estimate 100 x 1.25 x 1.4 + 5 = 180 s) follows spec2 in the last.
        self.assertEqual(len(report['leases']), 3)
        self.assertEqual(self.leases[-1], ['spec2', 'chunk1'])

    def test_a_budget_cut_capture_runs_again_in_a_fresh_directory(self):
        self.failing = {'chunk8': 124}
        self.job_seconds = 40
        report = self.gate('retry')
        self.assertEqual(report['status'], 'PASS')
        self.assertIn('chunk8', self.leases[1])
        self.assertEqual(report['captures']['chunk8']['returncode'], 0)

    def test_engine_failure_fails_the_gate(self):
        self.failing = {'chunk1': 3}
        args = self.arguments('engine-fail')
        with patch.object(g.subprocess, 'run', self.subprocess_run), self.assertRaisesRegex(ValueError, 'engine failed'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['steps'], {'gate1': True})

    def test_a_lease_that_dies_fails_the_gate_with_its_log(self):
        def dies(plan, results):
            # The outer timeout killed the runner after one capture finished.
            first = plan['jobs'][0]['name']
            self.clock = Clock()
            lr.run_plan({**plan, 'jobs': plan['jobs'][:1]}, results, self.engine, self.clock)
            return SimpleNamespace(returncode=124)
        self.dying_leases = [dies]
        args = self.arguments('lease-died')
        with patch.object(g.subprocess, 'run', self.subprocess_run), \
                self.assertRaisesRegex(ValueError, r'lease 0 exited 124.*leases/0\.log'):
            g.run(args)
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['leases'][0]['returncode'], 124)

    def test_a_capture_left_running_reruns_in_a_fresh_directory(self):
        def stops(plan, results):
            # A runner that ended with a capture still marked running: its files are not trusted.
            job = plan['jobs'][0]
            Path(job['stdout']).write_text('partial\n')
            (Path(job['stdout']).parent / 'dump' / 'stale.f32').write_bytes(b'x')
            lr.save(results, {'lease_started_ns': 1, 'budget_s': 285,
                              'jobs': [{'name': job['name'], 'status': 'running', 'started_ns': 1}]})
            return SimpleNamespace(returncode=0)
        self.dying_leases = [stops]
        report = self.gate('left-running')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(self.leases[1][0], 'candidate')
        self.assertFalse((self.root / 'left-running/candidate/dump/stale.f32').exists())

    def test_cache_hit_leaves_the_baseline_out_of_the_lease(self):
        cache = self.root / 'cache'
        patcher = patch.object(bc, 'platform_identity', return_value={'rocm_version': 'test', 'kernel': 'test', 'firmware': {}, 'libraries': {}})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gate('fill', baseline_cache=cache, baseline_recheck=0)
        report = self.gate('hit', baseline_cache=cache, baseline_recheck=0)
        self.assertEqual(report['baseline_source'], 'cache')
        self.assertEqual(self.leases[-1], ['candidate', 'chunk8', 'serial', 'spec2', 'chunk1'])
        self.assertEqual(report['status'], 'PASS')

    def test_cli_flag(self):
        argv = ['gates', 'ckpt', '--engine', 'e', '--ids', 'i', '--output', 'o', '--baseline-name', 'b',
                '--baseline-engine', 'x']
        for extra, value in (([], False), (['--lease-batch'], True)):
            with patch.object(sys, 'argv', argv + extra), patch.object(g, 'run') as execute, \
                    patch.object(g, 'prune_dumps', return_value={'files': 0, 'bytes': 0}):
                self.assertEqual(g.main(), 0)
            self.assertEqual(execute.call_args.args[0].lease_batch, value)


if __name__ == '__main__':
    unittest.main()
