#!/usr/bin/env python3
"""Run several gate captures inside one GPU lease, one process each.

gates.py --lease-batch starts this under gpu-run. Each job keeps its own
command, environment changes, logs and timeout, so the evidence of a capture
is the same as with one lease per capture. The runner starts a job only when
the rest of the lease budget fits it (estimate: the slowest job seen so far in
this lease); a job cut short by the budget is marked "budget" so that gates.py
can run it again in a new lease. Results are written after every job.

Memory floor: a plan with mem_floor_gb first polls
MemAvailable (tools/mem_floor.sh, every mem_poll_s for at most mem_wait_s) before the first engine starts. A
lease that starts right after another one can see 74-118 GB free for a while. If the floor does not come, no job runs
and the runner exits 76 (EXIT_MEMORY); gates.py runs the lease again. The job budget starts after the wait.
"""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXIT_MEMORY = 76  # mem_floor.sh: below the floor, no engine started
# The estimate for the next job is the slowest job so far times this, plus a margin.
ESTIMATE_FACTOR = 1.25
ESTIMATE_MARGIN_S = 5


class Terminated(Exception):
    def __init__(self, signum):
        super().__init__(f'lease runner received signal {signum}')
        self.signum = signum


STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
# After a stop signal the capture gets this long to end on SIGTERM before SIGKILL.
STOP_GRACE_S = 12


def execute(job, env, limit):
    with open(job['stdout'], 'wb') as stdout, open(job['stderr'], 'wb') as stderr:
        # --foreground keeps the capture in the lease's process group, which gpu-run kills after
        # the job; a runner that dies by SIGKILL therefore cannot leave a capture on the GPU.
        # Stop signals stay blocked until the try block, so none can arrive between Popen and it.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        try:
            child = subprocess.Popen(['timeout', '--foreground', '-k', '10', str(limit), *job['cmd']],
                                     stdout=stdout, stderr=stderr, env=env,
                                     preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS))
        except BaseException:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
            raise
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
            return child.wait()
        except Terminated:
            # timeout passes SIGTERM to the capture and kills it 10 s later if it does not end.
            child.terminate()
            try:
                child.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def raise_terminated(signum, _frame):
    raise Terminated(signum)


def job_environment(base, job):
    env = {k: v for k, v in base.items() if k not in job.get('env_unset', ())}
    return {**env, **job.get('env_set', {})}


def save(path, value):
    temp = Path(f'{path}.tmp')
    temp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(temp, path)


def wait_memory(plan, results_path):
    """The memory floor of the plan: {'ok', 'returncode', 'log'} from tools/mem_floor.sh."""
    log = Path(f'{results_path}.mem')
    env = {**os.environ, 'MEM_FLOOR_WAIT_S': str(plan.get('mem_wait_s', 300)),
           'MEM_FLOOR_POLL_S': str(plan.get('mem_poll_s', 5))}
    code = subprocess.run(['bash', str(ROOT / 'tools/mem_floor.sh'), str(plan['mem_floor_gb']), str(log)],
                          env=env, check=False).returncode
    text = log.read_text() if log.exists() else ''
    return {'ok': code == 0, 'returncode': code, 'floor_gb': plan['mem_floor_gb'], 'log': text}


def run_plan(plan, results_path, run=execute, clock=time.monotonic, base_env=None, memory=None):
    base_env = dict(os.environ) if base_env is None else base_env
    start, slowest = clock(), 0.0
    result = {'lease_started_ns': time.time_ns(), 'budget_s': plan['budget_s'], 'jobs': [],
              **({'memory': memory} if memory is not None else {})}
    save(results_path, result)
    for job in plan['jobs']:
        remaining = plan['budget_s'] - (clock() - start)
        # The first job always runs: a lease that cannot fit one job is no better later.
        # estimate_scale: a job that the planner knows to be longer than the others (gates.py: chunk1).
        if result['jobs'] and remaining < slowest * ESTIMATE_FACTOR * job.get('estimate_scale', 1) + ESTIMATE_MARGIN_S:
            break
        limit = max(1, min(job['timeout_s'], int(remaining)))
        began, began_ns = clock(), time.time_ns()
        # Name the capture in flight, so a lease that dies leaves a record of it.
        result['jobs'].append({'name': job['name'], 'status': 'running', 'timeout_s': limit, 'started_ns': began_ns})
        save(results_path, result)
        try:
            code = run(job, job_environment(base_env, job), limit)
        except Terminated:
            result['jobs'][-1] = {**result['jobs'][-1], 'status': 'killed', 'ended_ns': time.time_ns()}
            save(results_path, result)
            raise
        took = clock() - began
        slowest = max(slowest, took)
        cut = code in (124, 137) and limit < job['timeout_s']
        result['jobs'][-1] = {'name': job['name'], 'returncode': code, 'timeout_s': limit,
                              'status': 'budget' if cut else 'done', 'started_ns': began_ns,
                              'ended_ns': time.time_ns(), 'seconds': took}
        save(results_path, result)
        if cut:
            break
    result['lease_ended_ns'] = time.time_ns()
    save(results_path, result)
    return result


def main():
    if len(sys.argv) != 3:
        print('usage: lease_runner.py PLAN.json RESULTS.json', file=sys.stderr)
        return 2
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, raise_terminated)
    try:
        plan, results_path = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
        memory = None
        if plan.get('mem_floor_gb') is not None:
            memory = wait_memory(plan, results_path)
            if not memory['ok']:
                save(results_path, {'lease_started_ns': time.time_ns(), 'budget_s': plan['budget_s'], 'jobs': [],
                                    'memory': memory})
                print(f"lease runner: no engine start, {memory['log'].strip()}", file=sys.stderr)
                return EXIT_MEMORY if memory['returncode'] == EXIT_MEMORY else memory['returncode'] or 1
        run_plan(plan, results_path, memory=memory)
    except Terminated as error:
        print(error, file=sys.stderr)
        return 128 + error.signum
    return 0


if __name__ == '__main__':
    sys.exit(main())
