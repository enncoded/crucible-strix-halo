"""CPU coordinator. Queue only one bounded server lease at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from . import runner


def command(args, label, plan, mode, limit, budget, *, probe=False):
    return [str(args.gpu_run), '-p', '2', '-w', '7200', '--', 'timeout', '-k', '10', str(limit),
            sys.executable, '-m', 'bench.agents.lease', '--plan', str(plan), '--engine', str(args.engine),
            '--record', str(args.record), '--checkpoint', str(args.checkpoint), '--mode', mode,
            '--limit', str(limit), '--cache-mib', str(budget), '--label', label,
            '--output', str(args.root / label), *(['--probe'] if probe else [])]


def run_one(args, label, argv, outcomes):
    print('START', label, flush=True)
    with (args.root / (label + '.queue.log')).open('x') as log:
        value = subprocess.run(argv, stdout=log, stderr=log)
    updated = [*outcomes, {'label': label, 'returncode': value.returncode, 'command': argv}]
    print('END', label, value.returncode, flush=True)
    (args.root / 'outcomes.json').write_text(json.dumps(updated, indent=2) + '\n')
    if value.returncode:
        runner.write_new(args.root / 'STOP.json', {'failed_label': label, 'returncode': value.returncode})
    return updated, value.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu-run', dest='gpu_run', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--label', default='t5')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    outcomes, budgets = [], {}
    for k in (1, 2, 4, 8):
        for policy in ('greedy', 'sampled'):
            for mode in ('serial', 'batch'):
                key = (mode, k if mode == 'batch' else 1)
                if key not in budgets:
                    label = f'{args.label}-probe-{mode}-k{k}'
                    plan = Path(f'results/plans/k{k}-{policy}.json')
                    argv = command(args, label, plan, mode, 300, 0, probe=True)
                    outcomes, rc = run_one(args, label, argv, outcomes)
                    if rc: return 1
                    receipt = json.loads((args.root / label / 'lease.json').read_text())
                    budgets[key] = receipt['recommended_cache_mib']
                starts = (None,) if k <= 2 else (0, 5, 10) if k == 4 else (0, 5, 7, 9, 11, 13)
                for start in starts:
                    suffix = '' if start is None else f'-s{start}'
                    label = f'{args.label}-{mode}-k{k}-{policy}{suffix}'
                    plan_suffix = suffix + '-v2' if k == 8 else suffix
                    plan = Path(f'results/plans/k{k}-{policy}{plan_suffix}.json')
                    limit = 600 if k == 8 else 300
                    argv = command(args, label, plan, mode, limit, budgets[key])
                    outcomes, rc = run_one(args, label, argv, outcomes)
                    if rc: return 1
    runner.write_new(args.root / 'PASS.json', {'completed_segments': len(outcomes)})
    return 0


if __name__ == '__main__': raise SystemExit(main())
