#!/usr/bin/env python3
"""Measure startup in fresh processes. Each paired batch gets one queue turn."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402


def system_state():
    selected = {'MemAvailable', 'SwapFree', 'Cached'}
    memory = {line.split(':')[0]: line.split(':')[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.split(':')[0] in selected}
    return {'loadavg': os.getloadavg(), 'memory': memory,
            'cpu_pressure': Path('/proc/pressure/cpu').read_text(),
            'memory_pressure': Path('/proc/pressure/memory').read_text()}


def inside(binary, checkpoint, modes, output):
    for trial, mode in enumerate(modes):
        prefix = output / f'{trial}-{mode}'
        command = [str(binary), str(checkpoint), str(mode)]
        env = dict(os.environ)
        for key in ('CRUCIBLE_PROFILE', 'CRUCIBLE_WARMUP', 'HSA_USE_SVM', 'GPU_PINNED_MIN_XFER_SIZE'):
            env.pop(key, None)
        before, usage = system_state(), resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        prefix.with_suffix('.stdout').write_text(result.stdout)
        prefix.with_suffix('.stderr').write_text(result.stderr)
        prefix.with_suffix('.command.json').write_text(json.dumps(command, indent=2) + '\n')
        record = {'exit': result.returncode, 'wall_seconds': time.monotonic() - start,
                  'before': before, 'after': system_state(),
                  'child_user_seconds': after.ru_utime - usage.ru_utime,
                  'child_system_seconds': after.ru_stime - usage.ru_stime,
                  'child_major_faults': after.ru_majflt - usage.ru_majflt,
                  'child_minor_faults': after.ru_minflt - usage.ru_minflt}
        prefix.with_suffix('.system.json').write_text(json.dumps(record, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{prefix}: exit {result.returncode}')
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        phases = {x['phase']: x['seconds'] for x in events if 'phase' in x}
        if not {'engine_ready', 'resident_ready', 'model_load', 'plan_build'} <= phases.keys():
            raise RuntimeError(f'{prefix}: missing phases')
        print(prefix, phases, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--modes', default='0,4,0,4')
    parser.add_argument('--pairs', default='4,1,2,3')
    args = parser.parse_args()
    if args.inside:
        if args.checkpoint is None or not args.output.is_dir():
            parser.error('inside requires checkpoint and existing output directory')
        modes = [int(x) for x in args.modes.split(',')]
        if any(x not in range(5) for x in modes):
            parser.error('modes must be 0 through 4')
        inside(args.binary, args.checkpoint, modes, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    (args.output / 'binary.json').write_text(json.dumps(
        {'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}, indent=2) + '\n')
    queue = work_paths.gpu_run()
    for pair in (int(x) for x in args.pairs.split(',')):
        if pair not in range(1, 5):
            parser.error('pairs must be 1 through 4')
        for tier, suffix in (('305', '3.05bpw_h5_ng5'), ('405', '4.05bpw_h6_ng6')):
            target = args.output / f'{tier}-pair{pair}'
            target.mkdir()
            checkpoint = Path.home() / f'models/exl3/Qwen3.8-Flash-Next-exl3-{suffix}'
            command = [str(queue), '-p', '1', '-w', '7200', '--', 'timeout', '-k', '10', '300',
                       sys.executable, str(Path(__file__).resolve()), '--inside', '--binary', str(binary),
                       '--output', str(target), '--checkpoint', str(checkpoint), '--modes', f'0,{pair},0,{pair}']
            target.with_suffix('.command.json').write_text(json.dumps(command, indent=2) + '\n')
            result = subprocess.run(command, capture_output=True, text=True)
            target.with_suffix('.stdout').write_text(result.stdout)
            target.with_suffix('.stderr').write_text(result.stderr)
            target.with_suffix('.rc').write_text(str(result.returncode) + '\n')
            if result.returncode:
                raise RuntimeError(f'{target}: exit {result.returncode}')
            print(target, 'complete', flush=True)


if __name__ == '__main__':
    main()
