#!/usr/bin/env python3
"""Proof for gen --jobs: captures of one engine process with several
jobs must equal separate processes byte for byte, in any job order.

Input: report.json files of gates.py runs in separate mode (--capture-mode separate --keep-dumps),
for example dec and bulk of one checkpoint, all with the same candidate binary. The tool takes the
candidate captures of every report, groups them by load (--ctx, --prefill), and runs each group as one
gen --jobs process in reverse gate order. Every group starts and ends with the same canary job. All
processes share one GPU lease. Then it compares every dump file and the output IDs of each job with
its separate capture, and the canary with itself. Optionally it compares the captures of gates.py
job-mode reports (--jobs-report) with the separate captures.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates as g

CANDIDATE_CAPTURES = ('candidate', 'chunk1', 'chunk8', 'serial', 'spec2')


def separate_captures(reports):
    """[(label, report dir, capture name, record)] for every candidate capture, in gate order."""
    found, binary = [], None
    for path in reports:
        report = json.loads(path.read_text())
        if report.get('status') != 'PASS' or report.get('capture_mode', {}).get('used', 'separate') != 'separate':
            raise ValueError(f'{path}: needs a passing gates.py report in separate mode')
        if binary not in (None, report['binary_sha256']):
            raise ValueError('all reports must come from one candidate binary')
        binary = report['binary_sha256']
        for name in CANDIDATE_CAPTURES:
            found.append((f'{path.parent.name}-{name}', path.parent, name, report['captures'][name]))
    return found, binary


def proof_groups(captures):
    """Groups by load; each group runs in reverse order with its first job repeated last as a canary."""
    groups = {}
    for capture in captures:
        groups.setdefault(g.load_key(capture[3]['command']), []).append(capture)
    return [(key, list(reversed(members))) for key, members in groups.items()]


def write_group(index, key, members, output):
    """Fresh dump directories and the job file of one process. Returns [(label, dir, source capture)]."""
    lines, jobs = [], []
    canary = ('canary-' + members[0][0],) + members[0][1:]
    for label, source_dir, name, record in members + [canary]:
        directory = output / label
        (directory / 'dump').mkdir(parents=True)
        args = list(record['command'][3:])
        args[args.index('--dump') + 1] = str(directory / 'dump')
        lines.append(json.dumps({'args': args, 'stdout': str(directory / 'stdout.log'),
                                 'stderr': str(directory / 'stderr.log')}))
        jobs.append({'label': label, 'dir': str(directory), 'source': str(source_dir / name), 'name': name,
                     'rows': record['dump_rows'], 'tokens': len(record['output_ids']),
                     'output_ids': record['output_ids']})
    path = output / f'jobs{index}.jsonl'
    path.write_text('\n'.join(lines) + '\n')
    return path, jobs


def compare_job(job):
    directory = Path(job['dir'])
    g.check_dumps(directory / 'dump', job['rows'], 0)
    g.compare_dumps(Path(job['source']) / 'dump', directory / 'dump')
    ids = g.parse_output((directory / 'stdout.log').read_text(), job['tokens'])
    if ids != job['output_ids']:
        raise ValueError(f'{job["label"]}: output IDs differ from the separate capture')
    return {'label': job['label'], 'source': job['source'], 'dumps_equal': True, 'ids_equal': True,
            'dump_files': len(g.NAMES), 'dump_bytes': job['rows'] * g.WIDTH * 4 * len(g.NAMES)}


def compare_jobs_report(path, separate):
    """Byte-for-byte check of a gates.py job-mode run against the separate captures of the same gate."""
    report = json.loads(path.read_text())
    if report.get('capture_mode', {}).get('used') != 'jobs':
        raise ValueError(f'{path}: not a job-mode report')
    results = []
    for name in CANDIDATE_CAPTURES:
        source = separate / name
        g.compare_dumps(source / 'dump', path.parent / name / 'dump')
        reference = json.loads((separate / 'report.json').read_text())['captures'][name]['output_ids']
        if report['captures'][name]['output_ids'] != reference:
            raise ValueError(f'{path}: {name} output IDs differ')
        results.append({'capture': name, 'process': report['captures'][name]['process'], 'dumps_equal': True,
                        'ids_equal': True})
    return {'report': str(path), 'separate': str(separate / 'report.json'), 'captures': results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--separate-report', type=Path, nargs='+', required=True)
    ap.add_argument('--jobs-report', type=Path, nargs='*', default=[],
                    help='gates.py job-mode reports; each is compared with the separate report of the same gate')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-run', type=Path, default=Path.home() / 'crucible/bin/gpu-run')
    ap.add_argument('--priority', type=int, default=1)
    ap.add_argument('--queue-wait', type=int, default=g.QUEUE_WAIT_S)
    ap.add_argument('--skip-run', action='store_true', help='compare an earlier run in --output')
    args = ap.parse_args()
    captures, binary = separate_captures(args.separate_report)
    engine = args.engine.resolve(strict=True)
    if g.digest(engine) != binary:
        raise SystemExit('the engine is not the binary of the separate reports')
    result = {'engine': str(engine), 'engine_sha256': binary, 'checkpoint': str(args.checkpoint),
              'separate_reports': [str(p) for p in args.separate_report], 'processes': []}
    groups = proof_groups(captures)
    if not args.skip_run:
        args.output.mkdir(parents=True)
        units = []
        for index, (key, members) in enumerate(groups):
            process = {'name': f'jobs{index}', 'load': {'ctx': key[0], 'prefill': key[1]}}
            result['processes'].append(process)
            def prepare(index=index, key=key, members=members, process=process):
                # A rerun after a lease cut starts from empty dump directories.
                for label in [m[0] for m in members] + ['canary-' + members[0][0]]:
                    shutil.rmtree(args.output / label, ignore_errors=True)
                path, jobs = write_group(index, key, members, args.output)
                process.update({'jobs_file': str(path), 'jobs': jobs})
                return {'name': process['name'], 'timeout_s': 300,
                        'cmd': [str(engine), 'gen', str(args.checkpoint), '--jobs', str(path)],
                        'stdout': str(args.output / f'jobs{index}.stdout.log'),
                        'stderr': str(args.output / f'jobs{index}.stderr.log'),
                        'env_set': {'HSA_USERPTR_FOR_PAGED_MEM': '0'}}
            units.append((process['name'], prepare))
        lease_args = argparse.Namespace(output=args.output, gpu_run=args.gpu_run, priority=args.priority,
                                        queue_wait=args.queue_wait)
        done = g.run_units_in_leases(units, lease_args, result)
        for process in result['processes']:
            item, _ = done[process['name']]
            process.update({'returncode': item['returncode'], 'seconds': item['seconds']})
        g.save(args.output / 'proof.json', result)
    else:
        result = json.loads((args.output / 'proof.json').read_text())
    failures = []
    for process in result['processes']:
        process['compared'] = []
        for job in process['jobs']:
            try:
                process['compared'].append(compare_job(job))
            except (OSError, ValueError) as error:
                failures.append(f'{process["name"]} {job["label"]}: {error}')
        first, canary = process['jobs'][0], process['jobs'][-1]
        try:
            g.compare_dumps(Path(first['dir']) / 'dump', Path(canary['dir']) / 'dump')
            process['canary'] = {'first': first['label'], 'last': canary['label'], 'match': True}
        except (OSError, ValueError) as error:
            failures.append(f'{process["name"]} canary: {error}')
    by_gate = {p.parent.name: p.parent for p in args.separate_report}
    result['jobs_reports'] = []
    for path in args.jobs_report:
        gate = path.parent.name.removesuffix('-jobs')
        try:
            result['jobs_reports'].append(compare_jobs_report(path, by_gate[gate]))
        except (OSError, ValueError, KeyError) as error:
            failures.append(f'{path}: {error}')
    result['failures'] = failures
    result['status'] = 'PASS' if not failures else 'FAIL'
    g.save(args.output / 'proof.json', result)
    print(f'{result["status"]}: {sum(len(p["jobs"]) for p in result["processes"])} jobs in '
          f'{len(result["processes"])} processes, {len(result["jobs_reports"])} job-mode reports; '
          f'evidence: {args.output}/proof.json')
    for failure in failures:
        print('FAIL:', failure)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
