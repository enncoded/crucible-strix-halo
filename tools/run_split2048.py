#!/usr/bin/env python3
"""Run one bounded supplemental qualification through the shared GPU queue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from split2048_probe import compare_case, digest
from gates import runtime_environment
import work_paths


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_inputs(record, binary):
    if digest(binary) != record['binary_sha256']:
        raise ValueError('probe binary changed')
    if any(digest(path) != value for path, value in record['source_hashes'].items()):
        raise ValueError('probe source or input changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('build', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'FAIL', 'kind': 'supplemental conditional 2048 qualification',
              'started_ns': time.time_ns(), 'runtime': runtime_environment(os.environ),
              'baseline_name': 'crucible-t5',
              'baseline_sha256': 'fb269999bd9d894dafa2bb5bab08b783001b5fbd65000ce1baccfe63b9bb6526'}
    try:
        if shutil.disk_usage(args.output).free < 150_000_000_000:
            raise ValueError('less than 150 GB free')
        build = args.build.resolve()
        record_path = build / 'build.json'
        record_hash = digest(record_path)
        record = json.loads(record_path.read_text())
        binary = build / 'split2048'
        check_inputs(record, binary)
        baseline = work_paths.worktree('baseline') / 'build/engine/crucible-t5'
        if digest(baseline) != report['baseline_sha256']:
            raise ValueError('named baseline changed')
        if record['binary_sha256'] == report['baseline_sha256']:
            raise ValueError('candidate cannot be its baseline')
        report['build_record_sha256'] = record_hash
        report['binary_sha256'] = record['binary_sha256']
        report['source_commit'] = record['source_commit']
        report['inputs'] = json.loads((build / 'cases.json').read_text())
        report['checkpoint_metadata'] = {p.name: digest(p) for p in sorted(args.checkpoint.glob('*.json'))}
        if not report['checkpoint_metadata']:
            raise ValueError('missing checkpoint metadata')
        command = [str(binary), str(args.checkpoint.resolve()), str((args.output / 'captures').resolve())]
        for case in report['inputs']:
            command.extend((case['ids'], str(case['cut'])))
        queued = [str(work_paths.gpu_run()), '-p', '2', '-w', '7200',
                  '--', 'timeout', '-k', '10', '230', *command]
        report['command'] = queued
        write(args.output / 'report.json', report)
        with (args.output / 'grant.log').open('w') as log:
            result = subprocess.run(queued, stdout=log, stderr=subprocess.STDOUT)
        report['returncode'] = result.returncode
        if result.returncode:
            raise ValueError('qualification process failed')
        captures = args.output / 'captures'
        if set(p.name for p in captures.iterdir()) != {'case0', 'case1', 'case2'}:
            raise ValueError('invalid case inventory')
        report['cases'] = [compare_case(captures / f'case{i}', modes=('whole', 'split', 'restored'),
                                       expected_cut=report['inputs'][i]['cut']) for i in range(3)]
        check_inputs(record, binary)
        if digest(record_path) != record_hash:
            raise ValueError('build record changed')
        if any(digest(args.checkpoint / name) != value for name, value in report['checkpoint_metadata'].items()):
            raise ValueError('checkpoint metadata changed')
        if not all(case['equal'] for case in report['cases']):
            raise ValueError('bitwise mismatch')
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = str(error)
    finally:
        report['ended_ns'] = time.time_ns()
        write(args.output / 'report.json', report)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
