#!/usr/bin/env python3
"""Qualify a resident adapter against its ordinary CLI, not engine numerics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

# Cache writes would change a source snapshot during the worker import.
sys.dont_write_bytecode = True

import gates as g
import long_capture as capture
import long_gates as lg
import resident_capture as resident
import build_long_probe as builder
from build_resident_probe import RESIDENT_FILES

LENGTHS = (1024, 4096, 16384)
TOKENS = 8
ROWS = 4096
CONTEXT = 32768


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def private(path, *, allow_empty=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (not info.st_size and not allow_empty):
        raise ValueError(f'need a private regular input with valid size: {path}')
    return info


def distinct(left, right):
    a, b = private(left), private(right)
    if (left.resolve() == right.resolve() or (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
            or g.digest(left) == g.digest(right)):
        raise ValueError('resident and ordinary CLI dispatch identities must differ')


def hash_tree(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'need a real input directory: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError(f'input tree contains a symlink: {path}')
        if path.is_dir(): continue
        private(path)
        result[str(path)] = g.digest(path)
    if not result: raise ValueError(f'empty input tree: {root}')
    return result


def snapshot(plan):
    trees = {name: hash_tree(Path(path)) for name, path in plan['trees'].items()}
    files, sizes = {}, {}
    for name in plan['inputs']:
        path = Path(name)
        info = private(path, allow_empty=name in plan.get('imports', ()))
        files[name], sizes[name] = g.digest(path), info.st_size
    return {'trees': trees, 'files': files, 'file_sizes': sizes}


def imports():
    return sorted({str(Path(path).resolve()) for module in tuple(sys.modules.values())
                   for path in (getattr(module, '__file__', None), getattr(module, '__cached__', None))
                   if path and Path(path).is_file()})


def prepare(args, root):
    engine, probe, source = args.engine.resolve(strict=True), args.resident.resolve(strict=True), args.source.resolve(strict=True)
    distinct(engine, probe)
    record_path = Path(str(engine)+'.build.json')
    record = json.loads(record_path.read_bytes())
    source_hashes = builder.source_hashes(source)
    flags = builder.compiler_flags(record.get('command'), source/'engine/engine.hip', record.get('build_cwd'))
    identity = resident.identity(probe, engine)
    if (record.get('source_commit') != 'c5dfa33' or record.get('source_hashes') != source_hashes
            or record.get('binary_sha256') != g.digest(engine)
            or identity['build']['source_hashes'] != source_hashes
            or identity['build']['engine_build_record_sha256'] != g.digest(record_path)
            or identity['build']['inherited_compiler_flags'] != list(flags)
            or identity['build']['bound_engine_source_path'] != str(source/'engine/engine.hip')):
        raise ValueError('complete c5dfa33 source and build identities must match')
    prompts = []
    fixed_path = g.ROOT/'bench/suite/kl/manifest.json'
    fixed = json.loads(fixed_path.read_bytes())['prompts']
    for length in LENGTHS:
        path = g.ROOT/f'bench/suite/kl/code-{length}.ids'
        private(path)
        if g.digest(path) != fixed[path.name]: raise ValueError('fixed code prompt changed')
        prompts.append({'file': str(path.resolve()), 'ids': g.parse_ids(path.read_text(), length),
                        'sha256': g.digest(path)})
    commands = [[str(probe), str(args.checkpoint.resolve()), str(CONTEXT), str(ROWS), str(root/'resident.tsv')]]
    lines = []
    for index, prompt in enumerate(prompts):
        lines.append('\t'.join(resident.safe_field(value) for value in
                     (prompt['file'], '-', root/f'resident-{index}', TOKENS)))
        commands.append([str(engine), 'gen', str(args.checkpoint.resolve()), '--ids-file', prompt['file'],
                         '--n', str(TOKENS), '--ctx', str(CONTEXT), '--chunk', '8', '--spec', '0',
                         '--prefill', str(ROWS), '--dump-logits', str(root/f'cli-{index}/logits.f32'),
                         '--draft-q', '0', '--draft-adapt', '0', '--draft-margin', '0', '--lookup', '0',
                         '--draft-overlap', '0', '--draft-last-head', '1', '--draft-head', 'full',
                         '--draft-head-auto', '0', '--mtp-prefix', '0'])
    (root/'resident.tsv').write_text('\n'.join(lines)+'\n')
    paths = [engine, probe, record_path, Path(str(probe)+'.build.json'), fixed_path,
             root/'resident.tsv', Path(sys.executable).resolve(), args.gpu_run.resolve(strict=True),
             Path(shutil.which('timeout')).resolve(), *(g.ROOT/name for name in RESIDENT_FILES)]
    imported = imports()
    plan = {'nonce': root.name, 'root': str(root), 'tokens': TOKENS, 'prompts': prompts, 'commands': commands,
            'imports': imported,
            'identity': identity, 'engine_build': record,
            'trees': {'source': str(source), 'checkpoint': str(args.checkpoint.resolve(strict=True)),
                      'tools': str(g.ROOT/'tools'), 'prompts': str(g.ROOT/'bench/suite/kl')},
            'inputs': sorted({str(path) for path in paths} | set(imported))}
    return plan


def artifact_tree(root, minimal=False):
    expected = {'worker.json', 'resident.stdout.log', *(f'child-{i}.stderr.log' for i in range(4)),
                *(f'cli-{i}' for i in range(3)), *(f'resident-{i}' for i in range(3))}
    if not minimal: expected |= {'plan.json', 'resident.tsv', 'report.json', 'queue.stdout.log', 'queue.stderr.log'}
    resident.check_tree(root, expected)
    for index in range(3):
        resident.check_tree(root/f'cli-{index}', ('stdout.log', 'logits.f32'))
        resident.check_tree(root/f'resident-{index}', ('native',))
        resident.check_tree(root/f'resident-{index}/native', ('stdout.log', 'logits.f32'))


def validate(plan, started_ns, ended_ns, minimal=True):
    root = Path(plan['root'])
    if root.name != plan['nonce']: raise ValueError('output directory is not bound to this nonce')
    artifact_tree(root, minimal)
    g.regular(root/'worker.json', started_ns)
    receipt = json.loads((root/'worker.json').read_bytes())
    children = receipt.get('children', [])
    if receipt.get('nonce') != plan['nonce'] or len(children) != 4:
        raise ValueError('missing, duplicate, or stale worker receipts')
    previous = started_ns
    for child, command in zip(children, plan['commands']):
        if (child.get('nonce') != plan['nonce'] or child.get('command') != command
                or type(child.get('returncode')) is not int or child['returncode'] != 0
                or type(child.get('started_ns')) is not int or type(child.get('ended_ns')) is not int
                or not previous <= child['started_ns'] <= child['ended_ns'] <= ended_ns):
            raise ValueError('child dispatch, time, nonce, or exit code differs')
        previous = child['ended_ns']
    g.regular(root/'resident.stdout.log', children[0]['started_ns'])
    events = [json.loads(line) for line in (root/'resident.stdout.log').read_text().splitlines() if line.startswith('{')]
    if events != [{'event': 'resident', 'requests': 3, 'bulk_rows': ROWS}]:
        raise ValueError('resident did not complete three native requests')
    result, inodes = [], set()
    for index, prompt in enumerate(plan['prompts']):
        evidence = []
        for role, child in (('resident', children[0]), ('cli', children[index+1])):
            target = root/f'{role}-{index}'
            if role == 'resident': target /= 'native'
            for name in ('stdout.log', 'logits.f32'):
                info = g.regular(target/name, child['started_ns'])
                inode = (info.st_dev, info.st_ino)
                if inode in inodes: raise ValueError('duplicate output identity')
                inodes.add(inode)
                if g.digest(target/name) == prompt['sha256']:
                    raise ValueError('output bytes equal prompt input')
            measured = capture.native_evidence(target/'logits.f32', target/'stdout.log', child['started_ns'], plan['tokens'])
            values = next(lg.rows(target/'logits.f32', 1))
            if not any(values) or max(values) == min(values): raise ValueError('degenerate logit comparator')
            if measured['output_ids'] == prompt['ids'][:plan['tokens']]:
                raise ValueError('output IDs equal prompt input')
            evidence.append({**measured, 'stdout_sha256': g.digest(target/'stdout.log')})
        if (evidence[0]['output_ids'] != evidence[1]['output_ids'] or
                Path(evidence[0]['logit_path']).read_bytes() != Path(evidence[1]['logit_path']).read_bytes()):
            raise ValueError('complete output IDs or raw logit bytes differ')
        result.append({'prompt': prompt['file'], 'resident': evidence[0], 'ordinary_cli': evidence[1]})
    return result


def unchanged_launch(plan):
    for name, expected in plan['launch_hashes'].items():
        path = Path(name); private(path, allow_empty=name in plan.get('imports', ()))
        if g.digest(path) != expected: raise ValueError(f'launch input changed: {path}')


def worker(plan):
    root = Path(plan['root']); children = []
    active = None
    def cleanup(signum=None, frame=None):
        if active is not None:
            try: os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        if signum is not None: raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, cleanup); signal.signal(signal.SIGINT, cleanup)
    for index, command in enumerate(plan['commands']):
        unchanged_launch(plan)
        if index: (root/f'cli-{index-1}').mkdir()
        stdout_path = root/'resident.stdout.log' if not index else root/f'cli-{index-1}/stdout.log'
        start = time.time_ns()
        with stdout_path.open('xb') as stdout, (root/f'child-{index}.stderr.log').open('xb') as stderr:
            try:
                active = subprocess.Popen(['timeout', '-k', '10', '300', *command],
                                          stdout=stdout, stderr=stderr, start_new_session=True)
                code = active.wait(timeout=312)
            finally:
                cleanup()
                if active is not None: active.wait(timeout=10)
                active = None
        children.append({'nonce': plan['nonce'], 'command': command, 'returncode': code,
                         'started_ns': start, 'ended_ns': time.time_ns()})
        if code: raise ValueError(f'child {index} failed ({code})')
    unchanged_launch(plan)
    g.save(root/'worker.json', {'nonce': plan['nonce'], 'children': children})


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.output.resolve()/uuid.uuid4().hex; root.mkdir(mode=0o700)
    report = {'status': 'FAIL', 'qualifier': 'resident_equivalence', 'primary_qualification': 'tools/gates.py'}
    try:
        plan = prepare(args, root)
        before = snapshot(plan)
        launch = {**before['files'], **before['trees']['source'],
                  **before['trees']['tools'], **before['trees']['prompts']}
        plan = {**plan, 'launch_hashes': launch}
        frozen = encode(plan); (root/'plan.json').write_bytes(frozen)
        queued = [str(args.gpu_run), '-p', str(args.priority), '-w', '7200', '--', 'timeout', '-k', '10', '300',
                  sys.executable, str(Path(__file__).resolve()), '--worker']
        report = {**report, 'plan': plan, 'plan_sha256': hashlib.sha256(frozen).hexdigest(),
                  'before': before, 'queued_command': queued, 'started_ns': time.time_ns(),
                  'environment': g.runtime_environment(os.environ)}
        g.save(root/'report.json', report)
        with (root/'queue.stdout.log').open('xb') as stdout, (root/'queue.stderr.log').open('xb') as stderr:
            done = subprocess.run(queued, input=frozen, stdout=stdout, stderr=stderr, check=False)
        report = {**report, 'returncode': done.returncode, 'ended_ns': time.time_ns(), 'after': snapshot(plan)}
        if done.returncode: raise ValueError(f'grant failed ({done.returncode}); inspect {root}')
        if (report['after'] != before or g.digest(root/'plan.json') != report['plan_sha256']):
            raise ValueError('source, build, binary, checkpoint, prompt, manifest, or import changed')
        measurements = validate(plan, report['started_ns'], report['ended_ns'], minimal=False)
        report = {**report, 'status': 'PASS', 'measurements': measurements}
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        report = {**report, 'status': 'FAIL', 'error': str(error)}
    g.save(root/'report.json', report)
    print(root/'report.json')
    return 0 if report['status'] == 'PASS' else 1


def main():
    if sys.argv[1:] == ['--worker']:
        worker(json.loads(sys.stdin.buffer.read())); return 0
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'resident', 'source', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    # An explicit dest keeps args.gpu_run when the public snapshot renames the flag.
    parser.add_argument('--gpu-run', dest='gpu_run', type=Path, required=True)
    parser.add_argument('--priority', type=int, choices=range(4), default=3)
    return run(parser.parse_args())


if __name__ == '__main__': sys.exit(main())
