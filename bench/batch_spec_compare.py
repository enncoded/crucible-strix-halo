"""Fail-closed comparison for the two-request native and packed spec probes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

VOCABULARY = 248320
STACK_WIDTH = 10240
STATE_SPANS = 13 * 4 + 36 * 3 + 1

def files(root):
    paths = list(root.rglob('*'))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError('capture links are not allowed')
    records = {path.relative_to(root): path for path in paths if path.is_file()}
    for request in range(2):
        ids = root / f'ids_{request}.i32'
        if not ids.is_file() or ids.stat().st_size < 8 or ids.stat().st_size % 4:
            raise ValueError(f'missing published IDs for request {request}')
        steps = sorted((root / f'request_{request}').glob('step_*'), key=lambda p: p.name)
        if not steps or {p.name for p in steps} != {f'step_{i}' for i in range(len(steps))}:
            raise ValueError(f'missing or invalid step sequence for request {request}')
        for step in steps:
            fields = (step / 'schedule.txt').read_text().split()
            schedule = dict(field.split('=', 1) for field in fields)
            if len(fields) != 6 or set(schedule) != {'request', 'generation', 'base', 'rows', 'replay', 'mpos'} or any(
                    re.fullmatch(r'[0-9]+', value) is None for value in schedule.values()):
                raise ValueError('invalid resolved request schedule')
            values = {key: int(value) for key, value in schedule.items()}
            rows = values['rows']
            if values['request'] != request or values['generation'] != 1 or not 1 <= rows <= 8 or values['replay'] > 7:
                raise ValueError('invalid request span')
            sizes = {'input.i32': rows * 4, 'target.i32': rows * 4, 'logits.f32': rows * VOCABULARY * 4,
                     'stacks.f32': rows * STACK_WIDTH * 4, 'history.i32': (values['base'] + rows) * 4}
            for name, size in sizes.items():
                path = step / name
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError(f'missing or truncated capture: {path}')
            for index in range(STATE_SPANS):
                path = step / f'state_{index}.bin'
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f'missing sequence-state span: {path}')
        check_chain(root, request, steps)
    return records

def words(path):
    return tuple(value[0] for value in struct.iter_unpack('<i', path.read_bytes()))

def check_chain(root, request, steps):
    published = words(root / f'ids_{request}.i32')
    emitted = [published[0]]
    next_base, replay = None, 0
    for step in sorted(steps, key=lambda p: int(p.name[5:])):
        schedule = {key: int(value) for key, value in (field.split('=', 1) for field in (step / 'schedule.txt').read_text().split())}
        inputs, target = words(step / 'input.i32'), words(step / 'target.i32')
        if inputs[0] != emitted[-1] or schedule['replay'] != replay or (next_base is not None and schedule['base'] != next_base):
            raise ValueError('broken request continuation')
        if any(token < 0 or token >= VOCABULARY for token in (*inputs, *target)):
            raise ValueError('invalid captured token ID')
        accepted = 0
        while accepted + 1 < len(inputs) and inputs[accepted + 1] == target[accepted]:
            accepted += 1
        emitted.extend(target[:accepted + 1])
        next_base, replay = schedule['base'] + accepted + 1, accepted
    if tuple(emitted[:len(published)]) != published or len(emitted) < len(published):
        raise ValueError('missing steps or mismatched published IDs')

def digest(path):
    value = sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            value.update(chunk)
    return value.digest()

def identity(root, mode, producer):
    path = root.parent / (root.name + '_provenance.json')
    if not path.is_file() or path.is_symlink():
        raise ValueError('missing probe-written provenance')
    record = json.loads(path.read_text())
    if set(record) != {'mode', 'binary_sha256', 'source_sha256', 'command_hex', 'cwd_hex'} or record['mode'] != mode:
        raise ValueError('wrong probe provenance mode')
    for key, expected in [('binary_sha256', producer['binary_sha256']), ('source_sha256', producer['source_manifest_sha256'])]:
        if record[key] != expected or re.fullmatch('[0-9a-f]{64}', expected) is None:
            raise ValueError('probe identity does not match its build record')
    if digest(Path(producer['binary_path'])).hex() != record['binary_sha256']:
        raise ValueError('producer binary changed')
    encoded = bytes.fromhex(record['command_hex'])
    if not encoded.endswith(b'\0'):
        raise ValueError('incomplete recorded command')
    argv = encoded[:-1].decode().split('\0')
    cwd = Path(bytes.fromhex(record['cwd_hex']).decode())
    if len(argv) < 8 or not cwd.is_absolute() or (cwd / argv[3]).resolve() != root.resolve() or argv[4:6] != [mode, 'capture']:
        raise ValueError('command does not bind this capture root')
    if (cwd / argv[0]).resolve() != Path(producer['binary_path']).resolve():
        raise ValueError('command does not name the bound producer')
    request_paths = [str((cwd / arg).resolve()) for arg in argv[1:3]]
    tail = argv[5:8] + ([str((cwd / argv[8]).resolve())] if len(argv) == 9 else [])
    if len(argv) > 9:
        raise ValueError('unexpected recorded command arguments')
    return tuple(request_paths + tail)

def compare(reference, candidate, producer_record=None):
    reference, candidate = Path(reference), Path(candidate)
    if reference.resolve() == candidate.resolve():
        raise ValueError('use a distinct native reference')
    if not producer_record or producer_record.get('schema') != 'batch-spec-diagnostic-v1':
        raise ValueError('need the diagnostic producer build record')
    producers = producer_record['producers']
    if producers['native']['binary_sha256'] == producers['packed']['binary_sha256']:
        raise ValueError('native and packed producers must differ')
    if identity(reference, 'alone', producers['native']) != identity(candidate, 'packed', producers['packed']):
        raise ValueError('native and packed request settings differ')
    expected, actual = files(reference), files(candidate)
    if expected.keys() != actual.keys():
        raise ValueError('capture file sets differ')
    for name in expected:
        if digest(expected[name]) != digest(actual[name]):
            raise ValueError(f'capture differs: {name}')
    return dict(status='PASS', matched_files=len(expected), matched_bytes=sum(p.stat().st_size for p in expected.values()),
                requests=2, state_spans_per_frame=STATE_SPANS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--producer-record', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new comparison output.')
    report = compare(args.reference, args.candidate, json.loads(args.producer_record.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
