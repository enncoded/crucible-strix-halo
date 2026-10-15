#!/usr/bin/env python3
"""Supplementary byte proof for long-prefill logits and indexer captures. Not shared gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

LAYERS = tuple(range(3, 48, 4))
VOCAB = 248320


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equal(left, right, count):
    for path in (left, right):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != count:
            raise ValueError(f'wrong artifact size or type: {path}')
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            if x != y:
                raise ValueError(f'byte difference: {left} versus {right}')
            if not x:
                return sha(left)


def capture(base, candidate, layer):
    left, right = base/f'qsa_{layer}', candidate/f'qsa_{layer}'
    metadata = [Path(str(p)+'.meta').read_text() for p in (left, right)]
    if metadata[0] != metadata[1]:
        raise ValueError(f'metadata difference: layer {layer}')
    pos, rows = map(int, metadata[0].split())
    if pos < 0 or rows < 1 or rows > 2048 or pos+rows > 262144:
        raise ValueError(f'invalid metadata: layer {layer}')
    sizes = {'.idx.f32': rows*640*4, '.pool.f32': ((pos+rows+127)//128)*4096*4,
             '.sel.i32': rows*512*4, '.nsel.i32': rows*4}
    hashes = {name: equal(Path(str(left)+name), Path(str(right)+name), size)
              for name, size in sizes.items()}
    counts = [v[0] for v in struct.iter_unpack('<i', Path(str(left)+'.nsel.i32').read_bytes())]
    selected = [v[0] for v in struct.iter_unpack('<i', Path(str(left)+'.sel.i32').read_bytes())]
    for r, count in enumerate(counts):
        nb = (pos+r+1)//4
        if count != (-1 if nb <= 512 else 512):
            raise ValueError(f'invalid selected count: layer {layer} row {r}')
        if count == 512:
            ids = selected[r*512:(r+1)*512]
            if any(b < 0 or b >= nb for b in ids) or any(a >= b for a, b in zip(ids, ids[1:])):
                raise ValueError(f'invalid selected IDs: layer {layer} row {r}')
    return {'pos0': pos, 'rows': rows, 'sha256': hashes}


def generated_ids(root, tokens):
    lines = [line[4:] for line in root.joinpath('run.log').read_text().splitlines() if line.startswith('ids:')]
    if len(lines) != 1:
        raise ValueError('need one generated IDs record')
    try:
        values = list(map(int, lines[0].split()))
    except ValueError:
        raise ValueError('invalid generated IDs') from None
    if len(values) != tokens or any(v < 0 or v >= VOCAB for v in values):
        raise ValueError('invalid generated IDs count or range')
    return values


def run(base, candidate, tokens=1):
    if tokens < 1 or tokens > 262144:
        raise ValueError('invalid expected token count')
    identities = [p.joinpath('input.sha256').read_text().splitlines() for p in (base, candidate)]
    binary = [lines[0].split()[0] for lines in identities]
    prompt = [lines[1].split()[0] for lines in identities]
    if any(len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in binary+prompt):
        raise ValueError('invalid provenance hashes')
    if binary[0] == binary[1] or prompt[0] != prompt[1]:
        raise ValueError('need independent binaries and identical prompt hashes')
    names = {f'qsa_{l}.meta' for l in LAYERS}
    for root in (base, candidate):
        if {p.name for p in root.joinpath('capture').glob('*.meta')} != names:
            raise ValueError(f'need all 12 QSA layers: {root}')
    ids = [generated_ids(root, tokens) for root in (base, candidate)]
    if ids[0] != ids[1]:
        raise ValueError('generated IDs differ')
    digest = equal(base/'logits.f32', candidate/'logits.f32', VOCAB*4)
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', (base/'logits.f32').read_bytes())):
        raise ValueError('nonfinite logits')
    records = {str(l): capture(base/'capture', candidate/'capture', l) for l in LAYERS}
    return {'status': 'EXACT', 'kind': 'supplementary, not shared gates', 'binary_sha256': binary,
            'prompt_sha256': prompt[0], 'generated_ids': ids[0], 'logits_sha256': digest, 'captures': records,
            'comparison_tool_sha256': sha(Path(__file__)), 'ended_ns': time.time_ns()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=1, help='Expected generated IDs count.')
    args = ap.parse_args()
    try:
        result = run(args.baseline, args.candidate, args.tokens)
        with args.output.open('x') as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write('\n')
    except (OSError, ValueError, IndexError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print(f"EXACT: supplementary logits and selection byte proof: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
