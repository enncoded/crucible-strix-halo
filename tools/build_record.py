#!/usr/bin/env python3
"""Write <binary>.build.json for an engine binary built from a source tree.

The record has the format of the train records (for example crucible-t12.build.json):
binary sha256, source commit, sha256 of every engine/kernels/bench source, compiler and
command. gates.py checks a given record against the binary that runs.

  tools/build_record.py --binary build/engine/crucible-lf --source-commit SHA --command "hipcc ..."
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from build_batch_gates import digest, source_hashes

HIPCC = Path('/opt/rocm/bin/hipcc')


def compiler_lines():
    out = subprocess.run([str(HIPCC), '--version'], capture_output=True, text=True, timeout=30, check=True).stdout
    return [line for line in out.splitlines() if line.startswith(('HIP version', 'AMD clang', 'Target'))]


def write_record(binary, root, source_commit, command):
    binary, root = Path(binary).absolute(), Path(root).absolute()
    path = binary.with_name(binary.name + '.build.json')
    if path.exists():
        raise FileExistsError(f'build record exists: {path}')
    record = {'binary': os.path.relpath(binary, root), 'binary_sha256': digest(binary),
              'source_commit': source_commit, 'source': f'working tree at {root}',
              'source_hashes': source_hashes(root), 'compiler': compiler_lines(), 'command': command,
              'cwd': str(root)}
    # A record describes one build; never replace one in place.
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, default=Path.cwd())
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--command', required=True, help='the exact build command')
    args = ap.parse_args()
    print(write_record(args.binary, args.source_root, args.source_commit, args.command))
    return 0


if __name__ == '__main__':
    sys.exit(main())
