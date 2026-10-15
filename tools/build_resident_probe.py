#!/usr/bin/env python3
"""Build a source-bound producer that reuses one model for each capture role."""
import argparse
from pathlib import Path
import subprocess

import build_long_probe as builder

RESIDENT_ORIGIN='crucible resident production+clipped capture v1'
RESIDENT_FILES=('bench/resident_probe.hip','bench/resident_probe_runtime.h','bench/long_probe_runtime.h')


def build(args):
    return builder.build(args,probe_files=RESIDENT_FILES,capture_origin=RESIDENT_ORIGIN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--engine-build-record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hipcc',default='/opt/rocm/bin/hipcc')
    try: build(parser.parse_args())
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        parser.exit(1,f'resident probe build: {error}\n')


if __name__=='__main__': main()
