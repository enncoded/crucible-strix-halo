#!/usr/bin/env python3
"""Time the phases of one engine process without changing the engine.

Runs one command, stamps each stderr line on arrival (the engine writes stderr
unbuffered), samples /proc/<pid>/io and /proc/meminfo every 0.1 s, and records
the page-cache residency of the checkpoint files before the run (fincore).
Writes one JSON file. Run it inside a GPU lease.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def fincore(checkpoint):
    files = sorted(str(p) for p in Path(checkpoint).glob('*.safetensors'))
    done = subprocess.run(['fincore', '--bytes', '--noheadings', '--raw', '-o', 'RES,SIZE,FILE', *files],
                          capture_output=True, text=True, check=False)
    rows = [line.split(' ', 2) for line in done.stdout.splitlines() if line.strip()]
    return {Path(f).name: {'resident': int(r), 'bytes': int(s)} for r, s, f in rows}


def meminfo():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemAvailable', 'Cached', 'MemFree'):
            values[name] = int(value.split()[0]) * 1024
    return values


def proc_io(pid):
    try:
        text = Path(f'/proc/{pid}/io').read_text()
    except OSError:
        return None
    return {k: int(v) for k, v in (line.split(': ') for line in text.splitlines())
            if k in ('rchar', 'read_bytes')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, type=Path)
    # The profiler kills the command itself, so the sampled PID is the engine, not a wrapper.
    ap.add_argument('--timeout', type=float, default=150)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    before = {'fincore': fincore(a.checkpoint), 'meminfo': meminfo()}
    t0 = time.monotonic()
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, 'HSA_USERPTR_FOR_PAGED_MEM': os.environ.get('HSA_USERPTR_FOR_PAGED_MEM', '0')})
    lines, samples = [], []

    def read(stream, name):
        for line in stream:
            lines.append({'t': time.monotonic() - t0, 'stream': name, 'line': line.rstrip('\n')})

    readers = [threading.Thread(target=read, args=(child.stderr, 'stderr')),
               threading.Thread(target=read, args=(child.stdout, 'stdout'))]
    for reader in readers:
        reader.start()
    while child.poll() is None:
        if time.monotonic() - t0 > a.timeout:
            child.kill()
        samples.append({'t': time.monotonic() - t0, 'io': proc_io(child.pid), **meminfo()})
        time.sleep(0.1)
    end = time.monotonic() - t0
    for reader in readers:
        reader.join()
    a.out.write_text(json.dumps({'command': cmd, 'returncode': child.returncode, 'wall_s': end,
                                 'before': before, 'lines': sorted(lines, key=lambda x: x['t']),
                                 'samples': samples}, indent=1) + '\n')
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
