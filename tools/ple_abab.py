#!/usr/bin/env python3
"""Interleaved A/B timing for CRUCIBLE_PLE_HOST inside one GPU lease.

Usage: ple_abab.py ENGINE CKPT IDS OUT.jsonl --configs off,hot,off,hot [--budget 280] [-- extra gen args]
       ple_abab.py ... -- pfbench RUN...   (crucible pfbench; each RUN reports its second, rows-hot pass)
Configs: off (resident table), hot (host gather, page cache as left by the previous run),
cold (host gather after POSIX_FADV_DONTNEED on the n-gram file, so every row is a first touch),
warm (host gather after build/bench/ple_host_cold reads the prompt's rows into the page cache).
Each run records prefill and decode tok/s from the engine, the busy sclk mean (samples above
1 GHz from amdgpu hwmon, every 20 ms), and the free device memory the engine reports after warmup.
"""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path

HWMON = next(Path('/sys/class/drm/card1/device/hwmon').glob('hwmon*')) / 'freq1_input'


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


WARM_TOOL = Path(__file__).resolve().parents[1] / 'build/bench/ple_host_cold'
extra_ids = None


def warm(ckpt, ids_file):
    # Row geometry of the two packs.
    geometry = ('26736', '102', '320001536') if '3.05' in str(ckpt) else ('6328', '122', '320001536')
    subprocess.run([str(WARM_TOOL), str(Path(ckpt) / 'ngram_embedding.safetensors'), *geometry, ids_file],
                   check=True, capture_output=True)


def sample(stop, out):
    while not stop.is_set():
        try:
            out.append(int(HWMON.read_text()) / 1e6)
        except OSError:
            pass
        time.sleep(0.02)


def disk_read_mb_s(seconds=1.0):
    """Other processes' NVMe reads just before a run: cold rows queue behind them."""
    def sectors():
        return int(Path('/sys/block/nvme0n1/stat').read_text().split()[2])
    a = sectors()
    time.sleep(seconds)
    return round((sectors() - a) * 512 / seconds / 1e6, 1)


def run(engine, ckpt, ids, config, extra):
    env = dict(os.environ, CRUCIBLE_PLE_HOST='0' if config == 'off' else '1')
    if config == 'cold':
        drop(Path(ckpt) / 'ngram_embedding.safetensors')
    if config == 'warm':  # the prompt's rows resident, as in a long-lived server
        warm(ckpt, extra_ids)
    disk_before = disk_read_mb_s()
    samples, stop = [], threading.Event()
    t = threading.Thread(target=sample, args=(stop, samples))
    t.start()
    start = time.time()
    if extra and extra[0] == 'pfbench':  # pfbench RUN...: the second pass of each run, rows hot
        cmd = [engine, 'pfbench', ckpt, '--prefill', '4096', *extra[1:]]
    else:
        cmd = [engine, 'gen', ckpt, '--ids', ids, *extra]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    pf = re.findall(r'^(\S+): (\d+) tok .* ([\d.]+) s ([\d.]+) tok/s$', p.stdout, re.M)
    stop.set()
    t.join()
    m = re.search(r'prefill (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\), decode (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)',
                  p.stdout + p.stderr)
    free = re.search(r'memory ready: device free (\d+)', p.stderr)
    avail = re.search(r'memory ready: MemAvailable:\s+(\d+) kB', p.stderr)
    busy = [s for s in samples if s > 1000]
    # CRUCIBLE_PLE_TRACE lines: time the forward waited for the CPU gather.
    trace = [(int(a), float(b), float(c)) for a, b, c in
             re.findall(r'ple host gather: (\d+) rows, ([\d.]+) ms since start, ([\d.]+) ms waited', p.stderr)]
    ids_line = re.search(r'^ids: (.*)$', p.stdout, re.M)
    return {
        'config': config, 'rc': p.returncode, 'wall_s': round(time.time() - start, 2),
        'prefill_tok_s': float(m[3]) if m else None, 'prefill_s': float(m[2]) if m else None,
        'decode_tok_s': float(m[6]) if m else None, 'decode_tokens': int(m[4]) if m else None,
        'sclk_busy_mhz': round(sum(busy) / len(busy), 1) if busy else None, 'sclk_samples': len(busy),
        'device_free': int(free[1]) if free else None, 'mem_available_kb': int(avail[1]) if avail else None,
        'ids_sha': __import__('hashlib').sha256(ids_line[1].encode()).hexdigest()[:16] if ids_line else None,
        'disk_read_mb_s_before': disk_before,
        'pfbench': [dict(tag=t, tokens=int(n), s=float(x), tok_s=float(y)) for t, n, x, y in pf],
        'gather_big_ms': [round(b, 3) for r, b, _ in trace if r >= 64],
        'wait_big_ms': [round(c, 3) for r, _, c in trace if r >= 64],
        'wait_small_ms_sum': round(sum(c for r, _, c in trace if r < 64), 3),
        'stderr_tail': p.stderr[-400:] if p.returncode else '',
    }


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    engine, ckpt, ids_file, out = argv[:4]
    configs = argv[argv.index('--configs') + 1].split(',')
    budget = float(argv[argv.index('--budget') + 1]) if '--budget' in argv else 280
    global extra_ids
    extra_ids = ids_file
    ids = ','.join(re.findall(r'\d+', Path(ids_file).read_text()))
    began, longest = time.time(), 0.0
    with open(out, 'a') as f:
        for config in configs:
            if time.time() - began + longest * 1.2 > budget:
                print('stop: lease budget', flush=True)
                break
            r = run(engine, ckpt, ids, config, extra)
            longest = max(longest, r['wall_s'])
            r['time'] = time.strftime('%H:%M:%S')
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
