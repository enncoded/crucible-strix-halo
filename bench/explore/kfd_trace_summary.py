#!/usr/bin/env python3
"""Summarize a kfd_trace.so log: KFD allocations by kind, the USERPTR BOs
(movable host pages mapped by the GPU) with their lifetimes and call sites,
and the queue evictions by trigger.

usage: kfd_trace_summary.py LOG [--frames N]
"""
import argparse
import collections
import re

TRIGGERS = {0: 'SVM', 1: 'USERPTR', 2: 'TTM', 3: 'SUSPEND', 4: 'CRIU'}


def parse(path):
    allocs, frees, svm, events, samples = {}, {}, [], [], []
    t0 = boot0 = None
    with open(path) as f:
        for line in f:
            if line.startswith('# kfd_trace'):
                m = re.search(r't0=([\d.]+) boottime=([\d.]+)', line)
                if m:
                    t0, boot0 = float(m.group(1)), float(m.group(2))
            elif line.startswith('A '):
                kv = dict(re.findall(r'(\w+)=(\S+)', line))
                frames = re.split(r' off=\S+', line, maxsplit=1)[1].split()
                allocs[kv['handle']] = dict(t=float(line.split()[1]), kind=kv['kind'], va=int(kv['va'], 16),
                                            size=int(kv['size']), off=int(kv['off'], 16), frames=frames)
            elif line.startswith('F '):
                kv = dict(re.findall(r'(\w+)=(\S+)', line))
                frees[kv['handle']] = float(line.split()[1])
            elif line.startswith('V '):
                svm.append(line.strip())
            elif line.startswith('E '):
                parts = line.split()
                events.append((float(parts[1]), parts[2:]))
            elif line.startswith('S '):
                kv = dict(re.findall(r'(\w+)=(-?\d+)', line))
                samples.append((float(line.split()[1]), {k: int(v) for k, v in kv.items()}))
    return t0, boot0, allocs, frees, svm, events, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--frames', type=int, default=8)
    a = ap.parse_args()
    t0, boot0, allocs, frees, svm, events, samples = parse(a.log)
    end = max([x['t'] for x in allocs.values()] + list(frees.values()) + [e[0] for e in events] + [0])

    by = collections.defaultdict(lambda: [0, 0])
    for x in allocs.values():
        by[x['kind']][0] += 1
        by[x['kind']][1] += x['size']
    print('KFD allocations by kind (count, GiB):')
    for k, (n, b) in sorted(by.items()):
        print(f'  {k:9s} {n:6d} {b / 2**30:9.3f}')

    up = sorted((x['t'], h, x) for h, x in allocs.items() if x['kind'] == 'USERPTR')
    print(f'USERPTR BOs: {len(up)}')
    sites = collections.defaultdict(lambda: [0, 0, 0.0])
    for t, h, x in up:
        life = frees.get(h, end) - t
        key = ' '.join(x['frames'][:a.frames])
        sites[key][0] += 1
        sites[key][1] += x['size']
        sites[key][2] = max(sites[key][2], life)
    for key, (n, b, life) in sorted(sites.items(), key=lambda kv: -kv[1][1]):
        print(f'  {n:5d} BOs {b / 2**20:10.1f} MiB  longest life {life:8.3f} s  {key}')
    live = [(t - (t0 or 0), x['size'], frees.get(h)) for t, h, x in up if h not in frees]
    print(f'  USERPTR live at end of log: {len(live)} BOs, {sum(s for _, s, _ in live) / 2**20:.1f} MiB')

    print(f'SVM ioctls: {len(svm)}')
    for s in svm[:10]:
        print('  ' + s[:200])

    ev = collections.Counter()
    for t, p in events:
        if p and p[0] == '9' and len(p) >= 5:
            ev[TRIGGERS.get(int(p[4]), p[4])] += 1
        elif p:
            ev['event ' + p[0]] += 1
    print('SMI events:', dict(ev))
    if samples:
        first, last = samples[0][1], samples[-1][1]
        mig = sum(s[1].get('pgmigrate_success', 0) for s in samples[1:])
        print(f"evicted_ms {first.get('evicted_ms')} -> {last.get('evicted_ms')}; pgmigrate_success during run {mig}")


if __name__ == '__main__':
    main()
