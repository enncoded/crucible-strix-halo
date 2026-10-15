#!/usr/bin/env python3
"""Per-request KFD activity for a suite worker run traced with kfd_trace.so.

usage: first_request.py OUT_JSONL TRACE_LOG [ERR_LOG]
OUT_JSONL lines carry a CLOCK_MONOTONIC prefix (bench/explore/ts.py). For each
request window it prints the prefill time, SVM ioctls (new GPU-mapped host
ranges), allocations, queue evictions with their total time, and migrations.
"""
import json
import re
import sys


def windows(path):
    out = []
    for line in open(path):
        stamp, _, body = line.partition(' ')
        try:
            d = json.loads(body)
        except ValueError:
            continue
        if d.get('event') == 'ready':
            out.append(('ready', None, float(stamp), d))
        elif d.get('event') == 'result':
            end = float(stamp)
            out.append((d['name'], end - d['prefill_seconds'] - d['decode_seconds'], end, d))
    return out


def trace(path):
    svm, alloc, ev, samples = [], [], [], []
    cur = None
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == 'V':
            svm.append((float(p[1]), int(re.search(r'size=(\d+)', line).group(1))))
        elif p[0] == 'A':
            alloc.append((float(p[1]), int(re.search(r'size=(\d+)', line).group(1))))
        elif p[0] == 'E' and len(p) > 3:
            t = int(p[3]) / 1e9
            if p[2] == '9' and cur is None:
                cur = t
            elif p[2] == 'a' and cur is not None:
                ev.append((cur, t))
                cur = None
        elif p[0] == 'S':
            kv = dict(re.findall(r'(\w+)=(-?\d+)', line))
            samples.append((float(p[1]), int(kv.get('pgmigrate_success', 0))))
    return svm, alloc, ev, samples


def main():
    reqs = windows(sys.argv[1])
    svm, alloc, ev, samples = trace(sys.argv[2])
    start = samples[0][0] if samples else 0
    prev = start
    print('window             start_s  prefill_s  svm_n  svm_MiB  alloc_n  alloc_MiB  evict_n  evict_ms  migrated')
    for name, begin, end, d in reqs:
        lo = prev if begin is None else begin
        inside = lambda t: lo <= t <= end
        s = [b for t, b in svm if inside(t)]
        a = [b for t, b in alloc if inside(t)]
        e = [(x, y) for x, y in ev if x <= end and y >= lo]
        ems = sum(min(y, end) - max(x, lo) for x, y in e) * 1e3
        mig = sum(m for t, m in samples if inside(t))
        pf = d.get('prefill_seconds', d.get('load_seconds', 0))
        print(f'{name:16s} {lo - start:8.2f} {pf:10.3f} {len(s):6d} {sum(s) / 2**20:8.1f} {len(a):8d} '
              f'{sum(a) / 2**20:10.1f} {len(e):8d} {ems:9.1f} {mig:9d}')
        prev = end


if __name__ == '__main__':
    main()
