#!/usr/bin/env python3
"""Per-op time of one verify forward from a verify_rows kernel trace.

  trace_verify.py KERNEL_TRACE_CSV [MODES] [VERIFY_ROWS_STDOUT]

verify_rows launches TraceMarkKernel with grid R + 16 * mode (32 threads per
block) before each timed forward. This script takes the kernels between one
mark and the first idle gap over 0.5 ms (the host sync after the forward),
groups them by kernel name (template arguments dropped), and prints the
median over reps of the busy time per name, the forward span (first start to
last end) and the gaps. Lookahead kernels run on a side stream: they count
in their own line, not in the busy sum.
MODES: the comma list given to verify_rows (default base,sched,ring,ringla).
Copy kernels are left out (the state restore before the next mark).
Each mode also lists the launches and the idle gap before them, charged to the kernel
that ends the gap (launch_gaps).
With VERIFY_ROWS_STDOUT (its "bytes" line and distinct-expert counts), each
op class also gets GB/s and the percent of 245.5 GB/s.
"""
import csv
import os
import re
import statistics
import sys
from collections import defaultdict


# verify_rows marks R + 16 * mode; packed_rows marks rows + 64 * split.
TAG_BASE = int(os.environ.get("TAG_BASE", "16"))


def short(name):
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"\(.*$", "", name)
    base = re.sub(r"<.*$", "", name)
    base = base.split("::")[-1]
    return base


def launch_gaps(forward):
    """Launches per kernel name and the idle ns before each launch, charged to that kernel.

    forward: (start, end, name) of one forward in start order. Lookahead kernels run on a
    side stream, so they neither open nor close a gap. The first kernel gets no gap.
    """
    launches, gaps = defaultdict(int), defaultdict(int)
    last_end = None
    for s, e, name in forward:
        n = short(name)
        if n.startswith("Lookahead"):
            continue
        launches[n] += 1
        gaps[n] += 0 if last_end is None else max(0, s - last_end)
        last_end = e if last_end is None else max(last_end, e)
    return dict(launches), dict(gaps)


def main():
    path = sys.argv[1]
    modes = (sys.argv[2] if len(sys.argv) > 2 else "base,sched,ring,ringla").split(",")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r["Grid_Size_X"])))
    rows.sort()
    # runs[(mode, R)] = list of {name: ns} per rep, plus span and sum.
    runs = defaultdict(list)
    i = 0
    while i < len(rows):
        s, e, name, grid = rows[i]
        if "TraceMarkKernel" not in name:
            i += 1
            continue
        tag = grid // 32
        R, m = tag % TAG_BASE, tag // TAG_BASE
        j = i + 1
        per = defaultdict(int)
        forward = []
        t0 = t1 = None
        while j < len(rows) and "TraceMarkKernel" not in rows[j][2]:
            s2, e2, n2, _ = rows[j]
            if t1 is not None and s2 - t1 > 500_000:
                break
            if "rocclr_copyBuffer" in n2:
                j += 1
                continue
            per[short(n2)] += e2 - s2
            forward.append((s2, e2, n2))
            t0 = s2 if t0 is None else t0
            t1 = e2 if t1 is None else max(t1, e2)
            j += 1
        if t0 is not None:
            runs[(m, R)].append((per, t1 - t0, *launch_gaps(forward)))
        i = j
    table = {}
    for (m, R) in sorted(runs):
        reps = runs[(m, R)][1:] or runs[(m, R)]  # the first rep warms up
        span = statistics.median(r[1] for r in reps) / 1e6
        names = set().union(*(r[0].keys() for r in reps))
        med = {n: statistics.median(r[0].get(n, 0) for r in reps) / 1e6 for n in names}
        busy = sum(t for n, t in med.items() if not n.startswith("Lookahead"))
        print(f"== mode={modes[m] if m < len(modes) else m} R={R} span_ms={span:.3f} busy_ms={busy:.3f} "
              f"gaps_ms={span - busy:.3f} reps={len(reps)}")
        for n, t in sorted(med.items(), key=lambda kv: -kv[1]):
            print(f"  {t:8.3f} {n}")
        # The gap before each launch, by the kernel that ends the gap (median over reps).
        gnames = set().union(*(r[3].keys() for r in reps))
        gap = {n: statistics.median(r[3].get(n, 0) for r in reps) / 1e6 for n in gnames}
        count = {n: statistics.median(r[2].get(n, 0) for r in reps) for n in gnames}
        print(f"   gaps by next kernel: launches={sum(count.values()):.0f} gap_ms={sum(gap.values()):.3f}")
        for n, t in sorted(gap.items(), key=lambda kv: -kv[1]):
            print(f"    {t:8.3f} ms {count[n]:5.0f} launches {t * 1e3 / max(1, count[n]):6.2f} us/launch {n}")
        table[(m, R)] = dict(med, span=span, busy=busy)
    # One matrix per mode: kernel names down, R across (ms per forward).
    for m in sorted({k[0] for k in table}):
        Rs = sorted(R for mm, R in table if mm == m)
        names = sorted({n for (mm, R), d in table.items() if mm == m for n in d},
                       key=lambda n: -max(table[(m, R)].get(n, 0) for R in Rs))
        print(f"\n## mode={modes[m] if m < len(modes) else m}: ms per forward")
        print("kernel," + ",".join(f"R{R}" for R in Rs))
        for n in names:
            print(n + "," + ",".join(f"{table[(m, R)].get(n, 0):.3f}" for R in Rs))
    if len(sys.argv) > 3:
        print_rates(table, modes, *byte_classes(sys.argv[3]))


CLASSES = {
    "dense": ("LinPolicyKernel", "LinRingRowsKernel", "LinRingFlowKernel", "LinGemvKernel", "LinInKernel", "LinOutKernel"),
    "moe": ("MoeFusedKernel",),
    "hc": ("HcWmmaKernel", "HcDotsKernel", "HcMixKernel"),
    "router_ple": ("DenseGemvKernel",),
    "gdn": ("GdnKernel",),
}


def byte_classes(path):
    """Weight bytes per op class and distinct experts per R from verify_rows stdout."""
    b, experts = {}, {}
    for line in open(path):
        if line.startswith("bytes "):
            b = {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}
        parts = line.strip().split(",")
        if len(parts) == 7 and parts[1].isdigit():
            experts[int(parts[1])] = float(parts[4])
    return b, experts


def print_rates(table, modes, b, experts):
    print("\n## GB/s by op class (pct of 245.5)")
    for (m, R), d in sorted(table.items()):
        bytes_ = {
            "dense": b["attn_in"] + b["attn_out"] + b["head"],
            "moe": b["shared"] + experts.get(R, 0) * 48 * b["routed_per_expert_layer"],
            "hc": b["hc"],
            "router_ple": b["router"] + b["ple"],
            "gdn": b["gdn_state_rw"],
        }
        cells = []
        for c, names in CLASSES.items():
            t = sum(v for n, v in d.items() if n in names)
            if t > 0:
                gbs = bytes_[c] / t / 1e6
                cells.append(f"{c}={t:.2f}ms/{gbs:.0f}GB/s/{gbs / 245.5 * 100:.0f}%")
        total = sum(bytes_.values())
        print(f"{modes[m] if m < len(modes) else m} R={R} busy={d['busy']:.2f}ms floor={total / 245.5e6:.2f}ms "
              + " ".join(cells))


if __name__ == "__main__":
    main()
