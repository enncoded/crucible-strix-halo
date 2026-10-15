#!/usr/bin/env python3
"""Idle time between main-stream kernels in decode calls (excludes LookaheadKernel).
Prints, per call type, mean ms of: inner gaps (between kernels of one call), the boundary gap
(last kernel end of a call to first kernel start of the next), the largest inner gaps by
kernel pair, and the kernel count per call. Usage: gaps.py TRACE.csv"""
import csv, re, sys
from collections import defaultdict
rows = []
for r in csv.DictReader(open(sys.argv[1])):
    n = re.sub(r"\(.*$", "", r["Kernel_Name"]); n = re.sub(r"^(void )?(crucible::)?", "", n)
    if n.startswith("LookaheadKernel"): continue
    rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), n, int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Workgroup_Size_X"])))
rows.sort()
calls = []
for k in rows:
    b = re.sub(r"<.*$", "", k[2])
    if b == "EmbedKernel": calls.append(["fwd", k[4], []])
    elif b == "MtpPrepKernel": calls.append(["mtp", k[3] // k[5], []])
    if calls: calls[-1][2].append(k)
last_big = max(i for i, c in enumerate(calls) if c[1] > 8)
dec = calls[last_big + 1:]
st = defaultdict(lambda: defaultdict(float)); pairs = defaultdict(float); npairs = defaultdict(int)
for i in range(len(dec) - 1):
    typ, R, ks = dec[i]; key = f"{typ} R={R}" if typ == "mtp" else "fwd"
    s = st[key]; s["n"] += 1; s["kernels"] += len(ks)
    for a, b in zip(ks, ks[1:]):
        g = max(0, b[0] - a[1]) / 1e6; s["inner"] += g
        pairs[(key, re.sub(r"<.*$", "", a[2]), re.sub(r"<.*$", "", b[2]))] += g
    s["boundary"] += max(0, dec[i + 1][2][0][0] - ks[-1][1]) / 1e6
    s["busy"] += sum(e - b for b, e, *_ in ks) / 1e6
for key, s in sorted(st.items()):
    n = s["n"]
    print(f"{key:10s} n={int(n):4d} kernels={s['kernels']/n:6.1f} busy={s['busy']/n:7.3f} inner_gap={s['inner']/n:6.3f} boundary_gap={s['boundary']/n:6.3f} ms")
print("largest inner gap pairs (total ms):")
for (k, a, b), g in sorted(pairs.items(), key=lambda x: -x[1])[:25]:
    print(f"  {g:8.2f} {k:10s} {a} -> {b}")
