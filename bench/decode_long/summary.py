#!/usr/bin/env python3
"""One line per trace directory: decode rate, per-step and per-verify-forward costs, the QSA
select share (scores + top-512), launch gaps. Usage: summary.py DIR..."""
import csv, re, sys, statistics
from collections import defaultdict

def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        n = re.sub(r"\(.*$", "", r["Kernel_Name"]); n = re.sub(r"^(void )?(crucible::)?", "", n)
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), n, int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Workgroup_Size_X"])))
    rows.sort(); return rows

print("| Run | tok/s | fwd | acc/fwd | step ms | verify ms | verify busy ms | score ms | top ms | select % of verify | verify gap ms | MTP calls/step | MTP ms/call |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in sys.argv[1:]:
    log = open(d + "/log").read()
    m = re.search(r"decode (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\), (\d+) verify forwards, ([\d.]+) drafts", log)
    toks, dec_s, tps, fwds, acc = int(m[1]), float(m[2]), float(m[3]), int(m[4]), float(m[5])
    rows = [k for k in load(d + "/kernels_kernel_trace.csv") if not k[2].startswith("Lookahead")]
    calls = []
    for k in rows:
        b = re.sub(r"<.*$", "", k[2])
        if b == "EmbedKernel": calls.append(["fwd", k[4], []])
        elif b == "MtpPrepKernel": calls.append(["mtp", k[3] // k[5], []])
        if calls: calls[-1][2].append(k)
    last = max(i for i, c in enumerate(calls) if c[1] > 8)
    dec = calls[last + 1:]
    v = defaultdict(list); mt = []
    for i in range(len(dec) - 1):
        typ, R, ks = dec[i]
        span = (dec[i + 1][2][0][0] - ks[0][0]) / 1e6
        busy = sum(e - s for s, e, *_ in ks) / 1e6
        sc = sum(e - s for s, e, n, *_ in ks if n.startswith("QsaScore")) / 1e6
        tp = sum(e - s for s, e, n, *_ in ks if n.startswith("QsaTop")) / 1e6
        if typ == "fwd": v["span"].append(span); v["busy"].append(busy); v["sc"].append(sc); v["tp"].append(tp)
        else: mt.append(span)
    nf = len(v["span"]); med = statistics.median
    step = dec_s * 1e3 / fwds
    print(f"| {d.rstrip('/').split('/')[-1]} | {tps:.1f} | {fwds} | {acc:.2f} | {step:.1f} | {med(v['span']):.1f} | {med(v['busy']):.1f} | "
          f"{statistics.mean(v['sc']):.2f} | {statistics.mean(v['tp']):.2f} | {100*sum(v['sc']+v['tp'])/sum(v['span']):.1f} | "
          f"{med([s-b for s,b in zip(v['span'],v['busy'])]):.2f} | {len(mt)/nf:.2f} | {med(mt):.2f} |")
