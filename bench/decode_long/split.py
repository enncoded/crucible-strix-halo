#!/usr/bin/env python3
"""Per-call decode breakdown from a rocprofv3 kernel trace (kernels_kernel_trace.csv).

A call starts at EmbedKernel (trunk forward, R = grid y) or MtpPrepKernel (MTP step,
R = grid x / 1024). Prefill calls (R > 8) are skipped. For each decode call type the
script prints mean ms per call by kernel class, the busy sum, and the wall span (first
kernel start to the next call's first kernel start), so span - busy is launch and sync time.
Usage: split.py TRACE.csv [--names]
"""
import csv, re, sys
from collections import defaultdict

CLASSES = [
    ("qsa_score", r"^QsaScore"), ("qsa_top", r"^QsaTop"), ("qsa_attn", r"^Qsa(GqaSplit|Merge)"),
    ("qsa_prep", r"^Qsa"), ("gdn", r"^Gdn|^PersistentGdn"), ("moe", r"^Moe|^CountExperts"),
    ("hc", r"^Hc"), ("head", r"^Argmax|^DraftChoice|^DraftDense"), ("lookahead", r"^Lookahead"),
    ("mtp_glue", r"^Mtp"), ("copy", r"rocclr|copyBuffer|fillBuffer"),
]

def kclass(name):
    base = re.sub(r"^(void )?(crucible::)?", "", name)
    base = re.sub(r"^(prefill::)", "", base)
    for c, pat in CLASSES:
        if re.search(pat, base): return c
    return "dense"  # LinIn, GEMV groups, Embed, router dense, PLE

def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            n = r["Kernel_Name"]
            short = re.sub(r"\(.*$", "", n)
            short = re.sub(r"^(void )?", "", short)
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), short,
                         int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Workgroup_Size_X"])))
    rows.sort()
    calls = []
    for k in rows:
        s, e, name, gx, gy, wx = k
        bare = re.sub(r"^.*::", "", re.sub(r"<.*$", "", name))
        if bare == "EmbedKernel": calls.append(["fwd", gy, []])
        elif bare == "MtpPrepKernel": calls.append(["mtp", gx // wx, []])
        if calls: calls[-1][2].append(k)
    # decode calls: after the last call with R > 8
    last_big = max((i for i, c in enumerate(calls) if c[1] > 8), default=-1)
    dec = calls[last_big + 1:]
    stats = defaultdict(lambda: {"n": 0, "span": 0.0, "busy": 0.0, "cls": defaultdict(float), "names": defaultdict(float)})
    for i, (typ, R, ks) in enumerate(dec):
        if i + 1 >= len(dec): break  # last call has no next start
        key = f"{typ} R={R}"
        st = stats[key]
        st["n"] += 1
        st["span"] += (dec[i + 1][2][0][0] - ks[0][0]) / 1e6
        for s, e, name, *_ in ks:
            c = kclass(re.sub(r"^.*::", "", name))
            d = (e - s) / 1e6
            st["cls"][c] += d
            st["names"][name[:70]] += d
            if c != "lookahead": st["busy"] += d
    order = ["dense", "moe", "hc", "gdn", "qsa_prep", "qsa_score", "qsa_top", "qsa_attn", "head", "mtp_glue", "copy", "lookahead"]
    print(f"{'call':10s} {'n':>4s} " + " ".join(f"{c:>9s}" for c in order) + f" {'busy':>7s} {'span':>7s} {'gap':>6s}")
    tot = defaultdict(float)
    for key in sorted(stats):
        st = stats[key]; n = st["n"]
        vals = [st["cls"][c] / n for c in order]
        print(f"{key:10s} {n:4d} " + " ".join(f"{v:9.3f}" for v in vals) +
              f" {st['busy']/n:7.3f} {st['span']/n:7.3f} {(st['span']-st['busy'])/n:6.3f}")
        for c in order: tot[c] += st["cls"][c]
        tot["busy"] += st["busy"]; tot["span"] += st["span"]
    print(f"{'total ms':10s} {'':4s} " + " ".join(f"{tot[c]:9.1f}" for c in order) + f" {tot['busy']:7.1f} {tot['span']:7.1f} {tot['span']-tot['busy']:6.1f}")
    if "--names" in sys.argv:
        agg = defaultdict(float)
        for st in stats.values():
            for k, v in st["names"].items(): agg[k] += v
        for k, v in sorted(agg.items(), key=lambda x: -x[1])[:40]: print(f"{v:9.2f} {k}")

main()
