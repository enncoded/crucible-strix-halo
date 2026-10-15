#!/usr/bin/env python3
"""Per-tensor activation traffic of one 4096-row prefill chunk (v33 dataflow).

Bytes per row per layer for each producer -> consumer round trip, read from
the prefill work (engine/runtime/prefill_engine.inc, prefill_hc.h, prefill_moe.h,
prefill_linear.h). GDN-core and QSA internals use the v25 roofline table
(18.8 GB and 5.7 GB per 4K) because their scratch is owned elsewhere.
Each cut lists the bytes it removes and whether it keeps the bits.
"""

ROWS, ATTN, GDN = 4096, 12, 36
HID, WIDE, RANK, SLOTS, INTER = 2560, 10240, 320, 11, 640
F32, F16 = 4, 2
DRAM = 245.5e9

H = WIDE * F32  # fp32 residual stack, 4 streams
x32, x16 = HID * F32, HID * F16
dots = (RANK + 4) * 4 * F32
t = 2 * RANK * F16


def hc(write_xh):
    return {
        "H read (dots)": H,
        "dots+ss": 2 * dots,
        "thi/tlo": 2 * t,
        "H read (mix)": H,
        "x fp32 write": x32,
        "xh write (mix)": x16 if write_xh else 0,
    }


def layer(attn):
    jobs = 4 if attn else 2
    outs = (12288 + 512 + 512 + 640) if attn else (10240 + 6144)
    d = {}
    for k, v in hc(False).items():
        d["hca " + k] = v
    d["lin_in x read"] = jobs * x32
    d["lin_in xh w+r"] = jobs * x16 * 2
    d["proj out fp32 w+r"] = outs * F32 * 2
    if attn:
        d["qsa internal"] = 5.7e9 / (ATTN * ROWS) - outs * F32
    else:
        d["gdn_ab pack+read"] = x32 + 2 * x16 + 96 * F32
        d["gdn internal"] = 18.8e9 / (GDN * ROWS) - outs * F32
    d["oproj xpack w+r"] = 6144 * F16 * 2
    d["oproj H r+w"] = 2 * H
    for k, v in hc(True).items():
        d["hcm " + k] = v
    d["router xh read + logits"] = x16 + 2 * 513 * F32
    d["fx xh gather read"] = SLOTS * x16
    d["fx xg/xu w+r"] = 2 * SLOTS * x16 * 2
    d["gate/up pg/pu w+r"] = 2 * SLOTS * INTER * F32 * 2
    d["mid xd w+r"] = SLOTS * INTER * F16 * 2
    d["down pd w+r"] = SLOTS * HID * F32 * 2
    d["combine H r+w"] = 2 * H
    return d


def total(f):
    return ROWS * (ATTN * f(layer(True)) + GDN * f(layer(False)))


GB = lambda b: b / 1e9
all_bytes = total(lambda d: sum(d.values()))
print(f"modelled activations per 4K: {GB(all_bytes):.1f} GB (v25 measured 239)")
keys = list(dict.fromkeys(list(layer(True)) + list(layer(False))))
rows = sorted(((GB(total(lambda d, k=k: d.get(k, 0))), k) for k in keys), reverse=True)
for gb, k in rows[:14]:
    print(f"  {k:28s} {gb:6.1f} GB {gb * 1e9 / DRAM * 1e3:6.1f} ms")

# Cuts: (name, bytes saved per row for attn, for GDN, exact?)
cuts = [
    ("FX transform in gate/up A prologue", lambda a: 2 * SLOTS * x16 * 2, True),
    ("mid in gate/up epilogue, xd only", lambda a: 2 * SLOTS * INTER * F32 * 2, True),
    ("mid in gate/up epilogue, h fp32", lambda a: 2 * SLOTS * INTER * F32 * 2
     + SLOTS * INTER * F16 * 2 - SLOTS * INTER * F32 * 2, True),
    ("pd stored fp16", lambda a: SLOTS * HID * F16 * 2, False),
    ("HC mix re-reads H from MALL", lambda a: 2 * H, True),
    # Lazy residual: o-proj and combine write a 10 KB delta; hca dots
    # materialises H once; hcm and combine rebuild H_new from H + delta.
    ("lazy residual (deltas, not H)", lambda a: (H * 2 + H * 4 + H * 2)
     - (2 * H + 2 * x32 + H + x32 + H + x32 + H + x32 + x32), True),
    # hca mix writes xh not x; LinIn reads fp16; hcm drops x; a/b pack goes.
    ("x as fp16 only (fp16-exact)", lambda a: (x16 + 4 * x16 + x32) if a
     else (x16 + 2 * x16 + x32 + x32 + x16), True),
    ("lin_in in GEMM A prologue", lambda a: (4 if a else 2) * x16 * 2, True),
    ("proj outputs stored fp16", lambda a: ((12288 + 1664) if a else 16384) * F16 * 2, False),
]
print("\ncut                                   GB/4K  ms@DRAM  exact")
for name, f, exact in cuts:
    gb = GB(ROWS * (ATTN * f(True) + GDN * f(False)))
    print(f"  {name:36s} {gb:5.1f} {gb * 1e9 / DRAM * 1e3:7.1f}  {'yes' if exact else 'no (KL)'}")
