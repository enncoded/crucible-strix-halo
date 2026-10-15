#!/usr/bin/env python3
"""Probe: can the plain fp16 weights (HC mixers, routers, PLE) be stored
losslessly in fewer bits? Reads safetensors directly (no torch).

For each tensor family it reports how many values are bf16-exact (low 3
mantissa bits zero), the distinct exponent count, the exponent entropy,
and the size of three bitwise-lossless layouts:
  f16    : 16 bits (today)
  e4m10  : sign + 4-bit exponent code + 10-bit mantissa (needs <=16 exps)
  e4m7   : sign + 4-bit exponent code + 7-bit mantissa (needs bf16-exact)
  ent    : sign + exponent entropy + exact mantissa bits (an ideal bound)
"""
import json, struct, sys, math, collections
import numpy as np

CK = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # layers per family

idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
hdrs = {}


def header(fn):
    if fn not in hdrs:
        with open(f"{CK}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs[fn] = (8 + n, json.loads(f.read(n)))
    return hdrs[fn]


def load(name):
    fn = idx[name]
    base, h = header(fn)
    m = h[name]
    a, b = m["data_offsets"]
    assert m["dtype"] == "F16", (name, m["dtype"])
    with open(f"{CK}/{fn}", "rb") as f:
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=np.uint16), m["shape"]


FAMILIES = {
    "hc_down": "model.language_model.layers.{L}.{site}_hyper_connection.input_mix_weight_down.weight",
    "hc_up": "model.language_model.layers.{L}.{site}_hyper_connection.input_mix_weight_up.weight",
    "router": "model.language_model.layers.{L}.mlp.gate.weight",
    "gdn_a": "model.language_model.layers.{L}.linear_attn.in_proj_a.weight",
}

for fam, pat in FAMILIES.items():
    tot = collections.Counter()
    expc = collections.Counter()
    n = 0
    for L in [0, 1, 17, 46][:LIMIT]:
        for site in (["attn", "mlp"] if "{site}" in pat else [None]):
            name = pat.format(L=L, site=site)
            if name not in idx:
                continue
            u, shape = load(name)
            n += u.size
            e = (u >> 10) & 31
            man = u & 1023
            expc.update(dict(zip(*np.unique(e, return_counts=True))))
            tot["bf16_exact"] += int(np.count_nonzero((man & 7) == 0))
            tot["zero"] += int(np.count_nonzero((u & 0x7FFF) == 0))
            tot["sub"] += int(np.count_nonzero((e == 0) & (man != 0)))
            tot["low2"] += int(np.count_nonzero((man & 3) == 0))
            tot["low1"] += int(np.count_nonzero((man & 1) == 0))
    if n == 0:
        print(fam, "missing")
        continue
    p = np.array(list(expc.values()), dtype=np.float64) / n
    H = float(-(p * np.log2(p)).sum())
    exps = sorted(int(k) for k in expc)
    # exponent coverage of the top 15 exponents (code 15 = escape)
    top = sorted(expc.values(), reverse=True)
    cov15 = sum(top[:15]) / n
    cov16 = sum(top[:16]) / n
    print(f"{fam}: n={n} distinct_exp={len(exps)} range={exps[0]}..{exps[-1]} "
          f"H(exp)={H:.3f} bits top16={cov16:.6f} top15={cov15:.6f}")
    print(f"   bf16_exact={tot['bf16_exact']/n:.6f} low2={tot['low2']/n:.4f} "
          f"low1={tot['low1']/n:.4f} zero={tot['zero']/n:.6f} subnormal={tot['sub']/n:.6f}")
    mbits = 7 if tot["bf16_exact"] == n else 10
    print(f"   ideal bits/weight: {1 + H + mbits:.2f} (mantissa {mbits}); "
          f"e4 layout {1 + 4 + mbits} bits if <=16 exponents")
    print("   exp hist:", {k: round(v / n, 5) for k, v in sorted(expc.items())})

# Second pass: is every fp16 value exactly a bf16 value (fp32 low 16 bits
# zero)? Then sign + bf16 exponent code + 7-bit mantissa is lossless.
print("bf16 representability (all values, incl. subnormals):")
for fam, pat in FAMILIES.items():
    bad = n = 0
    bexp = collections.Counter()
    for L in [0, 1, 17, 46][:LIMIT]:
        for site in (["attn", "mlp"] if "{site}" in pat else [None]):
            name = pat.format(L=L, site=site)
            if name not in idx:
                continue
            u, _ = load(name)
            f = u.view(np.float16).astype(np.float32).view(np.uint32)
            n += u.size
            bad += int(np.count_nonzero(f & 0xFFFF))
            be = (f >> 23) & 255
            bexp.update(dict(zip(*np.unique(be[(f & 0x7FFFFFFF) != 0], return_counts=True))))
    ks = sorted(int(k) for k in bexp)
    top16 = sum(sorted(bexp.values(), reverse=True)[:16]) / n
    print(f"  {fam}: not_bf16={bad} of {n}; bf16 exps {len(ks)} ({ks[0]}..{ks[-1]}), "
          f"top16 mass {top16:.6f}")
