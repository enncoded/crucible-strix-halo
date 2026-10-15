#!/usr/bin/env python3
"""Long-context cost per token against a ceiling that counts KV and indexer bytes.

Decode ceiling per token at context t (bytes / 245.5 GB/s):
  weights 4.62 GB + GDN state read and write 2 x 113 MB
  + QSA K/V of the selected tokens: min(t, 2051) x 2048 B x 12 layers
  + indexer pool scan: t/4 blocks x 128 fp32 (512 B) x 12 layers
Prefill per-token floor at position p, on top of the context-free part:
  exact indexer scores (fixed fp32 FMA order, so VALU): 4 heads x 128 x p/4
  FMAs per row and layer, 12 layers, at 6.6 T FMA/s; plus the score array
  written and read once (p/4 x 4 B x 2 x 12) unless top-k is fused.
Measured inputs:
  decode: longctx.md final table (kernels 544e75b/e5561b4, in main; measured
  2026-10-02 on an earlier base, before the GTT policy and startup settling).
  prefill QSA parts: results/qsa-long-prefill.txt (main 093fc4c
  bench/prefill_qsa, 2048 rows, random data).
Usage: longctx_model.py [results/qsa-long-prefill.txt]
"""
import re
import sys

DRAM = 245.5e9
FMA = 6.6e12
WEIGHTS, GDN_STATE = 4.62e9, 2 * 113246208
LAYERS = 12
DECODE = {8192: 29.491, 32768: 29.818, 131072: 30.817, 262080: 31.902}  # median ms/token
CORE_US = {8192: 60.925, 32768: 81.043, 131072: 167.521, 262144: 282.776}  # QSA core us/layer
CONTEXTS = (16384, 65536, 131072, 262144)


def decode_ceiling_ms(t, idx_bytes=512):
    kv = min(t, 2051) * 2048 * LAYERS
    idx = t // 4 * idx_bytes * LAYERS
    return (WEIGHTS + GDN_STATE + kv + idx) / DRAM * 1e3, kv, idx


def interp(table, t):
    ks = sorted(table)
    for a, b in zip(ks, ks[1:]):
        if a <= t <= b:
            return table[a] + (table[b] - table[a]) * (t - a) / (b - a)
    return table[ks[-1]] if t > ks[-1] else table[ks[0]]


def parse_bench(path):
    out = {}
    try:
        text = open(path).read()
    except OSError:
        return out
    for m in re.finditer(r"qsa rows=(\d+) past=(\d+) prep_us=([\d.]+) pool_us=([\d.]+) score_us=([\d.]+) "
                         r"select_us=([\d.]+) flash_us=([\d.]+)", text):
        rows, past = int(m[1]), int(m[2])
        out[past + rows] = dict(zip(("prep", "pool", "score", "select", "flash"), map(float, m.groups()[2:])))
    for m in re.finditer(r"union rows=(\d+) past=(\d+) union_us=([\d.]+)", text):
        out.setdefault(int(m[2]) + int(m[1]), {})["union"] = float(m[3])
    return out


def main(path):
    print("decode, ms per token (measured median from longctx.md; 16K and 64K interpolated)")
    print(" context  ceiling  (KV MB, idx MB)  measured  % of ceiling  QSA core ms  core floor ms")
    for t in CONTEXTS:
        ceil, kv, idx = decode_ceiling_ms(t)
        meas = interp(DECODE, t)
        core = interp(CORE_US, t) * LAYERS / 1e3
        core_floor = (kv + idx) / DRAM * 1e3
        print(f" {t:7d}  {ceil:7.2f}  ({kv / 1e6:5.1f}, {idx / 1e6:6.1f})  {meas:8.2f}  {ceil / meas * 100:10.1f}%"
              f"  {core:11.2f}  {core_floor:12.2f}")
    print(" bf16 indexer pool would save:",
          ", ".join(f"{t // 1024}K {(decode_ceiling_ms(t)[0] - decode_ceiling_ms(t, 256)[0]):.2f} ms"
                    for t in CONTEXTS))

    bench = parse_bench(path)
    print("\nprefill QSA per 2048-row part per layer (us), random data; engine runs 2 parts x 12 layers per 4K")
    for ctx, d in sorted(bench.items()):
        print(f"  ctx {ctx:6d}: " + " ".join(f"{k} {v:.0f}" for k, v in d.items()))
    if not bench:
        return
    # Score floor: exact fp32 FMA order, so VALU; 4 heads x 128 dims per pooled block.
    print("\nscore kernel against its VALU floor")
    for ctx, d in sorted(bench.items()):
        fma = 2048 * (ctx // 4) * 4 * 128
        print(f"  ctx {ctx:6d}: {d['score'] / 1e3:6.2f} ms, floor {fma / FMA * 1e3:5.2f} ms "
              f"({fma / FMA * 1e6 / d['score'] * 100:4.1f}% of VALU)")
    # Marginal cost of one 4K chunk at position p (model):
    #   F (context-free part) = main 4K chunk 3.061 s minus its QSA (about 0.27 s)
    #   QSA = 24 parts x (prep + score + select + flash_real)
    #   flash_real = 0.72 x bench union flash: the engine's 16K average (15.8 ms per
    #   part) over the bench's 0-16K average (about 22 ms, random selections).
    F, REAL = 2.79, 0.72
    pts = sorted(bench)
    ext = dict(bench)
    last, prev = pts[-1], pts[-2]
    ratio = 262144 / last
    ext[262144] = {k: bench[last][k] * (ratio if k in ("score", "select") else 1.0)
                   + (0 if k in ("score", "select") else (bench[last][k] - bench[prev][k]) * (262144 - last) / (last - prev))
                   for k in bench[last]}

    def part_ms(p):
        ks = sorted(ext)
        if p <= ks[0]:
            d = {k: v * p / ks[0] if k in ("score", "select") else v for k, v in ext[ks[0]].items()}
        else:
            a = max(k for k in ks if k <= p)
            b = min(k for k in ks if k >= p)
            w = 0 if a == b else (p - a) / (b - a)
            d = {k: ext[a][k] + (ext[b][k] - ext[a][k]) * w for k in ext[a]}
        return (d["prep"] + d["score"] + d["select"] + REAL * d.get("union", d["flash"])) / 1e3

    print("\nprefill model, 3.05 (4K chunks): marginal chunk at p, and time to prefill T tokens from empty")
    print("        p   QSA s/chunk  chunk s  marginal tok/s |      T  total s  avg tok/s")
    for t in CONTEXTS:
        q = part_ms(t) * 24 / 1e3
        total = sum(F + part_ms(c + 4096) * 24 / 1e3 for c in range(0, t, 4096))
        print(f"  {t:7d}  {q:10.2f}  {F + q:7.2f}  {4096 / (F + q):12.0f} | {t:6d}  {total:7.1f}  {t / total:9.0f}")
    print("  (16K check: main measured 1,293 tok/s average)")




if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qsa-long-prefill.txt")
