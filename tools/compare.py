#!/usr/bin/env python3
"""Per-layer comparison of residual-stack dumps.

Usage: compare.py OURS_DIR REF_DIR
Both dirs hold h_XX.f32 files, [positions][4][2560] float32 (XX = 00 for the
embedding, 01..48 after each layer). Prints relative L2 error and cosine
similarity per layer, worst position first.
"""
import os
import sys

import numpy as np

D = 4 * 2560


def load(d, i):
    p = os.path.join(d, f"h_{i:02d}.f32")
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.float32)
    return a.reshape(-1, D)


def main():
    ours, ref = sys.argv[1], sys.argv[2]
    print(f"{'layer':>5} {'pos':>4} {'relL2':>10} {'worst':>10} {'@pos':>5} {'cos_min':>9} {'maxabs':>10}")
    for i in range(49):
        a, b = load(ours, i), load(ref, i)
        if a is None or b is None:
            break
        n = min(len(a), len(b))
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        num = np.linalg.norm(a - b, axis=1)
        den = np.linalg.norm(b, axis=1) + 1e-30
        rel = num / den
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
        tot = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
        w = int(rel.argmax())
        print(f"{i:5d} {n:4d} {tot:10.3e} {rel[w]:10.3e} {w:5d} {cos.min():9.6f} {np.abs(a - b).max():10.3e}")


if __name__ == "__main__":
    main()
