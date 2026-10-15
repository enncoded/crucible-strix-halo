"""Flip-like events between two dump dirs: (layer, positions) where the
relative difference jumps by 8x to above 3e-3 within one layer (MoE top-k
flips). Usage: pl_flips.py DIR_A DIR_B"""
import sys
from collections import defaultdict
import numpy as np
a_dir, b_dir = sys.argv[1], sys.argv[2]
def load(d, i):
    return np.fromfile(f"{d}/h_{i:02d}.f32", np.float32).reshape(-1, 10240).astype(np.float64)
prev, ev = None, defaultdict(list)
for L in range(49):
    a, b = load(a_dir, L), load(b_dir, L)
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    if prev is not None:
        for p in np.where((rel > 3e-3) & (rel > 8 * np.maximum(prev, 1e-5)))[0]:
            ev[L - 1].append(int(p))
    prev = rel
print(" ".join(f"{L}:" + ",".join(map(str, v)) for L, v in sorted(ev.items())))
print("events", sum(len(v) for v in ev.values()), "final mean rel", f"{rel.mean():.2e}")
