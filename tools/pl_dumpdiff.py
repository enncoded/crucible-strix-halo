"""Per-layer relative L2 between two dump dirs (h_00..h_48), per position.

Usage: pl_dumpdiff.py DIR_A DIR_B [TOP]
Prints, per layer, the mean and max relL2 over positions and the worst
position, so a routing flip or a divergent layer type stands out.
"""
import sys
from pathlib import Path

import numpy as np

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for layer in range(49):
    name = f"h_{layer:02d}.f32"
    a = np.fromfile(a_dir / name, np.float32).reshape(-1, 10240).astype(np.float64)
    b = np.fromfile(b_dir / name, np.float32).reshape(-1, 10240).astype(np.float64)
    rel = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30)
    worst = np.argsort(rel)[::-1][:top]
    kind = "attn" if layer > 0 and (layer - 1) % 4 == 3 else ("embed" if layer == 0 else "gdn")
    print(f"h_{layer:02d} ({kind:5s}) mean {rel.mean():.2e} max {rel.max():.2e} at "
          + ", ".join(f"{p}:{rel[p]:.1e}" for p in worst))
