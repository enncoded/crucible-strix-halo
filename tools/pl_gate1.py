"""Gate 1(b): compare oracle output lines with the fuse baseline table.
Usage: pl_gate1.py ORACLE_LOG [BASELINE]"""
import math
import re
import sys
from pathlib import Path

base_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("pl_oracle_baseline.txt")
base = {}
for line in open(base_path):
    layer, *vals = line.split()
    for pos, v in zip((0, 14, 23), vals):
        base[(int(layer), pos)] = float(v)
got = {}
duplicates = []
pat = re.compile(r"layer (\d+) pos (\d+): relL2 vs fp64 ([0-9.eE+-]+)")
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        key = (int(m[1]), int(m[2]))
        if key in got:
            duplicates.append(key)
        got[key] = float(m[3])
missing = sorted(set(base) - set(got))
ratios = {k: got[k] / base[k] for k in got if k in base and base[k] > 0 and math.isfinite(got[k])}
mean_ratio = ((sum(got.values()) / len(got)) / (sum(base.values()) / len(base))
              if got and base and sum(base.values()) > 0 else float("inf"))
worst = max(ratios, key=ratios.get) if ratios else None
worst_ratio = ratios[worst] if worst is not None else float("nan")
bad = [k for k, value in got.items() if k not in base or not math.isfinite(value) or value < 0
       or value > 2 * base[k] or value > 1e-3]
expected = {(layer, pos) for layer in range(48) for pos in (0, 14, 23)}
valid_baseline = set(base) == expected and all(math.isfinite(v) and v > 0 for v in base.values())
passed = valid_baseline and set(got) == set(base) and not duplicates and not bad and mean_ratio <= 1.10
print(f"points {len(got)}/144 missing {len(missing)} mean ratio {mean_ratio:.6f} "
      f"max point ratio {worst_ratio:.4f} at {worst} max error {max(got.values(), default=float('nan')):.4e} "
      f"failing {bad} duplicate {duplicates}")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
