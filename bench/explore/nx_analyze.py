#!/usr/bin/env python3
"""Stage-4 numerics study: KL against the 8-row path and Gate 1(b) per variant.

Runs on the test machine in the explore-nx worktree (refenv python for numpy).
  nx_analyze.py kl TIER            KL(test||base), KL(base||test), top-1 for every variant and prompt
  nx_analyze.py gate TIER BASELINE Gate 1(b) from results/nx/gate-TIER/nxV/dump.oracle.txt
Variants (CRUCIBLE_NX bits): 0 current fast path, 1 HC no lo, 2 QSA no lo,
4 bf16 H, 8 bf16 pd, 15 all four; gate only: 16 HC dots no lo, 32 HC MidPack
no lo, 65 HC no lo except layer 0 (binary crucible-nx2); 128 fp16 pd, 146 = 16+2+128 (crucible-nx3).
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402

ROOT = Path("results/nx")
VARIANTS = (0, 1, 2, 4, 8, 15, 128, 146)
NAMES = {0: "current", 1: "HC no lo", 2: "QSA no lo", 4: "bf16 H", 8: "bf16 pd", 15: "all four",
         16: "HC dots", 32: "HC mid", 65: "HC no L0", 128: "fp16 pd", 146: "combo 146"}
GATE_VARIANTS = VARIANTS + (16, 32, 65)
# Existing 8-row bases from the prefill work (same command: gen --prefill 0 --n 1 --dump-logits).
OLD = work_paths.worktree("prefill") / "results"
SCAN = work_paths.worktree("prefill-scan") / "results"


def base_path(tier, prompt):
    own = ROOT / f"base{tier}-{prompt}.logits.f32"
    if own.exists():
        return own
    if tier == "305":
        legacy = {"code-4096": OLD / "base8-4096.logits.f32", "code-16384": OLD / "base8-16384.logits.f32",
                  "code-1024": SCAN / "code-base.logits.f32"}
    else:
        legacy = {"code-1024": OLD / "base8q-1024.logits.f32", "code-4096": OLD / "base8q-4096.logits.f32"}
    p = legacy.get(prompt)
    return p if p and p.exists() else None


def logp(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def kl(base, test):
    b, t = (np.fromfile(p, dtype=np.float32) for p in (base, test))
    lb, lt = logp(b), logp(t)
    return float(np.sum(np.exp(lt) * (lt - lb))), float(np.sum(np.exp(lb) * (lb - lt))), b.argmax() == t.argmax()


def cmd_kl(tier):
    prompts = sorted({p.name.split(".")[0] for v in VARIANTS for p in (ROOT / f"t{tier}-nx{v}").glob("*.logits.f32")},
                     key=lambda s: (s.split("-")[0], int(s.split("-")[1])))
    print(f"KL(test||base) against the 8-row path, {tier}; '*' marks a top-1 change")
    print("prompt        " + "".join(f"{NAMES[v]:>12s}" for v in VARIANTS))
    worst = {v: 0.0 for v in VARIANTS}
    vals = {v: [] for v in VARIANTS}
    for pr in prompts:
        base = base_path(tier, pr)
        cells = []
        for v in VARIANTS:
            test = ROOT / f"t{tier}-nx{v}" / f"{pr}.logits.f32"
            if base is None or not test.exists():
                cells.append(f"{'-':>12s}")
                continue
            f, r, same = kl(base, test)
            worst[v] = max(worst[v], f)
            vals[v].append(f)
            cells.append(f"{f:11.2e}{' ' if same else '*'}")
        print(f"{pr:13s} " + "".join(cells))
    print("max          " + "".join(f"{worst[v]:12.2e}" for v in VARIANTS))
    print("median       " + "".join(f"{np.median(vals[v]) if vals[v] else 0:12.2e}" for v in VARIANTS))
    print("n            " + "".join(f"{len(vals[v]):12d}" for v in VARIANTS))
    # The 8-row path has its own error at short prompts, so also
    # compare each variant with the current GEMM path (variant 0).
    print(f"\nKL(variant||current GEMM path), {tier}")
    print("prompt        " + "".join(f"{NAMES[v]:>12s}" for v in VARIANTS[1:]))
    worst = {v: 0.0 for v in VARIANTS[1:]}
    vals = {v: [] for v in VARIANTS[1:]}
    for pr in prompts:
        ref = ROOT / f"t{tier}-nx0" / f"{pr}.logits.f32"
        cells = []
        for v in VARIANTS[1:]:
            test = ROOT / f"t{tier}-nx{v}" / f"{pr}.logits.f32"
            if not ref.exists() or not test.exists():
                cells.append(f"{'-':>12s}")
                continue
            f, r, same = kl(ref, test)
            worst[v] = max(worst[v], f)
            vals[v].append(f)
            cells.append(f"{f:11.2e}{' ' if same else '*'}")
        print(f"{pr:13s} " + "".join(cells))
    print("max          " + "".join(f"{worst[v]:12.2e}" for v in VARIANTS[1:]))
    print("median       " + "".join(f"{np.median(vals[v]) if vals[v] else 0:12.2e}" for v in VARIANTS[1:]))


def cmd_gate(tier, baseline):
    pat = re.compile(r"\|\s*(\d+)\s*\|" + r"\s*([\d.eE+-]+)\s*\|" * 3)
    text = Path(baseline).read_text()
    quality = "- Quality checkpoint:" in text
    pat = re.compile(r"\|\s*(\d+)\s*\|" + r"\s*([\d.eE+-]+)\s*\|" * (6 if quality else 3))
    base = {}
    for line in text.splitlines():
        m = pat.fullmatch(line)
        if m:
            vals = m.groups()[4:] if quality else m.groups()[1:]
            for pos, val in zip((0, 14, 23), vals):
                base[int(m[1]), pos] = float(val)
    assert len(base) == 144, len(base)
    found = re.compile(r"layer (\d+) pos (\d+): relL2 vs fp64 (\S+)")
    print(f"Gate 1(b), {tier}: point <= min(2 x baseline, 1e-3) and mean <= 1.10 x baseline mean")
    for v in GATE_VARIANTS:
        log = ROOT / f"gate-{tier}" / f"nx{v}" / "dump.oracle.txt"
        if not log.exists():
            print(f"  {NAMES[v]:10s}: no oracle log")
            continue
        got = {(int(a), int(b)): float(c) for a, b, c in found.findall(log.read_text())}
        if set(got) != set(base):
            print(f"  {NAMES[v]:10s}: {len(got)} of 144 points")
            continue
        ratio = {k: got[k] / base[k] for k in base}
        worst = max(ratio, key=ratio.get)
        fail = [k for k in base if not math.isfinite(got[k]) or got[k] > min(2 * base[k], 1e-3)]
        mean = sum(got.values()) / 144 / (sum(base.values()) / 144)
        ok = not fail and mean <= 1.10
        print(f"  {NAMES[v]:10s}: {'PASS' if ok else 'FAIL'} mean ratio {mean:.4f}, max point ratio "
              f"{ratio[worst]:.3f} at {worst}, max error {max(got.values()):.3g}, failing points {len(fail)}")


if __name__ == "__main__":
    if sys.argv[1] == "kl":
        cmd_kl(sys.argv[2])
    else:
        cmd_gate(sys.argv[2], sys.argv[3])
