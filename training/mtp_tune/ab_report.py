"""Summarize spec_sweep A/B logs (measure.sh): tok/s and tokens per step by prompt, mode and group.

  python3 training/mtp_tune/ab_report.py PREFIX [PREFIX ...]

Reads PREFIX.stock.log and PREFIX.tuned.log. Prints per-prompt rows, group means of the tok/s ratio and
an exactness check (every speculative hash equals its serial reference in both arms, and the serial
hashes agree between arms).
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^== (\S+)\.(\S+): ([\d.]+) tok/s, (\d+) steps, ([\d.]+) tok/step, accept ([\d.]+)%, "
                  r"hash (\w+), (\S+)")


def parse(path):
    out = {}
    for line in open(path):
        m = LINE.match(line)
        if m:
            name, mode, tps, steps, tstep, acc, h, verdict = m.groups()
            out[(name, mode)] = dict(tps=float(tps), tstep=float(tstep), acc=float(acc), hash=h, verdict=verdict)
    return out


def group(name):
    return {"chat": "prose"}.get(name.split("_")[0], name.split("_")[0])


def main():
    stock, tuned = {}, {}
    for p in sys.argv[1:]:
        stock.update(parse(p + ".stock.log"))
        tuned.update(parse(p + ".tuned.log"))
    exact = all(r["verdict"] in ("ref", "exact", "exact-top1", "sampled", "noref")
                for r in list(stock.values()) + list(tuned.values()))
    exact &= all(stock[k]["hash"] == tuned[k]["hash"] for k in stock if k in tuned)
    # Sampled lines carry no verdict: compare each speculative hash with its arm's serial hash.
    for arm in (stock, tuned):
        for (name, mode), r in arm.items():
            ref = arm.get((name, "s" + "serial")) if mode.startswith("s") else arm.get((name, "serial"))
            exact &= ref is None or ref["hash"] == r["hash"]
    ratios = defaultdict(list)
    print(f"{'prompt':16} {'mode':9} {'stock tok/s':>11} {'tuned':>7} {'ratio':>6} {'tok/step':>15}")
    for k in sorted(stock):
        if k not in tuned:
            continue
        s, t = stock[k], tuned[k]
        r = t["tps"] / s["tps"]
        ratios[(group(k[0]), k[1])].append(r)
        print(f"{k[0]:16} {k[1]:9} {s['tps']:11.2f} {t['tps']:7.2f} {r:6.3f} {s['tstep']:6.3f} -> {t['tstep']:5.3f}")
    print("group means (tuned / stock tok/s):")
    for mode in sorted({m for _, m in ratios}):
        means = {g: sum(v) / len(v) for (g, m), v in ratios.items() if m == mode}
        allr = [r for (g, m), v in ratios.items() if m == mode for r in v]
        print(f"  {mode:9} " + " ".join(f"{g} {x - 1:+.1%}" for g, x in sorted(means.items())) +
              f"  mean {sum(allr) / len(allr) - 1:+.1%}")
    print("exact:", exact)


if __name__ == "__main__":
    main()
