"""Summarize ABAB leases (abab.sh): per-pair group ratios, their spread, and the GPU clock of each run.

  python3 training/mtp_tune/abab_report.py PREFIX [PREFIX ...]

PREFIX.<i>.<arm>.log are spec_sweep logs, PREFIX.<i>.<arm>.sclk clock samples in Hz.
"""
import glob
import re
import statistics
import sys
from collections import defaultdict

from ab_report import group, parse  # noqa: E402  (run from this directory or with it on sys.path)


def sclk(path):
    v = [int(x) / 1e6 for x in open(path).read().split() if x.isdigit()]
    return (statistics.median(v), min(v)) if v else (None, None)


def main():
    pairs = []
    for prefix in sys.argv[1:]:
        for f in sorted(glob.glob(prefix + ".*.stock.log"), key=lambda p: int(re.findall(r"\.(\d+)\.stock", p)[0])):
            i = re.findall(r"\.(\d+)\.stock", f)[0]
            t = f.replace(".stock.log", ".tuned.log")
            pairs.append((f"{prefix.rsplit('/', 1)[-1]}.{i}", parse(f), parse(t), sclk(f[:-4] + ".sclk"),
                          sclk(t[:-4] + ".sclk")))
    per_group = defaultdict(list)
    hashes_ok = True
    first = {}
    for name, s, t, cs, ct in pairs:
        ratios = defaultdict(list)
        for k in s:
            if k in t:
                ratios[group(k[0])].append(t[k]["tps"] / s[k]["tps"])
                for arm in (s, t):
                    hashes_ok &= first.setdefault(k[0], arm[k]["hash"]) == arm[k]["hash"]
        allr = [r for v in ratios.values() for r in v]
        means = {g: sum(v) / len(v) for g, v in ratios.items()}
        for g, x in means.items():
            per_group[g].append(x)
        per_group["mean"].append(sum(allr) / len(allr))
        print(f"{name}: " + " ".join(f"{g} {x - 1:+.1%}" for g, x in sorted(means.items())) +
              f" mean {sum(allr) / len(allr) - 1:+.1%}  sclk median/min MHz stock {cs[0]:.0f}/{cs[1]:.0f}"
              f" tuned {ct[0]:.0f}/{ct[1]:.0f}")
    print(f"over {len(pairs)} pairs (mean, min, max of the per-pair ratio):")
    for g, v in sorted(per_group.items()):
        print(f"  {g:6} {statistics.mean(v) - 1:+.1%}  [{min(v) - 1:+.1%}, {max(v) - 1:+.1%}]")
    print("same output hash per prompt in every run:", hashes_ok)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main()
