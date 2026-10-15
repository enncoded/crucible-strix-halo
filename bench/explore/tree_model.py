#!/usr/bin/env python3
"""Tree-verification and draft-quality estimates from a spec_sweep log.

Input: the top-2 probe sweep (scratch copy of draft2 d39f190 that records
the MTP runner-up). For each run NAME.q3 (default policy, spec 7, Q0.3) and
NAME.deep (always 7 drafts) the log has:
  top2 probe: ... runner-up hits by margin bucket (printed before '== NAME')
  accept rate by margin bucket per draft position, depth histograms,
  verify and MTP gpu ms per call.
Usage: tree_model.py results/top2-305.txt
"""
import re
import sys
from collections import defaultdict

B = 8  # margin buckets <1 <2 <3 <4 <6 <8 <12 >=12
CATS = {"code": ("code_fibonacci", "code_json"), "copy": ("copy_python",),
        "chat": ("chat_planning", "chat_explain", "chat_spanish"),
        "prose": ("prose_story", "prose_explain")}


def parse(path):
    runs, pending, cur = {}, None, None
    for line in open(path):
        if line.startswith("top2 probe by margin"):
            pending["hit"] = [tuple(map(int, x.split("/"))) for x in line.split(":")[1].split()]
        elif line.startswith("top2 probe:"):
            m = re.search(r"(\d+) MTP steps, (\d+) rejected, runner-up hits (\d+) \(first position (\d+)/(\d+)", line)
            pending = dict(zip(("steps", "rej", "hits", "f_hit", "f_n"), map(int, m.groups())))
        elif line.startswith("== "):
            m = re.match(r"== (\S+): ([\d.]+) tok/s, (\d+) steps, ([\d.]+) tok/step", line)
            cur = {"tps": float(m[2]), "steps": int(m[3]), "tpstep": float(m[4]),
                   "verify": {}, "mtp_first": [], "mtp_chain": [], "acc": {}, "hist": {}}
            if pending and not m[1].endswith("serial"):
                cur["probe"] = pending
            pending = None
            runs[m[1]] = cur
        elif cur is None:
            continue
        elif re.match(r"\s+verify\s+R=", line):
            m = re.search(r"R=(\d+) calls\s+(\d+)\s+gpu\s+([\d.]+)", line)
            cur["verify"][int(m[1])] = (int(m[2]), float(m[3]))
        elif re.match(r"\s+mtp_(first|chain)", line):
            m = re.search(r"(mtp_\w+)\s+R=(\d+) calls\s+(\d+)\s+gpu\s+([\d.]+)", line)
            cur[m[1]].append((int(m[3]), float(m[4])))
        elif re.match(r"\s+draft \d+:", line):
            k = int(line.split()[1][:-1])
            cur["acc"][k] = [(float(a), int(n)) for a, n in re.findall(r"([\d.]+)\((\d+)\)", line)]
        elif "accepted histogram" in line:
            d = int(re.search(r"depth (\d+):", line)[1])
            cur["hist"][d] = list(map(int, line.split("histogram")[1].split()))
    return runs


def pooled_verify(runs):
    tot = defaultdict(lambda: [0, 0.0])
    for r in runs.values():
        for R, (n, ms) in r["verify"].items():
            tot[R][0] += n
            tot[R][1] += n * ms
    return {R: s / n for R, (n, s) in sorted(tot.items()) if n}


def main(path):
    runs = parse(path)
    V = pooled_verify(runs)
    chain = sum(n * ms for r in runs.values() for n, ms in r["mtp_chain"]) / max(
        1, sum(n for r in runs.values() for n, _ in r["mtp_chain"]))
    print("verify ms by rows:", " ".join(f"R{R}={ms:.1f}" for R, ms in V.items()), f"| mtp chain {chain:.2f} ms")

    # Runner-up coverage at the first rejected draft, by margin bucket.
    for pol in ("q3", "deep"):
        hit, n = [0] * B, [0] * B
        fh = fn = 0
        for name, r in runs.items():
            if name.endswith("." + pol) and "probe" in r:
                for b, (h, k) in enumerate(r["probe"]["hit"]):
                    hit[b] += h
                    n[b] += k
                fh += r["probe"]["f_hit"]
                fn += r["probe"]["f_n"]
        cov = " ".join(f"{h}/{k}={h / k:.2f}" if k else "-" for h, k in zip(hit, n))
        print(f"{pol}: runner-up right when top-1 rejected, by margin: {cov}; all {sum(hit)}/{sum(n)}"
              f"={sum(hit) / sum(n):.2f}; first position {fh}/{fn}={fh / fn:.2f}")
        if pol == "q3":
            c = [h / k if k else 0 for h, k in zip(hit, n)]

    # Sibling at draft 1 when its margin is in buckets < T (always reached).
    print("\nsibling row at draft 1 for low margins (q3 runs; c from the pooled q3 coverage)")
    for drow in (3.4, 2.0):
        for T in (1, 2, 3):
            out = []
            gain_all = [0.0, 0.0]
            for cat, names in CATS.items():
                tok = ms = 0.0
                extra_tok = extra_rows = 0.0
                for nm in names:
                    r = runs[nm + ".q3"]
                    tok += r["steps"] * r["tpstep"]
                    ms += r["steps"] * r["tpstep"] / r["tps"] * 1e3
                    for b, (p, k) in enumerate(r["acc"].get(1, [])[:T]):
                        extra_tok += k * (1 - p) * c[b]
                        extra_rows += k
                base = tok / ms * 1e3
                new = (tok + extra_tok) / (ms + extra_rows * drow) * 1e3
                out.append(f"{cat} {base:.1f}->{new:.1f} ({(new / base - 1) * 100:+.1f}%)")
            print(f"  row {drow} ms, margin < bucket {T}: " + ", ".join(out))

    # Oracle upper bound: one sibling exactly at the rejected draft.
    print("\noracle sibling at the rejected position (upper bound), row 3.4 ms:")
    out = []
    for cat, names in CATS.items():
        tok = ms = et = er = 0.0
        for nm in names:
            r = runs[nm + ".q3"]
            tok += r["steps"] * r["tpstep"]
            ms += r["steps"] * r["tpstep"] / r["tps"] * 1e3
            et += r["probe"]["hits"]
            er += r["probe"]["rej"]
        out.append(f"{cat} {(tok + et) / (ms + er * 3.4) / (tok / ms) * 100 - 100:+.1f}%")
    print("  " + ", ".join(out))

    # Per-depth conditional acceptance of the MTP chain (deep runs).
    print("\nMTP chain: P(draft k accepted | drafts 1..k-1 accepted), deep runs")
    accs = {}
    for cat, names in CATS.items():
        h = [0] * 8
        for nm in names:
            for i, v in enumerate(runs[nm + ".deep"]["hist"].get(7, [])):
                h[i] += v
        tot = sum(h)
        reach = [sum(h[k:]) for k in range(8)]
        a = [reach[k] / reach[k - 1] if reach[k - 1] else 0 for k in range(1, 8)]
        accs[cat] = a
        print(f"  {cat:5s} ({tot} steps): " + " ".join(f"{x:.2f}" for x in a))

    # Fixed-depth chain model and a better-draft scenario.
    # Probe 2 v4 (megakernel dataflow, 8 compute waves): R=1 20.3, R=4 26.8,
    # R=8 36.1 ms; linear in between.
    mega = {R: 20.3 + (R - 1) * (26.8 - 20.3) / 3 if R <= 4 else 26.8 + (R - 4) * (36.1 - 26.8) / 4
            for R in range(1, 9)}

    def best(a, costs=None):
        costs = costs or V
        res = []
        for d in range(1, 8):
            e = 1 + sum(_prod(a[:k]) for k in range(1, d + 1))
            t = costs[d + 1] + chain * d
            res.append((e / t * 1e3, d, e))
        return max(res)

    print("\nbest fixed-depth chain (tok/s, depth, tok/step); better draft adds delta to depths >= 2")
    for cat, a in accs.items():
        line = [f"{cat:5s} now {best(a)[0]:.1f}"]
        for delta in (0.05, 0.10, 0.15):
            b = [a[0]] + [min(0.97, x + delta) for x in a[1:]]
            line.append(f"+{delta:.2f}: {best(b)[0]:.1f}")
        b = [min(0.97, x + 0.10) for x in a]
        line.append(f"all +0.10: {best(b)[0]:.1f}")
        b2 = [a[0]] + [min(0.97, x + 0.10) for x in a[1:]]
        m0, m1 = best(a, mega), best(b2, mega)
        line.append(f"| megakernel rows: now {m0[0]:.1f} (depth {m0[1]}), +0.10: {m1[0]:.1f}")
        print("  " + ", ".join(line))


def _prod(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/top2-305.txt")
