#!/usr/bin/env python3
"""Parse resident sweep logs and calculate the verification weight roofline.

The roofline counts each static weight and each distinct routed expert once.
It excludes MTP, activation traffic, caches, and redundant reads. It is a lower
bound on verification time, not a hardware bandwidth measurement.
"""
import argparse
import json
from pathlib import Path
import re

BANDWIDTH = 245.5e9
WEIGHTS = {
    "3.05": (4_615_337_572, 1_862_412),
    "4.05": (5_355_894_372, 2_476_812),
}
SUMMARY = re.compile(
    r"^== (\S+): ([\d.]+) tok/s, (\d+) steps, ([\d.]+) tok/step, "
    r"accept ([\d.]+)%, hash ([0-9a-f]+), (\w+)$", re.M)
VERIFY = re.compile(r"verify\s+R=(\d+) calls\s+(\d+)\s+gpu\s+([\d.]+) ms/call")
EXPERTS = re.compile(r"R=(\d+) ([\d.]+)/call \((\d+) calls\)")


def weight_roofline(block, checkpoint):
    total, expert = WEIGHTS[checkpoint]
    static = total - 48 * 10 * expert
    match = re.search(r"^  experts trunk:(.*)$", block, re.M)
    if not match:
        return None
    counts = {int(r): float(u) for r, u, _ in EXPERTS.findall(match[1])}
    floor_ms, gpu_ms = 0.0, 0.0
    for rows, calls, ms in VERIFY.findall(block):
        if int(rows) not in counts:
            return None
        count = int(calls)
        floor_ms += count * (static + 48 * counts[int(rows)] * expert) / BANDWIDTH * 1000
        gpu_ms += count * float(ms)
    return 100 * floor_ms / gpu_ms if gpu_ms else None


def summarize(text, checkpoint):
    if checkpoint not in WEIGHTS:
        raise ValueError("checkpoint must be 3.05 or 4.05")
    matches = list(SUMMARY.finditer(text))
    if not matches:
        raise ValueError("log has no completed sweep cases")
    rows = []
    for i, match in enumerate(matches):
        name, speed, steps, tokens, acceptance, ids_hash, verdict = match.groups()
        if verdict == "MISMATCH":
            raise ValueError(f"{name}: MISMATCH with serial")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.end():end]
        ids = re.search(r"^ids " + re.escape(name) + r":(.*)$", block, re.M)
        step = re.search(r"steps \d+: wall ([\d.]+) ms/step", block)
        head = re.search(r"head policy: restricted (\d+) steps, full (\d+) steps, final (\w+)", block)
        timings = re.search(r"^timings " + re.escape(name) +
                            r": prefill ([\d.]+) ms, decode ([\d.]+) ms, request ([\d.]+) ms$",
                            block, re.M)
        evicted = re.search(r"^counters " + re.escape(name) + r" .*evicted_ms=(\d+)", block, re.M)
        policy = ({"restricted_steps": int(head[1]), "full_steps": int(head[2]), "final": head[3]}
                  if head else None)
        rows.append({"name": name, "checkpoint": checkpoint, "tok_s": float(speed),
                     "steps": int(steps), "tok_step": float(tokens),
                     "acceptance_percent": float(acceptance), "hash": ids_hash,
                     "verdict": verdict, "ids": [int(x) for x in ids[1].split()] if ids else None,
                     "step_ms": float(step[1]) if step else None, "head_policy": policy,
                     "prefill_ms": float(timings[1]) if timings else None,
                     "decode_ms": float(timings[2]) if timings else None,
                     "request_ms": float(timings[3]) if timings else None,
                     "evicted_ms": int(evicted[1]) if evicted else None,
                     "verify_weight_roof_percent": weight_roofline(block, checkpoint)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=WEIGHTS)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [dict(row, source=str(path)) for path in args.logs
            for row in summarize(path.read_text(), args.checkpoint)]
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        roof = row["verify_weight_roof_percent"]
        pct = "unmeasured" if roof is None else f"{roof:.1f}%"
        print(f'{row["name"]}: {row["tok_s"]:.2f} tok/s, {row["tok_step"]:.3f} tok/step, '
              f'verify {pct} of weight roofline, {row["verdict"]}')


if __name__ == "__main__":
    main()
