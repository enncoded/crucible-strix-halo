#!/usr/bin/env python3
"""Fail closed when dump files or resident sweep results are incomplete."""
import argparse
import filecmp
from pathlib import Path
import re


def check_dumps(baseline, one, eight):
    names = {f"h_{layer:02d}.f32" for layer in range(49)}
    for directory in (baseline, one, eight):
        if {path.name for path in directory.glob("h_*.f32")} != names:
            raise ValueError(f"{directory}: expected exactly 49 named dump files")
        for name in names:
            size = (directory / name).stat().st_size
            if not size or size % (4 * 2560 * 4):
                raise ValueError(f"{directory / name}: empty or incomplete residual rows")
    for name in sorted(names):
        if not filecmp.cmp(baseline / name, one / name, shallow=False):
            raise ValueError(f"Gate 1(a): {name} differs from baseline")
        if not filecmp.cmp(one / name, eight / name, shallow=False):
            raise ValueError(f"Gate 2: {name} differs across row counts")


def check_sweep(text):
    expected = {"s0": "ref", "s2": "exact", "s4": "exact", "copy0": "ref", "copyl7": "exact"}
    rows = re.findall(r"^== ([a-z0-9]+): .*?, (ref|exact|noref|MISMATCH)$", text, re.MULTILINE)
    summaries = [line for line in text.splitlines() if line.startswith("==")]
    if len(summaries) != len(expected) or len(rows) != len(expected) or {name for name, _ in rows} != expected.keys():
        raise ValueError("Gate 3: missing, duplicate, or unexpected sweep results")
    for name, verdict in rows:
        if verdict != expected[name]:
            raise ValueError(f"Gate 3: bad verdict for {name}: {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("one", type=Path)
    parser.add_argument("eight", type=Path)
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    try:
        check_dumps(args.baseline, args.one, args.eight)
        check_sweep(args.sweep.read_text())
    except (ValueError, OSError) as error:
        parser.exit(1, f"gates: {error}\n")
    print("Gate 1(a): 49/49. Gate 2: 49/49. Gate 3: all expected outputs exact.")


if __name__ == "__main__":
    main()
