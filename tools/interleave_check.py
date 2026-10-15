#!/usr/bin/env python3
"""Compare CRUCIBLE_INTERLEAVE_CHECK files (engine/runtime/prefill_interleave.h) from two engine runs, key by key.

The check needs the engine CLI (crucible gen, a bench, or `crucible serve` driven directly on stdin). The Python
server removes the switch from the engine environment (RA-15), so a run through it writes no check file.

  interleave_check.py A.check B.check

A row key is (prompt hash, position): the logits of a row whose whole prefix is accepted depend only on the request.
A chunk key is (prefix hash, pos0, rows): the trunk stacks and final logits of a bulk prefill chunk depend only on its
prefix. Equal keys must give equal hashes, within a file and across files. Exit status 1 on any mismatch or when the
files share no key.
"""
import sys


def load(path):
    rows, chunks, conflicts = {}, {}, 0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "row" and len(parts) == 5:
                key, value = (parts[1], int(parts[2])), (int(parts[3]), parts[4])
                table = rows
            elif parts[0] == "chunk" and len(parts) == 6:
                key, value = (parts[1], int(parts[2]), int(parts[3])), (parts[4], parts[5])
                table = chunks
            else:
                raise ValueError(f"{path}: malformed line: {line.rstrip()}")
            if table.setdefault(key, value) != value:
                conflicts += 1
    return rows, chunks, conflicts


def compare(a, b):
    shared = a.keys() & b.keys()
    return len(shared), sum(a[k] != b[k] for k in shared)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    (ra, ca, xa), (rb, cb, xb) = load(argv[1]), load(argv[2])
    rows, row_bad = compare(ra, rb)
    chunks, chunk_bad = compare(ca, cb)
    print(f"rows: {len(ra)} and {len(rb)}, shared {rows}, mismatched {row_bad}")
    print(f"chunks: {len(ca)} and {len(cb)}, shared {chunks}, mismatched {chunk_bad}")
    print(f"conflicts within a file: {xa} and {xb}")
    ok = rows + chunks > 0 and row_bad == chunk_bad == xa == xb == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
