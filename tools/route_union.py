#!/usr/bin/env python3
"""Expert unions of a packed verify pass from captured serial routes (CRUCIBLE_ROUTE_DUMP files).

A token's routes do not depend on batching, so the union of B sessions x R consecutive tokens per layer is the
expert set that a packed pass of B spans with R rows each reads (accepted drafts are the next tokens).

  route_union.py FILE.routes [FILE.routes ...] [--max-b 4] [--max-r 8]
Prints mean distinct experts per layer U(B, R), averaged over session combinations and token windows.
"""
import argparse
import itertools
import statistics
import struct
from pathlib import Path

TOPK, LAYERS = 10, 48


def tokens(path):
    """Per decode token (R = 1 records), a list of 48 expert-ID sets."""
    data, out, current = Path(path).read_bytes(), [], {}
    pos = 0
    while pos < len(data):
        layer, rows = struct.unpack_from("<ii", data, pos)
        pos += 8
        ids = struct.unpack_from(f"<{rows * TOPK}i", data, pos)
        pos += 4 * rows * TOPK
        if rows != 1:
            continue
        current[layer] = set(ids)
        if layer == LAYERS - 1 and len(current) == LAYERS:
            out.append([current[l] for l in range(LAYERS)])
            current = {}
    return out


def union(streams, b, r, stride):
    values = []
    for combo in itertools.combinations(range(len(streams)), b):
        n = min(len(streams[i]) for i in combo)
        for t in range(0, n - r + 1, stride):
            for layer in range(LAYERS):
                experts = set()
                for i in combo:
                    for k in range(r):
                        experts |= streams[i][t + k][layer]
                values.append(len(experts))
    return statistics.mean(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-b", type=int, default=4)
    ap.add_argument("--max-r", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    streams = [tokens(f) for f in a.files]
    print("tokens per file:", [len(s) for s in streams])
    print("B \\ R " + " ".join(f"{r:>6}" for r in range(1, a.max_r + 1)))
    for b in range(1, a.max_b + 1):
        print(f"{b:>5} " + " ".join(f"{union(streams, b, r, a.stride):6.1f}" for r in range(1, a.max_r + 1)))


if __name__ == "__main__":
    main()
