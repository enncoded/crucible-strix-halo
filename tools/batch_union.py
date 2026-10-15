#!/usr/bin/env python3
"""Measure cross-stream unions over every prompt subset, at matched decode steps."""
import argparse
import array
import itertools
import json
import struct


def union_curve(routes, shape):
    streams, steps, layers, topk = shape
    if len(routes) != streams * steps * layers * topk:
        raise ValueError("routing payload length does not match the header")
    sets = []
    for stream in range(streams):
        points = []
        for point in range(steps * layers):
            start = (stream * steps * layers + point) * topk
            row = routes[start:start + topk]
            if len(set(row)) != topk or any(e < 0 or e >= 512 for e in row):
                raise ValueError("routes must contain ten unique expert IDs in [0, 512)")
            points.append(set(row))
        sets.append(points)
    curve = []
    for count in range(1, streams + 1):
        sums = []
        for subset in itertools.combinations(range(streams), count):
            total = sum(len(set().union(*(sets[s][p] for s in subset)))
                        for p in range(steps * layers))
            sums.append(total / (steps * layers))
        curve.append({"B": count, "mean": sum(sums) / len(sums),
                      "min_subset_mean": min(sums), "max_subset_mean": max(sums),
                      "subsets": len(sums), "points_per_subset": steps * layers})
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routes")
    args = parser.parse_args()
    with open(args.routes, "rb") as file:
        shape = struct.unpack("<4i", file.read(16))
        payload = file.read()
    if shape[0] != 8 or not 1 <= shape[1] <= 1024 or shape[2:] != (48, 10):
        raise ValueError("unsupported routing geometry")
    if len(payload) % 4:
        raise ValueError("routing payload has an incomplete integer")
    routes = array.array("i")
    routes.frombytes(payload)
    import sys
    if sys.byteorder != "little":
        routes.byteswap()
    print(json.dumps({"shape": shape, "curve": union_curve(routes, shape)}, indent=2))


if __name__ == "__main__":
    main()
