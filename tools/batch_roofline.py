#!/usr/bin/env python3
"""Print decode and depth-two speculative ceilings from a measured union curve."""
import argparse
import json

BANDWIDTH = 245.5e9
STATIC_BYTES = 3_721_379_812
EXPERT_BYTES = 1_862_412
LAYERS = 48
GDN_BYTES = 226_492_416
KV_BYTES = 50_405_376


def rates(count, union, window_union, context=32768):
    indexer = 12 * (context // 4) * 512
    state = GDN_BYTES + KV_BYTES + indexer
    decode = STATIC_BYTES + LAYERS * EXPERT_BYTES * union + count * state
    verify = (STATIC_BYTES + LAYERS * EXPERT_BYTES * window_union
              + count * (GDN_BYTES + 3 * (KV_BYTES + indexer)))
    mtp = (83_675_688 + 397_813_764 * 32768 / 248320 + EXPERT_BYTES * union
           + count * (4_196_352 + (context // 4) * 512))
    return decode, count * BANDWIDTH / decode, count * 2.84 * BANDWIDTH / (verify + 2 * mtp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("union_json")
    args = parser.parse_args()
    with open(args.union_json) as file:
        curve = json.load(file)["curve"]
    print("| B | Measured union | Decode GB/step | Decode ceiling | Decode expected | Spec ceiling | Spec expected |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in curve:
        count, union = row["B"], row["mean"]
        # This is an extrapolation, not a measured independent verification-window union.
        window_union = min(512, union * 22.4 / 10)
        traffic, decode, spec = rates(count, union, window_union)
        def pair(value):
            return f"{value:.1f} / {value / count:.1f}"
        print(f"| {count} | {union:.2f} | {traffic/1e9:.3f} | {pair(decode)} | "
              f"{pair(0.65*decode)} | {pair(spec)} | {pair(0.65*spec)} |")


if __name__ == "__main__":
    main()
