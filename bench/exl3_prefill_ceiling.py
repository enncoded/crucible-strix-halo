"""Calibrated EXL3 prefill floors. These estimates are not counter measurements."""
import argparse
import json
import math
import re
from pathlib import Path

ROWS = 4096
WMMA_TFLOPS = 49
GEMM_TFLOP = 43.33
CURRENT_WITHOUT_DECODE_MS = 1593
MOE_DECODE_305_MS = 105
DENSE_DECODE_305_MS = 75
SHARED_ROW_TILES = math.ceil(ROWS / 96)


def parse_cycles(text):
    matches = re.findall(r"decode K([3456]) LDS\s+\S+ ms\s+\S+ ns/SIMD-step\s+(\S+) cycles/SIMD-step", text)
    if len(matches) != 4 or len({k for k, _ in matches}) != 4:
        raise ValueError("need exactly one LDS cost for each K3/K4/K5/K6")
    return {int(k): float(value) for k, value in matches}


def estimate(weights, costs):
    if set(costs) != {3, 4, 5, 6} or any(not math.isfinite(c) or c <= 0 for c in costs.values()):
        raise ValueError("decoder costs must be finite and positive for K3 through K6")
    if set(weights) != {"305", "405"} or set(weights["305"]) != set(weights["405"]):
        raise ValueError("checkpoint matrix keys differ")
    totals = {pack: {"moe": 0., "dense": 0.} for pack in weights}
    for name, base in weights["305"].items():
        test = weights["405"][name]
        if (base["tiles"] != test["tiles"] or base["category"] != test["category"]
                or not isinstance(base["tiles"], int) or base["tiles"] <= 0):
            raise ValueError("checkpoint matrix shape or category differs")
        category = base["category"]
        if category not in {"routed", "shared", "dense"}:
            raise ValueError("unsupported matrix category")
        for pack in weights:
            row = weights[pack][name]
            if row["K"] not in costs:
                raise ValueError("unsupported matrix K")
            group = "dense" if category == "dense" else "moe"
            copies = SHARED_ROW_TILES if category == "shared" else 1
            totals[pack][group] += row["tiles"] * copies * costs[row["K"]]
    if any(not math.isfinite(value) or value <= 0 for groups in totals.values() for value in groups.values()):
        raise ValueError("need finite positive MoE and dense costs")
    gemm_ms = GEMM_TFLOP / WMMA_TFLOPS * 1000
    result = {}
    for pack in ("305", "405"):
        moe_ms = MOE_DECODE_305_MS * totals[pack]["moe"] / totals["305"]["moe"]
        dense_ms = DENSE_DECODE_305_MS * totals[pack]["dense"] / totals["305"]["dense"]
        total = CURRENT_WITHOUT_DECODE_MS + moe_ms + dense_ms
        result[pack] = {"gemm_ms": gemm_ms,
                        "non_gemm_ms": CURRENT_WITHOUT_DECODE_MS - gemm_ms,
                        "moe_decode_ms": moe_ms, "dense_decode_ms": dense_ms,
                        "decode_ms": moe_ms + dense_ms, "total_ms": total,
                        "ceiling_tps": ROWS / (total / 1000)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix_weights", type=Path)
    ap.add_argument("cycle_log", type=Path)
    args = ap.parse_args()
    try:
        result = estimate(json.loads(args.matrix_weights.read_text()), parse_cycles(args.cycle_log.read_text()))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        ap.error(str(error))
    print(json.dumps({"floors": result,
                      "assumptions": ["same1593ms current dataflow floor, before decode",
                                      "calibrate to305 MoE105ms and dense75ms decoder floors",
                                      "one row tile per routed expert;43 per shared expert",
                                      "metadata weights exclude MTP, head, and PLE embeddings",
                                      "ngram and routing differences stay outside this model"]}, indent=2))


if __name__ == "__main__": main()
