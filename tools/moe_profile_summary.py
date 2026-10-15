#!/usr/bin/env python3
"""Validate and summarize second-pass MoE counter captures. Not gate evidence."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

WMMA_TFLOPS = 49.0
DRAM_GB_S = 245.5
SIMDS = 80
# Known-size membw transfers expose a2x undercount on gfx1151/ROCm7.2.4.
SIZE_METRIC_BYTE_FACTOR = 2.0


@dataclass(frozen=True)
class Record:
    dispatch: int
    kernel: str
    counter: str
    value: float
    start: int
    end: int
    vgpr: int
    scratch: int
    threads: int
    grid: int

    @property
    def metadata(self):
        return (self.start, self.end, self.vgpr, self.scratch, self.threads, self.grid)

    @property
    def signature(self):
        return (self.kernel, self.grid, self.threads, self.vgpr, self.scratch)


def parse_row(row):
    result = Record(int(row["Dispatch_Id"]), row["Kernel_Name"], row["Counter_Name"],
                    float(row["Counter_Value"]), int(row["Start_Timestamp"]),
                    int(row["End_Timestamp"]), int(row["VGPR_Count"]),
                    int(row["Scratch_Size"]), int(row["Workgroup_Size"]), int(row["Grid_Size"]))
    if (not math.isfinite(result.value) or result.value < 0 or result.start < 0
            or result.end <= result.start or result.dispatch < 0):
        raise ValueError("invalid counter value, timestamp, or dispatch")
    if (not result.counter or min(result.vgpr, result.scratch) < 0 or result.threads != 128
            or result.grid <= 0 or result.grid % result.threads):
        raise ValueError("invalid counter metadata")
    return result


def dispatch_records(records):
    ordered = sorted(records, key=lambda r: (r.dispatch, r.counter))
    packets = tuple(tuple(group) for _, group in itertools.groupby(ordered, key=lambda r: r.dispatch))
    for packet in packets:
        if len({r.counter for r in packet}) != len(packet):
            raise ValueError("duplicate dispatch counter")
        if len({r.metadata for r in packet}) != 1:
            raise ValueError("conflicting dispatch metadata")
    if len({tuple(r.counter for r in p) for p in packets}) != 1:
        raise ValueError("counter set differs between dispatches")
    return packets


def kernel_summary(kernel, records, expected, rows):
    packets = dispatch_records(records)
    if len(packets) != expected:
        raise ValueError(f"expected {expected} dispatches for {kernel}, got {len(packets)}")
    if "PrefillMoeFxKernel" in kernel:
        factor = 4
    elif "PrefillMoeGemmKernel" in kernel:
        factor = 2
    else:
        raise ValueError(f"unknown MoE kernel: {kernel}")
    duration = sum(p[0].end - p[0].start for p in packets)
    flops = factor * rows * 11 * 2560 * 640 * expected
    useful = flops / duration * 1e-3
    counters = {name: statistics.median(r.value for r in records if r.counter == name)
                for name in sorted({r.counter for r in records})}
    clocks = [r.value for r in records if r.counter == "GRBM_GUI_ACTIVE"]
    result = {
        "kernel": kernel, "grid_size": packets[0][0].grid, "workgroup_size": packets[0][0].threads,
        "dispatches": len(packets), "milliseconds": duration * 1e-6,
        "vgpr_count": sorted({p[0].vgpr for p in packets}),
        "scratch_bytes": max(p[0].scratch for p in packets), "counters": counters,
        "useful_tflops": useful, "wmma_roofline_percent": useful / WMMA_TFLOPS * 100,
        "issue_clocks_valid": len(clocks) == expected and all(v > 0 for v in clocks),
    }
    for name, direction in (("FETCH_SIZE", "read"), ("WRITE_SIZE", "write")):
        if name in counters:
            external = sum(r.value for r in records if r.counter == name) * 1024 * SIZE_METRIC_BYTE_FACTOR / duration
            share = "external_read_roofline_percent" if direction == "read" else "external_write_vs_read_ceiling_percent"
            result = {**result, f"external_{direction}_gb_s": external, share: external / DRAM_GB_S * 100}
    if result["issue_clocks_valid"] and "SQ_INSTS_VALU" in counters:
        maps = tuple({r.counter: r.value for r in p} for p in packets)
        issue = statistics.median(p["SQ_INSTS_VALU"] / p["GRBM_GUI_ACTIVE"] / SIMDS for p in maps)
        result = {**result, "valu_instructions_per_simd_cycle": issue}
    return result


def summarize(source, expected=48, rows=4096):
    if expected <= 0 or rows <= 0:
        raise ValueError("rows and dispatch count must be positive")
    records = tuple(parse_row(row) for row in source)
    if not records:
        raise ValueError("empty counter capture")
    # A dispatch ID names one event globally, even when the kernel name differs.
    by_dispatch = sorted(records, key=lambda r: r.dispatch)
    for _, group in itertools.groupby(by_dispatch, key=lambda r: r.dispatch):
        if len({(r.kernel, r.metadata) for r in group}) != 1:
            raise ValueError("conflicting global dispatch identity or metadata")
    by_kernel = sorted(records, key=lambda r: r.kernel)
    for _, group in itertools.groupby(by_kernel, key=lambda r: r.kernel):
        dispatch_records(tuple(group))
    ordered = sorted(records, key=lambda r: r.signature)
    return [kernel_summary(signature[0], tuple(group), expected, rows)
            for signature, group in itertools.groupby(ordered, key=lambda r: r.signature)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--expected", type=int, default=48)
    parser.add_argument("--rows", type=int, default=4096)
    args = parser.parse_args()
    files = []
    try:
        for path in args.csv:
            with path.open(newline="") as stream:
                summary = summarize(csv.DictReader(stream), args.expected, args.rows)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files = [*files, {"path": str(path), "sha256": digest, "summary": summary}]
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"counter capture error: {error}\n")
    print(json.dumps({"wmma_roofline_tflops": WMMA_TFLOPS, "dram_roofline_gb_s": DRAM_GB_S,
        "size_metric_byte_factor": SIZE_METRIC_BYTE_FACTOR,
        "size_metric_calibration": "gfx1151/ROCm7.2.4, membw2GiB per direction, prefill-scan/calibration31",
        "warning": "Profiler times are diagnostic. GL2 external traffic does not separate MALL from DRAM. The read roofline is not a mixed-traffic ceiling.",
        "files": files}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
