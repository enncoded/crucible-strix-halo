#!/usr/bin/env python3
"""Count static AMDGPU assembly instructions and report compiler resources.

Usage: isa_mix.py FILE.s SUBSTRING [SUBSTRING...]
       isa_mix.py FILE.s --all --json

Counts are static, not execution counts or issue cycles. A backward branch
range is a loop candidate, not a proven hot loop. Occupancy is the compiler
limit, before dynamic LDS, launch geometry, or runtime resource limits.
"""
import argparse
import collections
import json
from pathlib import Path
import re


SYMBOL = re.compile(r"^([_A-Za-z0-9.$]+):\s*(?:[;#].*)?$")
LABEL = re.compile(r"^(\.LBB[0-9_]+):")
BRANCH = re.compile(r"^\s*s_(?:cbranch_\w+|branch)\s+(\.LBB[0-9_]+)")
OPCODE = re.compile(r"^(?:s_|v_|ds_|global_|buffer_|flat_|scratch_|image_|tbuffer_|exp\b)")
ZERO_WAIT = re.compile(
    r"(?:vmcnt|lgkmcnt|vscnt)\(0\)|s_waitcnt_vscnt\s+\w+,\s*(?:0x0+|0)$"
)
RESOURCE_KEYS = {
    "NumVgprs": "vgpr", "NumSgprs": "sgpr", "ScratchSize": "scratch_bytes_lane",
    "Occupancy": "occupancy_waves_simd", "LDSByteSize": "lds_bytes",
    "amdhsa_group_segment_fixed_size": "lds_bytes",
    "amdhsa_private_segment_fixed_size": "scratch_bytes_lane",
    "amdhsa_next_free_vgpr": "vgpr", "amdhsa_next_free_sgpr": "sgpr",
}


def functions(lines):
    """Keep trailing kernel metadata, which follows .Lfunc_end in clang output."""
    name, body = None, []
    executable, ended = False, False
    for line in lines:
        match = SYMBOL.match(line)
        if match and not match[1].startswith(".L"):
            if name and executable:
                yield name, body
            name, body = match[1], []
            executable, ended = False, False
        elif name:
            body.append(line)
            ended = ended or line.startswith(".Lfunc_end")
            # Data and YAML labels also match SYMBOL, but have no code body.
            if not ended and instruction(line):
                executable = True
    if name and executable:
        yield name, body


def instruction(line):
    text = line.strip().split(";", 1)[0].split("//", 1)[0].strip()
    if not OPCODE.match(text) or text.endswith(":"):
        return None
    return text


def group(op):
    if "wmma" in op or "mfma" in op:
        return "wmma"
    if "dot2" in op:
        return "dot2"
    if op.startswith(("global_", "buffer_", "flat_", "image_", "tbuffer_")):
        return "vmem"
    if op.startswith("scratch_"):
        return "scratch"
    if op.startswith(("s_load", "s_buffer_", "s_store", "s_mem")):
        return "smem"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("s_wait", "s_delay")):
        return "wait"
    if op.startswith(("s_branch", "s_cbranch", "s_endpgm", "s_barrier")):
        return "control"
    return "salu" if op.startswith("s_") else "valu"


def count_segment(body, start, end):
    mix, groups, features = (collections.Counter() for _ in range(3))
    waits = []
    issued = 0
    for index in range(start, end):
        text = instruction(body[index])
        if text is None:
            continue
        issued += 1
        parts = text.split("::")
        if len(parts) == 2:
            features["vopd_pairs"] += 1
        for part in parts:
            op = part.strip().split()[0]
            mix[op] += 1
            groups[group(op)] += 1
            features["packed"] += int(op.startswith("v_pk_"))
            features["readfirstlane"] += int(op.startswith("v_readfirstlane"))
            features["mul_lo_u32"] += int(op.startswith("v_mul_lo_u32"))
            features["exec_mask"] += int("exec" in part)
            features["barrier"] += int(op.startswith("s_barrier"))
        if parts[0].strip().startswith(("s_wait", "s_delay")):
            waits.append({"line": index + 1, "text": text})
            features["wait_zero"] += int(bool(ZERO_WAIT.search(text)))
    return {"instructions": issued, "operations": sum(mix.values()),
            "groups": dict(groups), "mnemonics": dict(mix),
            "features": dict(features), "waits": waits}


def resources(body):
    result = {}
    for line in body:
        match = re.search(r"(?:;\s*|\.)(\w+)(?::\s*|\s+)(\d+)", line)
        if match and match[1] in RESOURCE_KEYS:
            result[RESOURCE_KEYS[match[1]]] = int(match[2])
    return result


def analyze_function(name, body):
    # Do not count ISA-looking text in YAML or metadata after the code ends.
    end = next((i for i, line in enumerate(body) if line.startswith(".Lfunc_end")), len(body))
    labels, loops = {}, []
    for index, line in enumerate(body[:end]):
        label = LABEL.match(line)
        if label:
            labels[label[1]] = index
        branch = BRANCH.match(line)
        if branch and branch[1] in labels and labels[branch[1]] < index:
            start = labels[branch[1]]
            loops.append({"start_line": start + 1, "end_line": index + 1,
                          "target": branch[1], **count_segment(body, start, index + 1)})
    return {"name": name, "resources": resources(body),
            "whole": count_segment(body, 0, end), "loops": loops}


def analyze(lines):
    return [analyze_function(name, body) for name, body in functions(lines)]


def print_report(report, all_loops=False):
    print(report["name"])
    print("  resources: " + ", ".join(f"{key}={value}" for key, value in report["resources"].items()))
    loops = report["loops"]
    selected = loops if all_loops else [max(loops, key=lambda loop: loop["instructions"])] if loops else []
    segments = [("whole function", report["whole"])]
    segments.extend((f"backedge lines {loop['start_line']}:{loop['end_line']}", loop) for loop in selected)
    for label, segment in segments:
        print(f"  {label}: {segment['instructions']} instructions, {segment['operations']} operations")
        print("    " + ", ".join(f"{key} {value}" for key, value in segment["groups"].items()))
        if not all_loops:
            for op, count in sorted(segment["mnemonics"].items(), key=lambda item: -item[1]):
                print(f"    {count:5d} {op}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("substrings", nargs="*")
    parser.add_argument("--all", action="store_true", help="Report all matching functions.")
    parser.add_argument("--json", action="store_true", help="Write machine-readable reports.")
    parser.add_argument("--loops", action="store_true", help="Show every backedge candidate.")
    args = parser.parse_args()
    try:
        lines = args.file.read_text().splitlines()
    except OSError as error:
        parser.error(str(error))
    reports = []
    for name, body in functions(lines):
        if all(sub in name for sub in args.substrings):
            reports.append(analyze_function(name, body))
            if not args.all:
                break
    if not reports:
        parser.error("No matching function.")
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print_report(report, args.loops)


if __name__ == "__main__":
    main()
