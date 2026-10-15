#!/usr/bin/env python3
"""CPU model of PrefillLinear metadata bytes at default tile shapes."""
import ctypes
import json


class LinJob(ctypes.Structure):
    _fields_ = [(f"p{i}", ctypes.c_void_p) for i in range(9)] + [(f"i{i}", ctypes.c_int) for i in range(9)]


class GemmJob(ctypes.Structure):
    _fields_ = [(f"p{i}", ctypes.c_void_p) for i in range(4)] + [(f"i{i}", ctypes.c_int) for i in range(6)]


def group(rows, shapes, rm):
    items = sum(((rows + 16 * rm - 1) // (16 * rm)) * (o // 128) for i, o in shapes)
    transforms = rows * sum((i + o) // 128 for i, o in shapes)
    return len(shapes) * (ctypes.sizeof(LinJob) + ctypes.sizeof(GemmJob)) + items * 16 + transforms * 12


def bytes_for(rows):
    gdn = group(rows, ((2560, 10240), (2560, 6144)), 8)
    qsa = group(rows, ((2560, 12288), (2560, 512), (2560, 512), (2560, 640)), 8)
    output = group(rows, ((6144, 2560),), 8 if rows > 1024 else 4)
    return 36 * gdn + 12 * qsa + 48 * output


if __name__ == "__main__":
    for rows in (24, 100, 256, 1024, 2048, 4096):
        print(json.dumps(dict(rows=rows, plan_bytes=bytes_for(rows), mib=bytes_for(rows) / 1048576)))
