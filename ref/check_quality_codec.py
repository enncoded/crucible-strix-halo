#!/usr/bin/env python3
"""Compare quality_codec HIP dumps with independent NumPy storage decoding."""
import argparse
from pathlib import Path

import numpy as np

from exl3_ref import decode_codebook, reconstruct_inner
from oracle import Ckpt


def compare(actual, expected, label):
    a = np.ascontiguousarray(actual).view(np.uint8)
    b = np.ascontiguousarray(expected).view(np.uint8)
    mismatches = np.count_nonzero(a != b)
    print(f"{label}: {expected.size} values, {mismatches} unequal bytes")
    assert mismatches == 0, label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("dumps", type=Path)
    args = p.parse_args()
    ck = Ckpt(args.checkpoint)
    names = ["model.language_model.layers.0.linear_attn.in_proj_qkv",
             "model.language_model.layers.0.mlp.experts.0.down_proj", "mtp.fc_embedding"]
    for i, name in enumerate(names):
        stored = {k: ck.t(f"{name}.{k}").numpy() for k in ("trellis", "suh", "svh", "mul1")}
        expected = reconstruct_inner(stored)
        actual = np.fromfile(args.dumps / f"linear{i}.f16", dtype=np.float16).reshape(expected.shape)
        compare(actual, expected, name)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    single = prefix + "trellis" in ck.map
    shape = ck.t(prefix + ("trellis" if single else "shard_0.trellis")).shape
    K = (shape[1] - 1) * 16 // 160
    bias = ck.t(prefix + "head_bias").numpy().astype(np.float64)
    expected = np.empty((16, 160), dtype=np.float32)
    for h in range(16):
        row = h if h < 8 else 320001536 - 16 + h
        name = prefix + ("trellis" if single else f"shard_{row // shape[0]}.trellis")
        packed = ck.t(name)[row if single else row % shape[0]].numpy().view(np.uint16)
        scale = packed[:1].view(np.float16).astype(np.float64)[0]
        # Literal LSB-first ring oracle. This does not use GPU symbol extraction.
        bits = np.unpackbits(packed[1:].view(np.uint8), bitorder="little")
        states = np.zeros(160, dtype=np.uint16)
        for i in range(160):
            for bit in range(16):
                at = ((i - bit // K) % 160) * K + bit % K
                states[i] |= int(bits[at]) << bit
        values = decode_codebook(states).astype(np.float64)
        expected[h] = (values * scale + bias[h]).astype(np.float16).astype(np.float32)
    actual = np.fromfile(args.dumps / "ple.f32", dtype=np.float32).reshape(16, 160)
    compare(actual, expected, f"PLE K={K}")


if __name__ == "__main__":
    main()
