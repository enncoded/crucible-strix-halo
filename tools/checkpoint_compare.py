#!/usr/bin/env python3
"""Compare checkpoint metadata and tensor geometry without reading large payloads."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct


def header(path):
    with path.open("rb") as f:
        size, = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size))


def inventory(path):
    tensors, metadata = {}, {}
    for file in sorted(path.glob("*.safetensors")):
        data = header(file)
        metadata[file.name] = data.pop("__metadata__", {})
        for name, t in data.items():
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            tensors[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "bytes": t["data_offsets"][1] - t["data_offsets"][0]}
    return tensors, metadata


def family(name):
    return re.sub(r"(?<=\.)\d+(?=\.)|(?<=shard_)\d+", "*", name)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for key in sorted(a.keys() | b.keys()):
            out.update(differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip(".")))
        return out
    return {} if a == b else {prefix: [a, b]}


def weight_bytes(tensors, mtp=False):
    total = 0
    for name, t in tensors.items():
        if mtp:
            keep = name.startswith("mtp.") or name.startswith("lm_head.")
        else:
            keep = name.startswith("model.language_model.") or name.startswith("lm_head.")
        if not keep or ".ple_embedding." in name or name.endswith("embed_tokens.weight"):
            continue
        m = re.search(r"\.experts\.(\d+)\.", name)
        if m and int(m[1]) >= 10:
            continue
        total += t["bytes"]
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("quality", type=Path)
    args = p.parse_args()
    aa, ma = inventory(args.baseline)
    bb, mb = inventory(args.quality)
    grouped = Counter()
    for name in sorted(aa.keys() | bb.keys()):
        a, b = aa.get(name), bb.get(name)
        if a != b:
            grouped[(family(name), json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))] += 1
    ks = defaultdict(Counter)
    for name in aa.keys() | bb.keys():
        if not name.endswith(".trellis"):
            continue
        a, b = aa.get(name), bb.get(name)
        def bits(t):
            if t is None:
                return None
            s = t["shape"]
            return s[-1] / 16 if len(s) == 3 else (s[-1] - 1) * 16 / 160
        ks[family(name)][f"{bits(a)} -> {bits(b)}"] += 1
    qa = json.loads((args.baseline / "quantization_config.json").read_text())
    qb = json.loads((args.quality / "quantization_config.json").read_text())
    qdiff = differences(qa, qb)
    qgroups = Counter((family(k), json.dumps(v, sort_keys=True)) for k, v in qdiff.items())
    result = {
        "paths": [str(args.baseline), str(args.quality)],
        "tensor_counts": [len(aa), len(bb)],
        "payload_bytes": [sum(t["bytes"] for t in x.values()) for x in (aa, bb)],
        "streamed_trunk_bytes": [weight_bytes(x) for x in (aa, bb)],
        "streamed_mtp_bytes": [weight_bytes(x, True) for x in (aa, bb)],
        "config_diff": differences(*[json.loads((d / "config.json").read_text())
                                    for d in (args.baseline, args.quality)]),
        "quantization_diff": [{"field": k, "values": json.loads(v), "count": n}
                              for (k, v), n in sorted(qgroups.items())],
        "header_metadata": [ma, mb],
        "geometry_diff": [{"family": k, "baseline": json.loads(a), "quality": json.loads(b), "count": n}
                          for (k, a, b), n in sorted(grouped.items())],
        "trellis_bits": {k: dict(v) for k, v in sorted(ks.items())},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
