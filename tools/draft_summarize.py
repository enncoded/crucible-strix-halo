#!/usr/bin/env python3
"""Check every recorded continuation against serial and write compact results."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    baselines = {}
    for directory in ("baseline", "long-baseline"):
        for path in (root / directory).glob("*.serial.json"):
            data = json.loads(path.read_text())
            baselines[data["prompt"]] = data["ids"]
    rows = []
    for path in sorted(root.glob("*/*.json")):
        data = json.loads(path.read_text())
        if "ids" not in data:
            continue
        expected = baselines[data["prompt"]]
        if data["ids"] != expected:
            mismatch = next((i for i, (a, b) in enumerate(zip(data["ids"], expected)) if a != b),
                            min(len(data["ids"]), len(expected)))
            raise ValueError(f"{path}: serial mismatch at token {mismatch}")
        ids_hash = hashlib.sha256(",".join(map(str, expected)).encode()).hexdigest()
        rows.append({"variant": path.parent.name, "prompt": data["prompt"],
                     "mode": data["mode"], "tok_s": data["tok_s"],
                     "acceptance_percent": data["acceptance"],
                     "accepted_per_forward": data["accepted_per_forward"],
                     "tokens": len(expected), "exact_serial": True,
                     "ids_sha256": ids_hash, "time": data["time"],
                     "source": str(path.relative_to(root))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rows = summarize(args.results)
    args.output.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")
    print(f"PASS: {len(rows)} runs match serial")
    for row in rows:
        print(row["variant"], row["prompt"], row["mode"], row["tok_s"], row["acceptance_percent"])


if __name__ == "__main__":
    main()
