"""Check the resident sampling sweep after its GPU process stops."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=Path.home() / "crucible-work/wt/integrate/results/bounds-final/chunk8")
    args = parser.parse_args()
    files = sorted((args.output / "chunk8").glob("h_*.f32"))
    assert len(files) == 49, len(files)
    for file in files:
        data = file.read_bytes()
        assert data == (args.output / "chunk1" / file.name).read_bytes(), file
        assert data == (args.baseline / file.name).read_bytes(), file
    text = (args.output / "sweep.log").read_text()
    assert "MISMATCH" not in text
    pattern = r"^== ([^:]+): ([\d.]+) tok/s,.*hash ([0-9a-f]+), (\S+)$"
    rows = {name: dict(tokens_per_second=float(rate), hash=digest, verdict=verdict)
            for name, rate, digest, verdict in re.findall(pattern, text, re.MULTILINE)}
    for name in ("greedy2", "greedy4", "greedy_filters"):
        assert rows[name]["verdict"] == "exact", (name, rows[name])
    for name in ("top1_serial", "top1_spec"):
        assert rows[name]["verdict"] == "exact-top1", (name, rows[name])
    assert rows["code_filtered"]["hash"] == rows["code_filtered_repeat"]["hash"]
    for name, row in rows.items():
        if "serial" in name or name == "greedy0":
            row["percent_memory_roofline_53"] = 100 * row["tokens_per_second"] / 53
        elif name not in ("gate1", "gate8"):
            row["percent_spec_estimate_150"] = 100 * row["tokens_per_second"] / 150
            row["percent_spec_estimate_200"] = 100 * row["tokens_per_second"] / 200
    summary = dict(gate1_files=49, gate2_files=49, gate3="pass", seed_repeat="pass", cases=rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: 49 exact dumps, row equality, greedy gates, top-k 1 equality, and repeated seed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
