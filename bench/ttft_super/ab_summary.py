"""Summary of the ttft-super warm-TTFT A/B: paired client TTFT, server TTFT, the server time before submit,
and output-ID equality at equal restore points. Usage: python -I ab_summary.py AB_DIR
"""
import glob
import json
import re
import statistics
import sys
from pathlib import Path


def server_turns(path):
    turns = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = re.search(r"crucible_turn (\{.*\})", line)
        if match:
            turn = json.loads(match.group(1))
            if turn.get("prompt_tokens"):
                turns.append(turn)
    return turns


def main():
    root = Path(sys.argv[1])
    rows = {}  # (lease, arm, step) -> row with server fields
    for path in sorted(glob.glob(str(root / "rows-ab*.jsonl"))):
        lease = Path(path).stem.removeprefix("rows-")
        by_arm = {}
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            by_arm.setdefault(row["arm"], []).append(row)
        for arm, arm_rows in by_arm.items():
            turns = server_turns(root / f"{lease}-{arm}.server.log")
            for index, row in enumerate(arm_rows):
                turn = turns[index] if index < len(turns) else {}
                row.update(server_ttft=turn.get("ttft_s"), prompt=turn.get("prompt_tokens"),
                           cached=turn.get("cached_tokens"), prefill_tokens=turn.get("prefill_tokens"))
                rows[(lease, arm, row["step"])] = row
    steps = sorted({key[2] for key in rows}, key=lambda s: (s != "cold", s))
    leases = sorted({key[0] for key in rows})
    print("step | pairs | A client ms | B client ms | B-A client ms (median, min, max) | B-A server ms | "
          "pre-submit A/B ms | IDs equal")
    failures = 0
    for step in steps:
        pairs = [(rows[(l, "A", step)], rows[(l, "B", step)]) for l in leases
                 if (l, "A", step) in rows and (l, "B", step) in rows]
        if not pairs:
            continue
        ids = {tuple(row["ids"]) for (l, _, s), row in rows.items() if s == step}
        failures += len(ids) != 1
        diff = [1000 * (b["ttft_s"] - a["ttft_s"]) for a, b in pairs]
        sdiff = [1000 * (b["server_ttft"] - a["server_ttft"]) for a, b in pairs if a["server_ttft"] and b["server_ttft"]]
        pre = [[1000 * (r["ttft_s"] - r["server_ttft"]) for r in side if r["server_ttft"]] for side in zip(*pairs)]
        med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
        print(f"{step} | {len(pairs)} | {med([1000 * a['ttft_s'] for a, _ in pairs]):.0f} | "
              f"{med([1000 * b['ttft_s'] for _, b in pairs]):.0f} | {med(diff):+.0f} ({min(diff):+.0f}, {max(diff):+.0f}) | "
              f"{med(sdiff):+.0f} | {med(pre[0]):.0f}/{med(pre[1]):.0f} | {'yes' if len(ids) == 1 else 'NO'} "
              f"(prompt {pairs[0][0]['prompt']}, cached {pairs[0][0]['cached']}, T {pairs[0][0]['temperature']})")
    print("VERDICT", "PASS" if failures == 0 and rows else "FAIL", f"({failures} steps with different IDs)")
    return 0 if failures == 0 and rows else 1


if __name__ == "__main__":
    sys.exit(main())
