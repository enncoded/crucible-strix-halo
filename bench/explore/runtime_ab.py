#!/usr/bin/env python3
"""Run paired resident measurements. Each process takes one queue turn."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--priority", type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ("HSA_USE_SVM", "GPU_PINNED_MIN_XFER_SIZE", "CRUCIBLE_WARMUP", "CRUCIBLE_PROFILE"):
        env.pop(key, None)
    ids_root = work_paths.worktree("explore-nx") / "results/nx/prompts"
    queue = work_paths.gpu_run()
    requests = args.output / "requests.txt"
    requests.write_text("".join(
        f'{name} "{ids_root}/code-{rows}.ids" 128 8 0 {repeat}\n'
        for name, rows, repeat in (("first4k", 4096, 1), ("warm4k", 4096, 2),
                                   ("first16k", 16384, 1), ("last4k", 4096, 3))))
    metrics = []
    for tier, suffix in (("305", "3.05bpw_h5_ng5"), ("405", "4.05bpw_h6_ng6")):
        ckpt = Path.home() / f"models/exl3/Qwen3.8-Flash-Next-exl3-{suffix}"
        for trial, label in enumerate("ABAB"):
            binary = args.baseline if label == "A" else args.candidate
            tag = f"{tier}-{trial}-{label}"
            command = [str(queue), "-p", str(args.priority), "-w", "7200", "--", "timeout", "300",
                       binary, str(ckpt), str(requests), "32768", "4096"]
            result = subprocess.run(command, text=True, capture_output=True, env=env)
            (args.output / f"{tag}.stdout").write_text(result.stdout)
            (args.output / f"{tag}.stderr").write_text(result.stderr)
            (args.output / f"{tag}.command.json").write_text(json.dumps(command))
            if result.returncode:
                raise RuntimeError(f"{tag}: exit {result.returncode}")
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            if len(events) != 5 or events[0].get("event") != "ready":
                raise RuntimeError(f"{tag}: incomplete output")
            item = dict(tier=tier, trial=trial, label=label, events=events)
            metrics.append(item)
            (args.output / "runs.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(tag, [(x["name"], x["prefill_seconds"], x["decode_seconds"])
                        for x in events[1:]], flush=True)
    for tier in ("305", "405"):
        runs = [x for x in metrics if x["tier"] == tier]
        for position in range(1, 5):
            ids = runs[0]["events"][position]["output_ids"]
            if any(x["events"][position]["output_ids"] != ids for x in runs):
                raise RuntimeError(f"{tier}/{position}: output IDs differ")
    print("All paired output IDs match. Use tools/gates.py for gate evidence.", flush=True)


if __name__ == "__main__":
    main()
