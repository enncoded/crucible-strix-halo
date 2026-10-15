#!/usr/bin/env python3
"""Supplement shared gates with cache growth, logits, and mixed serve checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402


def run(command, output, env=None, input_text=None):
    queue = work_paths.gpu_run()
    wrapped = [str(queue), "-p", "2", "-w", "7200", "--", "timeout", "-k", "10", "300", *map(str, command)]
    result = subprocess.run(wrapped, env=env, input=input_text, capture_output=True, text=True)
    output.with_suffix(".stdout").write_text(result.stdout)
    output.with_suffix(".stderr").write_text(result.stderr)
    output.with_suffix(".command.json").write_text(json.dumps(wrapped, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"{output.name}: exit {result.returncode}")
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--before-probe", required=True)
    p.add_argument("--after-probe", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ("HSA_USE_SVM", "GPU_PINNED_MIN_XFER_SIZE", "CRUCIBLE_WARMUP", "CRUCIBLE_PROFILE"):
        env.pop(key, None)
    ids_root = work_paths.worktree("explore-nx") / "results/nx/prompts"
    c305 = Path.home() / "models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
    result = {}
    for label, binary in (("before", args.before_probe), ("after", args.after_probe)):
        stdout = run([binary, c305], args.output / f"cache-{label}", env)
        records = [json.loads(x) for x in stdout.splitlines() if x.startswith("{")]
        if len(records) != 30 or "cache clear: zero bytes" not in stdout:
            raise RuntimeError(f"{label}: incomplete cache probe")
        if label == "after" and any(x["cached_bytes"] > 1 << 30 for x in records):
            raise RuntimeError("cache exceeds byte cap")
        result[label] = dict(peak_retained_bytes=max(x["cached_bytes"] for x in records), records=records)
        print(label, result[label]["peak_retained_bytes"], flush=True)
    for tier, suffix in (("305", "3.05bpw_h5_ng5"), ("405", "4.05bpw_h6_ng6")):
        ckpt = Path.home() / f"models/exl3/Qwen3.8-Flash-Next-exl3-{suffix}"
        paths = {}
        for label, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
            target = args.output / f"logits-{tier}-{label}"
            target.mkdir()
            out_env = dict(env, CRUCIBLE_PF_OUT=str(target))
            tags = [f"code{n}:{ids_root}/code-{n}.ids:4096:0:0:0:0:0:0" for n in (1024, 4096, 16384)]
            run([binary, "pfbench", ckpt, "--prefill", "4096", *tags],
                args.output / f"pfbench-{tier}-{label}", out_env)
            paths[label] = target
        hashes = {}
        for n in (1024, 4096, 16384):
            files = [paths[label] / f"code{n}.logits.f32" for label in ("baseline", "candidate")]
            data = [f.read_bytes() for f in files]
            if len(data[0]) != 248320 * 4 or data[0] != data[1]:
                raise RuntimeError(f"{tier}/{n}: logits differ or truncate")
            hashes[n] = hashlib.sha256(data[0]).hexdigest()
        result[f"logits-{tier}"] = hashes
        print(tier, "1K/4K/16K logits bitwise equal", flush=True)
    ids = [int(x) for x in (ids_root / "code-4096.ids").read_text().strip().split(",")]
    lengths = (24, 100, 1024, 2048, 3072, 4096, 4095, 4094, 4093, 4096, 100)
    requests = "".join(json.dumps(dict(ids=ids[:n], max_tokens=1, spec=0, stop_ids=[])) + "\n" for n in lengths)
    runs = []
    for label, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
        stdout = run([binary, "serve", c305, "--ctx", "32768"],
                     args.output / f"serve-{label}", env, requests)
        events = [json.loads(x) for x in stdout.splitlines() if x.startswith("{")]
        if len([x for x in events if x.get("event") == "done"]) != len(lengths):
            raise RuntimeError(f"serve-{label}: incomplete protocol")
        if any(x.get("event") == "error" for x in events):
            raise RuntimeError(f"serve-{label}: request error")
        runs.append([x["id"] for x in events if x.get("event") == "token"])
    if len(runs[0]) != len(lengths) or runs[0] != runs[1]:
        raise RuntimeError("mixed serve output IDs differ")
    result["serve"] = dict(lengths=lengths, ids=runs[0])
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Mixed serve IDs match. Use tools/gates.py for gate evidence.", flush=True)


if __name__ == "__main__":
    main()
