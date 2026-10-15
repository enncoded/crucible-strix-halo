#!/usr/bin/env python3
"""mtp-slot: decode speed against the MTP history length, one batched engine process, 1 stream.

A capped MTP history keeps only the last H rows. A window of H rows gives the MTP the same rows on a cold prompt,
so this probe measures windows 256, H1, H2, ... and 0 (the whole prompt) on one long prefix:
- stages (plan()): one per cap H. A window-0 request on the prefix P - H saves the frontier at P - H - 300 with the
  MTP rows (it extends the frontier of the last stage), a second one saves the end frontier;
- arms: each window decodes MAX_TOKENS from a cache hit. Window H restores the frontier at P - H - 300 and warms
  its last H rows; window 0 continues the saved MTP rows. Each stage runs H and the small windows mirrored
  (H 256 0 0 256 H), REPS times.
Every output must be equal (the MTP only proposes drafts).

  run ENGINE CKPT IDS OUT.json [--depth 195584] [--windows 256,16384,32768,65536,0] [--ctx 196608]
  table OUT.json [LOG]   (LOG: the engine log with CRUCIBLE_SPEC_STATS=1, for accepted drafts per step)
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.mtp_snapshot.abab import Clock, backend_for, body, collect_at, read_ids, row  # noqa: E402

# <|im_end|>\n<|im_start|>assistant\n (checkpoint tokenizer): a raw depth prompt alone makes the model stop at once.
ASSISTANT_TAIL = (248046, 198, 248045, 74455, 198)


class Step(NamedTuple):
    label: str
    length: int  # prompt[:length]
    window: int
    boundary: int  # cache_prefix_tokens: the frontier the request saves and finds


def plan(end, windows, reps):
    """One stage per cap H, largest first: build the frontier at end - H - 300 (it extends the last stage's frontier),
    build the end frontier from it, then run the arms H, the small windows and back, mirrored. A stage needs only its cap
    frontier and the end frontier, so the cache holds two (l7 and l7b: four did not fit the shared memory budget). The
    last arm of a stage restores the cap frontier, so the next build finds it and the cache evicts the end frontier."""
    small = [w for w in windows if w <= 300]
    caps = sorted((w for w in windows if w > 300), reverse=True)
    steps = []
    for cap in caps or [None]:
        if cap is not None:
            cut = end - cap
            steps.append(Step(f"build-{cap}", cut, 0, cut - 300))
        steps.append(Step("build-end", end, 0, end - 300))
        order = ([cap] if cap is not None else []) + small
        for rep in range(reps):
            steps += [Step(f"w{w}-r{rep}", end, w, end - 300) for w in order + order[::-1]]
    return steps


def run(args):
    doc = read_ids(args.ids)[:args.depth]
    prompt = doc + list(ASSISTANT_TAIL)
    windows = [int(w) for w in args.windows.split(",")]
    rows = []
    backend = backend_for(args, 1)
    try:
        for step in plan(len(prompt), windows, args.reps):
            ids = prompt[:step.length]
            max_tokens = 2 if step.label.startswith("build") else args.max_tokens
            with Clock() as clock:
                result = collect_at(backend, ids, body(ids, step.window, "auto", max_tokens), 1800, step.boundary)
            item = row(step.label, ids, step.window, result, clock.summary())
            rows.append(item | dict(request_id=result["done"].get("request_id")))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        backend.close()
    args.output.write_text(json.dumps(dict(depth=args.depth, prompt_tokens=len(prompt), windows=windows,
                                           rows=rows)) + "\n")
    return 0


def table(path, log):
    report = json.loads(Path(path).read_text())
    rows = [r for r in report["rows"] if not r["label"].startswith("build")]
    stats = {}
    if log:
        for match in re.finditer(r"spec_stats id=(\d+) .*?proposed=(\d+) accepted=(\d+) steps=(\d+)", Path(log).read_text()):
            stats[int(match[1])] = (int(match[2]), int(match[3]), int(match[4]))
    outputs = {json.dumps(r["output_sha256"]) for r in rows}
    base = None
    for window in report["windows"]:
        mine = [r for r in rows if r["window"] == window]
        rate = statistics.fmean(r["tok_s"] for r in mine)
        base = base or rate
        steps = [stats[r["request_id"]] for r in mine if r.get("request_id") in stats]
        per_step = sum(s[1] for s in steps) / sum(s[2] for s in steps) if steps else float("nan")
        print(f"window {window:>6}: {rate:.2f} tok/s ({rate / base - 1:+.1%} against the first window), "
              f"accepted per step {per_step:.2f}, n {len(mine)}, cached {sorted({r['cached_tokens'] for r in mine})}")
    print(f"outputs equal: {len(outputs) == 1}")
    return 0 if len(outputs) == 1 else 1


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("engine")
    one.add_argument("checkpoint")
    one.add_argument("ids", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--depth", type=int, default=195584)
    one.add_argument("--windows", default="256,16384,32768,65536,0")
    one.add_argument("--reps", type=int, default=1)
    one.add_argument("--ctx", type=int, default=196608)
    one.add_argument("--cache-mib", type=int, default=13312)  # two frontiers (at most 12.4 GB at 195K); see plan()
    one.add_argument("--max-tokens", type=int, default=512)
    two = commands.add_parser("table")
    two.add_argument("output", type=Path)
    two.add_argument("log", type=Path, nargs="?")
    args = parser.parse_args(argv[1:])
    return run(args) if args.command == "run" else table(args.output, args.log)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
