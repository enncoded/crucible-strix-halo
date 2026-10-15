#!/usr/bin/env python3
"""mtp-slot decode ABAB in one GPU lease: engine processes alternate A, B, B, A, A, B, ... (two engines do not
fit in memory at once). Each process loads once and runs the same greedy speculative requests, one at a time, so
engine load time does not enter the timing.

  run ENGINE_A ENGINE_B CKPT OUT.json [--pairs 5] [--requests 3] [--ctx 65536] [--prompt-tokens 4096] [--max-tokens 384]
  table OUT.json

The requests use fixed synthetic prompts (bench/mtp_snapshot/probe.py), sent with cache bypass. Decode rate:
(completion tokens - 1) / (seconds - TTFT). The GPU clock (hwmon freq1_input) is sampled every 0.2 s; each row
records the mean clock of its request. The outputs of A and B must be equal.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCLK = sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/freq1_input"))


class Clock:
    def __init__(self):
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop.wait(0.2):
            if SCLK:
                self.samples.append((time.monotonic(), int(SCLK[0].read_text()) / 1e6))

    def mean(self, start, end):
        values = [mhz for t, mhz in self.samples if start <= t <= end]
        return statistics.fmean(values) if values else None


def run(args):
    from tokenizers import Tokenizer
    from server.api import validate_request
    from server.batch_backend import BatchedEngineBackend
    from tools.serve_replay_gate import collect
    sys.path.insert(0, str(ROOT / "bench" / "mtp_snapshot"))
    import probe
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    clock, rows = Clock(), []
    prompts = [probe.prompt(encode, 20000 + 500 * i, args.prompt_tokens) for i in range(args.requests)]
    try:
        for pair in range(args.pairs):
            for name in ("A", "B") if pair % 2 == 0 else ("B", "A"):
                path = args.engine_a if name == "A" else args.engine_b
                engine = BatchedEngineBackend(str(path), str(args.checkpoint), args.ctx, streams=4, prefill=4096,
                                              conversation_cache_mib=args.cache_mib)
                try:
                    for index, ids in enumerate(prompts):
                        body = dict(prompt=ids, max_tokens=args.max_tokens, spec=7, temperature=0.0, seed=1,
                                    cache_control="bypass")
                        start = time.monotonic()
                        row = collect(engine, ids, validate_request(body, "completion", "crucible"), 600)
                        end = time.monotonic()
                        tokens = len(row["output_ids"])
                        rows.append(dict(pair=pair, engine=name, request=index, tokens=tokens, ttft=row["ttft"],
                                         seconds=row["seconds"], tok_s=(tokens - 1) / (row["seconds"] - row["ttft"]),
                                         sclk_mhz=clock.mean(start + row["ttft"], end), output_ids=row["output_ids"]))
                        print(f"pair {pair} {name} request {index}: {rows[-1]['tok_s']:.2f} tok/s "
                              f"sclk {rows[-1]['sclk_mhz']}", flush=True)
                finally:
                    engine.close()
    finally:
        clock.stop.set()
    args.output.write_text(json.dumps(dict(engines=dict(A=str(args.engine_a), B=str(args.engine_b)), rows=rows)) + "\n")
    return 0


def table(path):
    rows = json.loads(Path(path).read_text())["rows"]
    pairs = sorted({r["pair"] for r in rows})
    first = {r["request"]: r["output_ids"] for r in rows if r["pair"] == 0 and r["engine"] == "A"}
    equal = all(r["output_ids"] == first[r["request"]] for r in rows)
    ratios = []
    for pair in pairs:
        mean = lambda name, key: statistics.fmean(r[key] for r in rows if r["pair"] == pair and r["engine"] == name)  # noqa: E731
        ratios.append(mean("B", "tok_s") / mean("A", "tok_s"))
        print(f"pair {pair}: A {mean('A', 'tok_s'):.2f} tok/s ({mean('A', 'sclk_mhz'):.0f} MHz)  "
              f"B {mean('B', 'tok_s'):.2f} tok/s ({mean('B', 'sclk_mhz'):.0f} MHz)  B/A {ratios[-1]:.4f}")
    print(f"B/A median {statistics.median(ratios):.4f} mean {statistics.fmean(ratios):.4f} "
          f"min {min(ratios):.4f} max {max(ratios):.4f}; outputs equal: {equal}")
    return 0 if equal else 1


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("engine_a", type=Path)
    one.add_argument("engine_b", type=Path)
    one.add_argument("checkpoint", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--pairs", type=int, default=5)
    one.add_argument("--requests", type=int, default=3)
    one.add_argument("--ctx", type=int, default=65536)
    one.add_argument("--cache-mib", type=int, default=1024)
    one.add_argument("--prompt-tokens", type=int, default=4096)
    one.add_argument("--max-tokens", type=int, default=384)
    two = commands.add_parser("table")
    two.add_argument("output", type=Path)
    args = parser.parse_args(argv[1:])
    return run(args) if args.command == "run" else table(args.output)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
