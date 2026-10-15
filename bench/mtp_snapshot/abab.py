#!/usr/bin/env python3
"""mtp-snapshot ABAB decode runs in batched serve, one engine process each.

depth: one long prompt (an ids file). A cold full-window request saves a frontier (cache_prefix_tokens puts it 300
  tokens before the end), then PAIRS ABBA pairs of cache hits with mtp_prefix 256 and 0. Each hit prefills the last
  rows only, so the decode tok/s of the two windows compare on one prefix. Every output must be equal.
sessions: four concurrent sessions of synthetic code (TOKENS each), turn 1 cold and turn 2 a replay hit, for each
  window in ABBA order over ROUNDS rounds. Each (round, window) uses its own texts, so no entry crosses windows.
The GPU clock (pp_dpm_sclk) is sampled every 0.2 s during each request.

  depth ENGINE CKPT IDS OUT.json [--pairs 5] [--ctx 200704] [--max-tokens 512] [--streams 2]
  sessions ENGINE CKPT OUT.json [--rounds 2] [--tokens 32768] [--max-tokens 256]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.api import validate_request  # noqa: E402
from tools.serve_replay_gate import collect, digest  # noqa: E402
from bench.mtp_snapshot.probe import next_turn, prompt  # noqa: E402

SCLK = Path("/sys/class/drm/card1/device/pp_dpm_sclk")


class Clock:
    """Samples the current GPU clock (MHz) in a thread while a request runs."""
    def __init__(self):
        self.samples, self.stop = [], threading.Event()
    def __enter__(self):
        def run():
            while not self.stop.wait(0.2):
                try:
                    line = next(row for row in SCLK.read_text().splitlines() if row.rstrip().endswith("*"))
                    self.samples.append(int(line.split(":")[1].strip().split("Mhz")[0]))
                except (OSError, StopIteration, ValueError, IndexError):
                    pass
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
    def summary(self):
        return dict(median=statistics.median(self.samples), low=min(self.samples)) if self.samples else {}


def body(ids, window, cache, max_tokens):
    return validate_request(dict(prompt=list(ids), max_tokens=max_tokens, spec=7, mtp_prefix=window,
                                 cache_control=cache, temperature=0.0, seed=1), "completion", "crucible")


def row(label, ids, window, result, clock):
    done = result["done"]
    tokens = len(result["output_ids"])
    return dict(label=label, window=window, prompt_tokens=len(ids), output_sha256=digest(result["output_ids"]),
                tokens=tokens, ttft=result["ttft"], decode_seconds=done.get("decode_seconds"),
                tok_s=(tokens - 1) / done["decode_seconds"] if done.get("decode_seconds") else None,
                cached_tokens=done.get("cached_tokens"), cache_class=done.get("cache_class"),
                cache_bytes=done.get("cache_bytes"), sclk=clock)


def backend_for(args, streams):
    from server.batch_backend import BatchedEngineBackend
    backend = BatchedEngineBackend(args.engine, args.checkpoint, args.ctx, streams=streams, prefill=4096,
                                   conversation_cache_mib=args.cache_mib, session_replay=True, prefill_interleave=10)
    return backend


def collect_at(backend, ids, request, timeout, boundary):
    """collect() with a stable boundary (cache_prefix_tokens), so the request saves and finds that frontier."""
    submit = backend.submit
    backend.submit = lambda job_ids, job_request: submit(job_ids, job_request, cache_prefix_tokens=boundary)
    try:
        return collect(backend, ids, request, timeout)
    finally:
        backend.submit = submit


def read_ids(path):
    """The ids of a prompt file: separated by commas or by white space."""
    return [int(token) for token in path.read_text().replace(",", " ").split()]


def depth(args):
    ids = read_ids(args.ids)
    if args.trim:
        ids = ids[:len(ids) - args.trim]
    boundary = len(ids) - 300  # the frontier: the last grid point at or before it
    backend = backend_for(args, args.streams)
    rows, status = [], "CAPTURED"
    def one(label, window):
        with Clock() as clock:
            result = collect_at(backend, ids, body(ids, window, "auto", args.max_tokens), 900, boundary)
        rows.append(row(label, ids, window, result, clock.summary()))
        print(json.dumps(rows[-1]), flush=True)
    try:
        one("cold-0", 0)
        for pair in range(args.pairs):
            for window in ((256, 0) if pair % 2 == 0 else (0, 256)):
                one(f"pair{pair}-{window}", window)
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        status = f"FAIL: {error}"
    finally:
        backend.close()
    return finish(args, status, rows)


def sessions(args):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    backend = backend_for(args, 4)
    rows, status = [], "CAPTURED"
    def turn(label, window, items):
        def run(item):
            index, ids = item
            with Clock() as clock:
                result = collect(backend, ids, body(ids, window, "auto", args.max_tokens), 900)
            return row(f"{label}-s{index}", ids, window, result, clock.summary()) | dict(output_ids=result["output_ids"])
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            out = list(pool.map(run, items))
        for item in out:
            print(json.dumps({k: v for k, v in item.items() if k != "output_ids"}), flush=True)
        return out
    try:
        for round_index in range(args.rounds):
            order = (256, 0) if round_index % 2 == 0 else (0, 256)
            for window in order:
                base = 100000 * (round_index + 1) + (0 if window == 256 else 50000)
                firsts = [prompt(encode, base + 7000 * i, args.tokens) for i in range(4)]
                one = turn(f"r{round_index}-w{window}-t1", window, list(enumerate(firsts)))
                seconds = [next_turn(encode, ids, item["output_ids"], base + 7000 * i + 3000)
                           for i, (ids, item) in enumerate(zip(firsts, one))]
                two = turn(f"r{round_index}-w{window}-t2", window, list(enumerate(seconds)))
                rows += [{k: v for k, v in item.items() if k != "output_ids"} for item in one + two]
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        status = f"FAIL: {error}"
    finally:
        backend.close()
    return finish(args, status, rows)


def finish(args, status, rows):
    summary = {}
    for window in (256, 0):
        picked = [item for item in rows if item["window"] == window and item["tok_s"] and
                  not item["label"].startswith("cold")]
        if picked:
            summary[str(window)] = dict(n=len(picked), tok_s_mean=statistics.mean(item["tok_s"] for item in picked),
                                        ttft_mean=statistics.mean(item["ttft"] for item in picked))
    if args.command == "depth" and len({item["output_sha256"] for item in rows}) > 1:
        status = "FAIL: outputs differ"
    report = dict(status=status, command=args.command, engine=str(args.engine), rows=rows, summary=summary,
                  finished=time.time())
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(dict(status=status, summary=summary)))
    return 0 if status == "CAPTURED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("depth")
    one.add_argument("engine", type=Path)
    one.add_argument("checkpoint", type=Path)
    one.add_argument("ids", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--pairs", type=int, default=5)
    one.add_argument("--ctx", type=int, default=200704)
    one.add_argument("--max-tokens", type=int, default=512)
    one.add_argument("--cache-mib", type=int, default=26624)
    one.add_argument("--trim", type=int, default=0, help="drop this many tokens from the end of the ids")
    one.add_argument("--streams", type=int, default=2, choices=range(1, 9), metavar="1..8",
                     help="batch slots (1 fits 196K on the daily cache budget)")
    two = commands.add_parser("sessions")
    two.add_argument("engine", type=Path)
    two.add_argument("checkpoint", type=Path)
    two.add_argument("output", type=Path)
    two.add_argument("--rounds", type=int, default=2)
    two.add_argument("--tokens", type=int, default=32768)
    two.add_argument("--max-tokens", type=int, default=256)
    two.add_argument("--ctx", type=int, default=65536)
    two.add_argument("--cache-mib", type=int, default=26624)
    args = parser.parse_args(argv)
    return depth(args) if args.command == "depth" else sessions(args)


if __name__ == "__main__":
    raise SystemExit(main())
