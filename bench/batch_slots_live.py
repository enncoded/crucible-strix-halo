#!/usr/bin/env python3
"""Device supplement for batch slot classes (G4).

One model load with 8 streams: 3 full slots and 5 short slots. A request in a short slot must give the
same output as the same request in a full slot. Run only under gpu-run with `timeout -k 10 300`.
Supplemental evidence; it does not replace tools/gates.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.api import validate_request
import server.backend as engine_backend
from server.batch_backend import BatchedEngineBackend


def prompt(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def request(ids, tokens, temperature=0.0, seed=0, cache="bypass"):
    body = dict(prompt=list(ids), max_tokens=tokens, spec=0, cache_control=cache, temperature=temperature, seed=seed)
    return validate_request(body, "completion", "crucible")


def collect(job):
    output = []
    while True:
        event = job.events.get(timeout=280)
        if event["event"] == "error": raise RuntimeError(event)
        if event["event"] == "done": return output, event
        output.append(event["id"])


def run(backend, ids, tokens, **sampling):
    started = time.monotonic()
    out, done = collect(backend.submit(ids, request(ids, tokens, **sampling)))
    return dict(ids=out, seconds=time.monotonic() - started, slot_context=done.get("slot_context"),
                cached_tokens=done.get("cached_tokens"),
                prefill_seconds=done.get("prefill_seconds"), decode_seconds=done.get("decode_seconds"))


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"): return int(line.split()[1]) * 1024
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--short-ctx", type=int, default=32768)
    parser.add_argument("--cache-mib", type=int, default=4096)
    options = parser.parse_args()
    # Random prompts often stop early; full-length outputs make the comparisons meaningful.
    engine_backend.STOP_IDS = ()
    x, long_ids = prompt(21, 3000), prompt(22, options.short_ctx + 200)
    base = prompt(23, 8300)
    grown = base + prompt(24, 200)
    rows = {}
    backend = BatchedEngineBackend(options.engine, options.checkpoint, options.ctx, streams=8, short_streams=5,
                                   short_ctx=options.short_ctx, conversation_cache_mib=options.cache_mib)
    try:
        rows["ready_mem_available"] = mem_available()
        rows["x_short"] = run(backend, x, 64)
        rows["x_short_sampled"] = run(backend, x, 64, temperature=0.8, seed=5)
        # Five holders fill the short slots, so the same requests must take a full slot.
        readers = ThreadPoolExecutor(max_workers=5)
        holders = [readers.submit(collect, backend.submit(ids, request(ids, 400)))
                   for ids in (prompt(30 + index, 64) for index in range(5))]
        time.sleep(2.0)
        rows["x_full"] = run(backend, x, 64)
        rows["x_full_sampled"] = run(backend, x, 64, temperature=0.8, seed=5)
        rows["long"] = run(backend, long_ids, 8)
        rows["seed_full"] = run(backend, base, 8, cache="auto")  # captures frontier 8192 from a full slot
        rows["holders"] = [dict(tokens=len(holder.result(timeout=280)[0]),
                                slot_context=holder.result()[1].get("slot_context")) for holder in holders]
        readers.shutdown()
        # Cross-class restore: the full-slot snapshot into a short slot, against bypass in a short slot.
        rows["hit_short"] = run(backend, grown, 32, cache="auto")
        rows["ref_short"] = run(backend, grown, 32)
        rows["after_mem_available"] = mem_available()
    finally:
        backend.close()
    checks = {
        "x_short_in_short_slot": rows["x_short"]["slot_context"] == options.short_ctx,
        "x_full_in_full_slot": rows["x_full"]["slot_context"] == options.ctx,
        "greedy_short_equals_full": rows["x_short"]["ids"] == rows["x_full"]["ids"],
        "sampled_short_equals_full": rows["x_short_sampled"]["ids"] == rows["x_full_sampled"]["ids"],
        "long_in_full_slot": rows["long"]["slot_context"] == options.ctx and len(rows["long"]["ids"]) == 8,
        "seed_in_full_slot": rows["seed_full"]["slot_context"] == options.ctx,
        "cross_class_hit": rows["hit_short"]["cached_tokens"] == 8192 and rows["hit_short"]["slot_context"] == options.short_ctx,
        "cross_class_hit_equals_bypass": rows["hit_short"]["ids"] == rows["ref_short"]["ids"]
                                         and rows["ref_short"]["slot_context"] == options.short_ctx
                                         and len(rows["hit_short"]["ids"]) == 32,
        "full_length_outputs": len(rows["x_short"]["ids"]) == 64 and len(rows["x_full_sampled"]["ids"]) == 64,
        "holders_in_short_slots": all(h["slot_context"] == options.short_ctx and h["tokens"] == 400 for h in rows["holders"]),
    }
    report = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, rows=rows,
                  slots=dict(full=[options.ctx] * 3, short=[options.short_ctx] * 5), cache_mib=options.cache_mib)
    Path(options.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(status=report["status"], checks=checks)))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
