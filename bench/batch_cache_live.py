#!/usr/bin/env python3
"""Device supplement for the batch prefix cache (G2).

One model load. Every cached request is compared with the same request under cache_control=bypass,
which keeps the same fixed chunk grid and does no cache reads or writes. Run only under gpu-run with
`timeout -k 10 230`. Supplemental evidence; it does not replace tools/gates.py.
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
from server.batch_backend import BatchedEngineBackend


def prompt(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def request(ids, tokens, cache, temperature=0.0, seed=0):
    body = dict(prompt=list(ids), max_tokens=tokens, spec=0, cache_control=cache, temperature=temperature, seed=seed)
    return validate_request(body, "completion", "crucible")


def collect(job):
    output = []
    while True:
        event = job.events.get(timeout=200)
        if event["event"] == "error": raise RuntimeError(event)
        if event["event"] == "done": return output, event
        output.append(event["id"])


def run(backend, ids, tokens, cache, shared=0, **sampling):
    started = time.monotonic()
    options = dict(shared_prefix_tokens=shared) if shared else {}
    out, done = collect(backend.submit(ids, request(ids, tokens, cache, **sampling), **options))
    keys = ("cached_tokens", "prefill_tokens", "cache_bytes", "restore_seconds", "capture_seconds", "cache_class")
    return dict(ids=out, seconds=time.monotonic() - started, **{key: done.get(key) for key in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--cache-mib", type=int, default=1024)
    options = parser.parse_args()
    base, second, third = prompt(1, 8300), prompt(2, 8300), prompt(3, 8300)
    a, b = base + prompt(11, 200), base + prompt(12, 300)
    rows, checks = {}, []
    backend = BatchedEngineBackend(options.engine, options.checkpoint, 16384, streams=4,
                                   conversation_cache_mib=options.cache_mib)
    try:
        if backend.prefill_rows != 4096: raise RuntimeError("expected the default 4096-row grid")
        peer_ids = prompt(4, 120)
        # Background readers drain long-running peers, so their output queues never pause them.
        readers = ThreadPoolExecutor(max_workers=2)
        peer = readers.submit(collect, backend.submit(peer_ids, request(peer_ids, 400, "bypass")))  # slot 0
        rows["a"] = run(backend, a, 16, "auto")  # slot 1, captures frontier 8192
        holder_ids = prompt(5, 64)
        holder = readers.submit(collect, backend.submit(holder_ids, request(holder_ids, 300, "bypass")))  # slot 1
        time.sleep(1.0)
        rows["b_hit"] = run(backend, b, 32, "auto")  # restores into slot 2: cross-slot
        rows["b_ref"] = run(backend, b, 32, "bypass")
        rows["b_sampled_hit"] = run(backend, b, 32, "auto", temperature=0.8, seed=11)
        rows["b_sampled_ref"] = run(backend, b, 32, "bypass", temperature=0.8, seed=11)
        rows["second"] = run(backend, second + prompt(13, 100), 8, "auto")
        rows["third"] = run(backend, third + prompt(14, 100), 8, "auto")  # evicts the base frontier
        rows["b_after_eviction"] = run(backend, b, 32, "auto")
        # G7: session A saves a shared system-and-tools frontier at 4096; session B shares only that block.
        system = prompt(6, 4500)
        session_a, session_b = system + prompt(15, 4000), system + prompt(16, 4000)
        rows["shared_seed"] = run(backend, session_a, 4, "auto", shared=4500)
        rows["shared_hit"] = run(backend, session_b, 32, "auto", shared=4500)
        rows["shared_ref"] = run(backend, session_b, 32, "bypass", shared=4500)
        rows["peer"] = dict(ids=peer.result(timeout=200)[0])
        rows["holder"] = dict(ids=holder.result(timeout=200)[0])
        readers.shutdown()
        rows["peer_alone"] = run(backend, peer_ids, 400, "bypass")
        rows["holder_alone"] = run(backend, holder_ids, 300, "bypass")
    finally:
        backend.close()
    checks = {
        "a_miss": rows["a"]["cached_tokens"] == 0,
        "b_hit_8192": rows["b_hit"]["cached_tokens"] == 8192 and rows["b_hit"]["cache_class"] == "session",
        "b_hit_equals_bypass": rows["b_hit"]["ids"] == rows["b_ref"]["ids"],
        "b_ref_no_cache": rows["b_ref"]["cached_tokens"] == 0,
        "sampled_hit_equals_bypass": rows["b_sampled_hit"]["ids"] == rows["b_sampled_ref"]["ids"]
                                     and rows["b_sampled_hit"]["cached_tokens"] == 8192,
        "eviction_miss_equals_bypass": rows["b_after_eviction"]["cached_tokens"] == 0
                                       and rows["b_after_eviction"]["ids"] == rows["b_ref"]["ids"],
        "shared_hit_4096": rows["shared_hit"]["cached_tokens"] == 4096 and rows["shared_hit"]["cache_class"] == "shared",
        "shared_hit_equals_bypass": rows["shared_hit"]["ids"] == rows["shared_ref"]["ids"],
        "peer_unchanged": rows["peer"]["ids"] == rows["peer_alone"]["ids"],
        "holder_unchanged": rows["holder"]["ids"] == rows["holder_alone"]["ids"],
    }
    report = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, rows=rows,
                  cache_mib=options.cache_mib, prefill_rows=4096)
    Path(options.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(status=report["status"], checks=checks)))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
