#!/usr/bin/env python3
"""Device check for superseded-first batch cache eviction.

One model load. Three sessions share an 8,300-token system block, so each saves or restores a Shared frontier
at 8192. Sessions A and B advance one turn each under a cache budget that holds the shared frontier and the two
newest session frontiers, but not the superseded ones. Session C then starts.

Candidate: each advancing capture evicts its own superseded frontier, so C restores 8192. Legacy LRU evicts the
shared frontier instead, so C misses. Every restored request must equal the same request under
cache_control=bypass. Run only under gpu-run with `timeout -k 10 300`. Engine stderr must go to --log.
"""
import argparse
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHARED = 8300
SUFFIX = 4100
TOKENS = 16
CAPTURE = re.compile(r"^cache_capture tokens=(\d+) class=(\w+) saved=(\d) evicted=(\d+) "
                     r"superseded_evicted=(\d+) cache_bytes=(\d+)$")
MARK = re.compile(r"^cache_policy_step (\w+)$")
# Turn order. Each name has its prompt parts: the shared block, then session suffixes.
STEPS = (("a1", "a", 1), ("b1", "b", 1), ("a2", "a", 2), ("b2", "b", 2), ("c1", "c", 1))
REFERENCES = ("b1", "a2", "b2", "c1")


def prompt(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def session_ids(session, turns):
    ids = prompt(1, SHARED)
    for turn in range(turns):
        ids += prompt(100 * (ord(session) - ord("a") + 1) + turn, SUFFIX)
    return ids


def parse_captures(lines):
    """Groups engine capture lines by the step marker that precedes them."""
    groups, step = {}, None
    for line in lines:
        line = line.strip()
        if marker := MARK.match(line):
            step = marker.group(1)
            groups.setdefault(step, [])
        elif (capture := CAPTURE.match(line)) and step:
            tokens, kind, saved, evicted, superseded, size = capture.groups()
            groups[step].append(dict(tokens=int(tokens), retention=kind, saved=saved == "1", evicted=int(evicted),
                                     superseded_evicted=int(superseded), cache_bytes=int(size)))
    return groups


def checks(rows, captures, candidate):
    """Expected hits for both policies; the candidate also needs its eviction trace and bypass equality."""
    cached = {name: rows[name]["cached_tokens"] for name, _, _ in STEPS}
    result = {
        "a1_miss": cached["a1"] == 0,
        "b1_shared_8192": cached["b1"] == 8192 and rows["b1"]["cache_class"] == "shared",
        "a2_own_12288": cached["a2"] == 12288 and rows["a2"]["cache_class"] == "session",
        "b2_own_12288": cached["b2"] == 12288 and rows["b2"]["cache_class"] == "session",
    }
    if not candidate:
        result["c1_legacy_miss"] = cached["c1"] == 0
        return result
    result["c1_shared_8192"] = cached["c1"] == 8192 and rows["c1"]["cache_class"] == "shared"
    for name in REFERENCES:
        result[f"{name}_equals_bypass"] = rows[name]["ids"] == rows[f"{name}_ref"]["ids"]
        result[f"{name}_ref_no_cache"] = rows[f"{name}_ref"]["cached_tokens"] == 0
    def trace(name):
        return [(c["tokens"], c["retention"], c["saved"], c["evicted"], c["superseded_evicted"])
                for c in captures.get(name, [])]
    result["a1_trace"] = trace("a1") == [(8192, "shared", True, 0, 0), (12288, "session", True, 0, 0)]
    result["b1_trace"] = trace("b1") == [(12288, "session", True, 0, 0)]
    # The cache is full here: each advance evicts only its own superseded 12288 frontier.
    result["a2_evicts_superseded"] = trace("a2") == [(16384, "session", True, 1, 1)]
    result["b2_evicts_superseded"] = trace("b2") == [(16384, "session", True, 1, 1)]
    # No superseded entry remains, so the new session's capture falls back to plain LRU.
    result["c1_evicts_lru"] = trace("c1") == [(12288, "session", True, 1, 0)]
    return result


def run(backend, collect, request, ids, cache):
    started = time.monotonic()
    job = backend.submit(ids, request(ids, cache), shared_prefix_tokens=SHARED)
    out, done = collect(job)
    keys = ("cached_tokens", "prefill_tokens", "cache_bytes", "restore_seconds", "capture_seconds", "cache_class")
    return dict(ids=out, seconds=time.monotonic() - started, **{key: done.get(key) for key in keys})


def main():
    from server.api import validate_request
    from server.batch_backend import BatchedEngineBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--log", required=True, help="file that receives this process's stderr")
    parser.add_argument("--cache-mib", type=int, default=1900)
    parser.add_argument("--legacy", action="store_true", help="baseline binary: no trace, no bypass references")
    options = parser.parse_args()

    def request(ids, cache):
        body = dict(prompt=list(ids), max_tokens=TOKENS, spec=0, cache_control=cache, temperature=0.0, seed=0)
        return validate_request(body, "completion", "crucible")

    def collect(job):
        output = []
        while True:
            event = job.events.get(timeout=200)
            if event["event"] == "error": raise RuntimeError(event)
            if event["event"] == "done": return output, event
            output.append(event["id"])

    rows = {}
    backend = BatchedEngineBackend(options.engine, options.checkpoint, 32768, streams=2,
                                   conversation_cache_mib=options.cache_mib)
    try:
        if backend.prefill_rows != 4096: raise RuntimeError("expected the default 4096-row grid")
        for name, session, turns in STEPS:
            print(f"cache_policy_step {name}", file=sys.stderr, flush=True)
            rows[name] = run(backend, collect, request, session_ids(session, turns), "auto")
        print("cache_policy_step references", file=sys.stderr, flush=True)
        if not options.legacy:
            for name in REFERENCES:
                _, session, turns = next(step for step in STEPS if step[0] == name)
                rows[f"{name}_ref"] = run(backend, collect, request, session_ids(session, turns), "bypass")
    finally:
        backend.close()
    sys.stderr.flush()
    captures = parse_captures(Path(options.log).read_text(errors="replace").splitlines())
    result = checks(rows, captures, not options.legacy)
    report = dict(status="PASS" if all(result.values()) else "FAIL", checks=result, rows=rows, captures=captures,
                  cache_mib=options.cache_mib, legacy=options.legacy, prefill_rows=4096, shared=SHARED)
    Path(options.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(status=report["status"], checks=result)))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
