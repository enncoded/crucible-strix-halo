#!/usr/bin/env python3
"""Lookup ABAB inside one serve engine process (train 16 lookup default).

EngineBackend.submit reads backend.lookup for each request, so lookup on (7) and off (0) alternate in one process
without a reload. Mode `serial` runs serial serve; mode `batched` runs batch serve with 4 streams. SESSIONS lists
the concurrent sessions of each phase (for example 1,4). TEMPS lists greedy (0) and sampled (above 0, fixed seed)
cases; keyed sampling makes a sampled output independent of the drafts too. Every output must equal the first
output of its prompt and temperature.
Batched mode sends cache_prefix_tokens and stops when a timed turn misses the prompt cache.
Usage: ab.py serial|batched ENGINE CHECKPOINT PROMPT_DIR OUT.jsonl PAIRS SESSIONS TEMPS
"""
import hashlib
import json
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server.api import validate_request  # noqa: E402
from server.backend import EngineBackend  # noqa: E402
from server.batch_backend import BatchedEngineBackend  # noqa: E402

TOKENS, CTX, CACHE_MIB, ON = 512, 32768, 8192, 7
SAMPLING = dict(top_k=20, top_p=0.95, seed=1234)


def cache_prefix_tokens(ids, request):
    """The prompt prefix that the engine may restore: it stops where the draft warmup starts (prompt - mtp_prefix).

    Without it the batched engine saves only the last grid point of the prompt, which no later turn can restore,
    and every turn prefills the whole prompt again.
    """
    return len(ids) - request.mtp_prefix if len(ids) > request.mtp_prefix else 0


def require_cached(results, kind):
    """A timed round that misses the prompt cache would run at the prefill speed, not the decode speed."""
    if kind != "warm" and any(not r["cached_tokens"] for r in results):
        raise RuntimeError("a timed turn missed the prompt cache; stopping before the step time limit")


def run(backend, ids, request, result, cache_prefix=True):
    job = backend.submit(ids, request, cache_prefix_tokens=cache_prefix_tokens(ids, request) if cache_prefix else 0)
    out, first, last, cached = [], None, None, 0
    while True:
        event = job.events.get(timeout=600)
        now = time.monotonic()
        if event["event"] == "token":
            first = now if first is None else first
            last = now
            out.append(event["id"])
        elif event["event"] == "done":
            cached = event.get("cached_tokens", 0)
            break
        else:
            raise RuntimeError(f"engine error: {event}")
    result.append({"ids": out, "decode_s": last - first, "tok_s": (len(out) - 1) / (last - first),
                   "cached_tokens": cached})


def round_of(backend, ids, request, sessions, cache_prefix=True):
    results = []
    threads = [threading.Thread(target=run, args=(backend, ids, request, results, cache_prefix))
               for _ in range(sessions)]
    start = time.monotonic()
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    wall = time.monotonic() - start
    if len(results) != sessions: raise RuntimeError("a session failed")
    return results, wall


def main():
    mode, engine, checkpoint, prompts, output, pairs, sessions_list, temps = sys.argv[1:9]
    backend = (EngineBackend(engine, checkpoint, CTX, CACHE_MIB, lookup=ON) if mode == "serial" else
               BatchedEngineBackend(engine, checkpoint, CTX, streams=4, conversation_cache_mib=CACHE_MIB, lookup=ON))
    requests = {float(t): validate_request(dict(prompt="x", max_tokens=TOKENS, temperature=float(t)) |
                                           (SAMPLING if float(t) > 0 else {}), "completion", "crucible")
                for t in temps.split(",")}
    texts = {name: [int(x) for x in (Path(prompts) / f"{name}.txt").read_text().split(",")] for name in ("copy", "summary")}
    reference, failures = {}, 0
    plan = [(1, "warm")] + [(int(n), "ab") for n in sessions_list.split(",")]
    # closing(): an error or a missed cache stops the engine process too, so it does not keep the GPU.
    with closing(backend), open(output, "a") as log:
        for sessions, kind in plan:
            for pair in range(1 if kind == "warm" else int(pairs)):
                # ABBA order across pairs, so a slow drift of the clock does not favor one arm.
                order = (ON, 0) if pair % 2 == 0 else (0, ON)
                for lookup in ((0,) if kind == "warm" else order):
                    for (temperature, request), (name, ids) in ((r, t) for r in requests.items() for t in texts.items()):
                        backend.lookup = lookup
                        # Serial serve keeps its night2 setting (no boundary); the batched engine needs the prefix.
                        results, wall = round_of(backend, ids, request, sessions, mode == "batched")
                        if mode == "batched": require_cached(results, kind)
                        key = f"{name}@{temperature}"
                        for item in results:
                            reference.setdefault(key, item["ids"])
                            failures += item["ids"] != reference[key]
                        row = {"mode": mode, "sessions": sessions, "kind": kind, "pair": pair, "lookup": lookup,
                               "prompt": name, "temperature": temperature, "wall_s": wall,
                               "tok_s": [r["tok_s"] for r in results], "cached_tokens": [r["cached_tokens"] for r in results],
                               "aggregate_tok_s": sum(len(r["ids"]) for r in results) / wall,
                               "tokens": [len(r["ids"]) for r in results],
                               "ids_sha256": hashlib.sha256(json.dumps(reference[key]).encode()).hexdigest(),
                               "identical": all(r["ids"] == reference[key] for r in results)}
                        log.write(json.dumps(row) + "\n"); log.flush()
    print(f"{mode}: {failures} outputs differ from the reference")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
