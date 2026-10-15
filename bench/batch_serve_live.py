#!/usr/bin/env python3
"""Temporary port8790 acceptance. Use refenv Python for jinja2 and tokenizers.
Run only under gpu-run and a 300-second timeout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen
from unittest.mock import patch
import queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.api import validate_request
from server.backend import EngineBackend, Job
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for
from server.tokenization import ModelTokenizer

URL = "http://127.0.0.1:8790/v1/completions"


def complete(payload):
    start = time.monotonic()
    request = Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response: result = json.load(response)
    ids = result["choices"][0]["token_ids"]
    if result["usage"]["completion_tokens"] != len(ids): raise ValueError("wrong HTTP token count")
    return ids, time.monotonic() - start


def collect(job):
    output, generation = [], None
    while True:
        event = job.events.get(timeout=180)
        if event["event"] == "error": raise ValueError(event)
        if event["event"] == "done":
            if event["completion_tokens"] != len(output): raise ValueError("wrong backend token count")
            return output, generation
        if generation is None: generation = event["slot_generation"]
        if generation != event["slot_generation"]: raise ValueError("recycled request changed its generation")
        output.append(event["id"])


def paused_lease(backend, job):
    if job.cancelled.is_set(): raise ValueError("the slow request stopped before the cancellation test")
    with job.events.mutex: events = list(job.events.queue)
    if job.events.maxsize != 4 or len(events) != 2 or any(event.get("event") != "token" for event in events):
        raise ValueError("the slow request did not retain only its two paused tokens")
    leases = [(id, entry) for id, entry in list(backend.active.items()) if entry.job is job]
    if len(leases) != 1 or not leases[0][1].ready or leases[0][1].count != 2:
        raise ValueError("the paused request is no longer active")
    return leases[0][0], leases[0][1].generation


def wait_closed(backend):
    deadline = time.monotonic() + 30
    while backend.active and time.monotonic() < deadline:
        if backend.fatal: raise ValueError(backend.fatal)
        time.sleep(.01)
    if backend.active or not backend.jobs.empty(): raise ValueError("the cancellation barrier did not drain the leases")


def listener(backend, tokenizer):
    server = ThreadingHTTPServer(("127.0.0.1", 8790), handler_for(Service(backend, tokenizer)))
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


def batch_checks(backend, requests, raw_ids):
    serial = [complete(request)[0] for request in requests]
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool: concurrent = list(pool.map(complete, requests))
    elapsed = time.monotonic() - start
    if [value[0] for value in concurrent] != serial: raise ValueError("concurrent HTTP IDs differ from independent requests")
    selected = next(row for row, ids in enumerate(serial) if len(ids) >= 3)
    options = validate_request(requests[selected] | dict(max_tokens=80), "completion", "crucible")
    # A smaller valid queue forces backpressure before a short answer can hit EOS.
    with patch("server.backend.Job", side_effect=lambda payload: Job(payload, events=queue.Queue(maxsize=4))):
        slow = backend.submit(raw_ids[selected], options)
    deadline = time.monotonic() + 60
    while slow.events.qsize() < 2 and time.monotonic() < deadline: time.sleep(.01)
    if slow.events.qsize() != 2: raise ValueError("slow consumer never reached its reserved output boundary")
    lease = paused_lease(backend, slow)
    peer = complete(requests[0])[0]
    if peer != serial[0]: raise ValueError("paused peer changed HTTP IDs")
    if paused_lease(backend, slow) != lease: raise ValueError("the paused lease changed during the peer request")
    slow.cancelled.set()
    wait_closed(backend)
    replacement = validate_request(requests[selected], "completion", "crucible")
    recycled, generation = collect(backend.submit(raw_ids[selected], replacement))
    if generation != lease[1] + 1: raise ValueError("the cancelled sequence slot was not recycled")
    if recycled != serial[selected]: raise ValueError("cancelled slot changed the next request")
    if backend.fatal: raise ValueError(backend.fatal)
    return dict(ids=serial, concurrent_seconds=elapsed,
                http_rate=sum(len(value[0]) for value in concurrent)/elapsed,
                request_seconds=[value[1] for value in concurrent], backpressure_and_cancel="PASS",
                slow_output_capacity=4, paused_generation=lease[1], recycled_generation=generation)


def read_prompt_ids(path):
    parts = Path(path).read_text().split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"Prompt file has an empty ID field: {path}")
    values = [value for part in parts for value in part.split()]
    if any(not value.isascii() or not value.isdecimal() for value in values):
        raise ValueError(f"Prompt IDs must use ASCII decimal digits: {path}")
    return [int(value) for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Use a new output file.")
    paths = args.prompts.read_text().splitlines()
    if len(paths) != 8: parser.error("Use exactly eight prompt paths.")
    raw = [read_prompt_ids(path) for path in paths]
    tokenizer = ModelTokenizer(args.checkpoint)
    requests = [dict(prompt=ids, spec=0, max_tokens=8, return_token_ids=True,
                     temperature=.7 if row % 2 else 0, top_k=50, seed=40+row) for row, ids in enumerate(raw)]
    report = dict(status="FAIL", context=1024, streams=8)
    try:
        with_backend = BatchedEngineBackend(args.engine, args.checkpoint, 1024, streams=8)
        server, worker = None, None
        try:
            server, worker = listener(with_backend, tokenizer)
            report["batch"] = batch_checks(with_backend, requests, raw)
        finally:
            if server is not None: server.shutdown(); server.server_close(); worker.join(timeout=5)
            with_backend.close()
        serial_backend = EngineBackend(args.engine, args.checkpoint, 1024)
        server, worker = None, None
        try:
            server, worker = listener(serial_backend, tokenizer)
            independent = [complete(request)[0] for request in requests]
            if independent != report["batch"]["ids"]: raise ValueError("batch IDs differ from ordinary independent serving")
            report["independent_serial"] = "PASS"
        finally:
            if server is not None: server.shutdown(); server.server_close(); worker.join(timeout=5)
            serial_backend.close()
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__": main()
