"""Run under gpu-run. Check resident defaults against the named main gen policy."""
import argparse
from datetime import datetime
from http.server import ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import socket
from statistics import median
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server.api import STOP_IDS, validate_request
from server.backend import EngineBackend
from server.http import Service, handler_for
from server.tokenization import ModelTokenizer

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8790/v1/completions"
# Logical checkpoint bytes per token, not measured DRAM traffic.
SERIAL_WEIGHT_BYTES = {"3.05": 4_615_344_324, "4.05": 5_355_901_444}
DRAM_BYTES_PER_SECOND = 245.5e9


def stopped(ids):
    for index, token in enumerate(ids):
        if token in STOP_IDS:
            return ids[:index + 1]
    return ids


def gen_reference(args, out, ids, temperature):
    path = out / "prompt.ids"
    path.write_text(",".join(map(str, ids)))
    command = [str(args.baseline), "gen", str(args.checkpoint), "--ids-file", str(path), "--n", "64",
               "--temperature", str(temperature), "--top-k", "50", "--top-p", "0.95", "--seed", "9123"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
                            timeout=120, text=True)
    (out / ("gen-" + str(temperature) + ".log")).write_text(result.stdout)
    line = next(line for line in result.stdout.splitlines() if line.startswith("ids:"))
    return stopped([int(value) for value in line[4:].split()])


def post(out, name, body):
    request = Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        raw = response.read()
    (out / (name + ".response")).write_bytes(raw)
    if not body.get("stream"):
        return json.loads(raw)["choices"][0]["token_ids"]
    events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
    assert raw.endswith(b"data: [DONE]\n\n") and not any("error" in event for event in events), events
    return [choice["token_id"] for event in events for choice in event["choices"] if "token_id" in choice]


def measure(backend, ids, body):
    job = backend.submit(ids, validate_request(body, "completion", "crucible"))
    tokens = []
    while True:
        event = job.events.get(timeout=90)
        assert event["event"] != "error", event
        if event["event"] == "done":
            assert event["completion_tokens"] == len(tokens), event
            return dict(ids=tokens, prefill_seconds=event["prefill_seconds"], decode_seconds=event["decode_seconds"],
                        tokens_per_second=(len(tokens) - 1) / event["decode_seconds"])
        tokens.append(event["id"])


def summarize_policy(runs):
    if not runs or any(run["ids"] != runs[0]["ids"] for run in runs):
        raise ValueError("policy output IDs differ or timing samples are missing")
    if any(not math.isfinite(run[field]) or run[field] <= 0 for run in runs
           for field in ("prefill_seconds", "decode_seconds", "tokens_per_second")):
        raise ValueError("timing samples must have finite positive values")
    rates = [run["tokens_per_second"] for run in runs]
    return dict(ids=list(runs[0]["ids"]), runs=[dict(run, ids=list(run["ids"])) for run in runs],
                median_prefill_seconds=median(run["prefill_seconds"] for run in runs),
                median_decode_seconds=median(run["decode_seconds"] for run in runs),
                tokens_per_second=median(rates), min_tokens_per_second=min(rates), max_tokens_per_second=max(rates))


def roofline_summary(tier, timings):
    serial_ceiling = DRAM_BYTES_PER_SECOND / SERIAL_WEIGHT_BYTES[tier]
    rate = timings["default"]["tokens_per_second"]
    return dict(serial_logical_weight_bytes_per_token=SERIAL_WEIGHT_BYTES[tier],
                serial_roofline_tokens_per_second=serial_ceiling,
                serial_roofline_percent=100 * timings["serial"]["tokens_per_second"] / serial_ceiling,
                serial_roofline_note="Caches, activations, and extra physical reads are not included.",
                estimated_code_roofline_tokens_per_second=[150, 200] if tier == "3.05" else None,
                default_code_estimate_percent=[100 * rate / 200, 100 * rate / 150] if tier == "3.05" else None,
                code_roofline_note="The speed-tier range is a code estimate, not a hardware ceiling." if tier == "3.05"
                else "No validated code roofline is available for this adaptive policy at 4.05 bpw.")


def check_api(out, body, references):
    for temperature, expected in references.items():
        request = body | dict(temperature=temperature)
        assert post(out, "plain-" + str(temperature), request) == expected
        assert post(out, "stream-" + str(temperature), request | dict(stream=True)) == expected
        assert post(out, "single-token-" + str(temperature), request | dict(max_tokens=1)) == expected[:1]
        assert post(out, "single-stream-" + str(temperature),
                    request | dict(max_tokens=1, stream=True)) == expected[:1]
    assert post(out, "serial-off", body | dict(spec=0, draft_q=0, draft_head="full", mtp_prefix=0,
                                               draft_adapt=0, temperature=0)) == references[0]
    assert post(out, "sample-topk1", body | dict(temperature=0.7, top_k=1)) == references[0]
    assert post(out, "full-warmup", body | dict(mtp_prefix=0)) == references[0]
    assert post(out, "fixed-sparse-head", body | dict(draft_adapt=0)) == references[0]
    try:
        post(out, "invalid-policy", body | dict(mtp_prefix=-1))
        raise AssertionError("invalid policy was accepted")
    except HTTPError as error:
        assert error.code == 400
    assert post(out, "after-error", body) == references[0]


def measure_policies(backend, short, body):
    policies = (("serial", dict(spec=0, draft_head="full", draft_q=0, mtp_prefix=0)),
                ("old-spec2", dict(spec=2, draft_head="full", draft_q=0, mtp_prefix=0)), ("default", {}))
    samples = {name: [] for name, _ in policies}
    for repeat in range(3):
        for name, options in policies[repeat:] + policies[:repeat]:
            sample = measure(backend, short, body | dict(prompt=short, max_tokens=128) | options)
            samples = {**samples, name: [*samples[name], sample]}
            assert sample["ids"] == samples["serial"][0]["ids"]
    return {name: summarize_policy(runs) for name, runs in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tier", choices=SERIAL_WEIGHT_BYTES, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/engine/candidate")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0, "Port 8790 already has a listener"
    out = ROOT / "results" / ("defaults-live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), tier=args.tier,
                    engine=str(args.engine.resolve()), baseline=str(args.baseline.resolve()),
                    engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest(),
                    baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tokenizer = ModelTokenizer(args.checkpoint)
    short = tokenizer.encode(validate_request(dict(prompt="Write a Python function for the nth Fibonacci number."),
                                              "completion", "crucible"))
    ids = short * (320 // len(short) + 1)
    references = {temperature: gen_reference(args, out, ids, temperature) for temperature in (0, 0.7)}
    body = dict(prompt=ids, max_tokens=64, top_k=50, top_p=0.95, seed=9123, return_token_ids=True)
    backend = EngineBackend(args.engine, args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", 8790), handler_for(Service(backend, tokenizer)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        check_api(out, body, references)
        timings = measure_policies(backend, short, body)
        summary = dict(metadata=metadata, prompt_tokens=len(ids), timings=timings,
                       **roofline_summary(args.tier, timings))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in summary.items() if key != "timings"}, indent=2), flush=True)
        for name, timing in timings.items():
            print(name, timing["tokens_per_second"], "tok/s", flush=True)
        print("PASS: greedy and sampled defaults equal named main gen; streaming, off switches, cap, and recovery", flush=True)
        print("results: " + str(out), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()
    assert backend.process.poll() is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0
    print("PASS: temporary defaults server and resident engine stopped", flush=True)


if __name__ == "__main__": main()
