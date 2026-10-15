"""Run under gpu-run. Test seeded gen/API equality with two bounded model loads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import threading
import time

from server.api import STOP_IDS, validate_request
from server.backend import EngineBackend, Job
from server.http import Service, handler_for
from server.tokenization import ModelTokenizer

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8790"


def stop_prefix(ids):
    for index, token in enumerate(ids):
        if token in STOP_IDS:
            return ids[:index + 1]
    return ids


def curl_plain(out, name, body, endpoint="/v1/chat/completions"):
    payload = out / (name + ".request.json")
    payload.write_text(json.dumps(body))
    raw = subprocess.check_output(["curl", "-sS", "--fail-with-body", "--max-time", "90",
                                   "-H", "Content-Type: application/json", "--data-binary", "@" + str(payload),
                                   URL + endpoint])
    (out / (name + ".response.json")).write_bytes(raw)
    return json.loads(raw)


def curl_stream(out, name, body, endpoint="/v1/chat/completions"):
    payload = out / (name + ".request.json")
    payload.write_text(json.dumps(body))
    begin = time.monotonic()
    process = subprocess.Popen(["curl", "-sS", "-N", "--fail-with-body", "--max-time", "90",
                                "-H", "Content-Type: application/json", "--data-binary", "@" + str(payload),
                                URL + endpoint], stdout=subprocess.PIPE, text=True)
    raw, ids, text, times, usage, done = [], [], [], [], None, False
    for line in process.stdout:
        raw.append(line)
        if line.strip() == "data: [DONE]":
            done = True
        elif line.startswith("data: {"):
            event = json.loads(line[6:])
            assert "error" not in event, event
            usage = event.get("usage", usage)
            for choice in event["choices"]:
                if "token_id" in choice:
                    ids.append(choice["token_id"])
                    times.append(time.monotonic() - begin)
                text.append(choice.get("delta", {}).get("content", choice.get("text", "")))
    assert process.wait(timeout=5) == 0 and done and ids and usage
    (out / (name + ".sse")).write_text("".join(raw))
    return dict(ids=ids, text="".join(text), usage=usage, first_token_seconds=times[0],
                decode_tokens_per_second=(len(ids) - 1) / (times[-1] - times[0]) if len(ids) > 1 else 0,
                total_seconds=time.monotonic() - begin)


def malformed_protocol(backend, ids):
    for extra in (dict(seed=-1), dict(seed=1 << 64), dict(temperature=1e309), dict(top_p=1.1)):
        job = Job(dict(ids=ids, max_tokens=64) | extra)
        backend.jobs.put(job, timeout=5)
        event = job.events.get(timeout=30)
        assert event["event"] == "error", event
        assert backend.fatal is None
    print("PASS: malformed sampling fields recover in the resident engine", flush=True)


def check_http(out, body, expected, ids):
    streamed = curl_stream(out, "sample-stream", body | dict(stream=True, stream_options=dict(include_usage=True)))
    plain = curl_plain(out, "sample-plain", body)
    assert streamed["ids"] == plain["choices"][0]["token_ids"] == expected
    assert streamed["text"] == plain["choices"][0]["message"]["content"]
    assert streamed["usage"] == plain["usage"]
    assert plain["usage"]["prompt_tokens"] == len(ids)
    print("PASS: seeded streamed/non-streamed chat IDs equal gen", flush=True)
    return streamed


def check_completions(out, body, expected, ids):
    completion = {key: value for key, value in body.items() if key != "messages"} | dict(prompt=ids)
    for spec in (0, 2):
        plain = curl_plain(out, f"completion-{spec}", completion | dict(spec=spec), "/v1/completions")
        stream = curl_stream(out, f"completion-stream-{spec}", completion | dict(spec=spec, stream=True,
                              stream_options=dict(include_usage=True)), "/v1/completions")
        assert stream["ids"] == plain["choices"][0]["token_ids"]
        if spec == 2:
            assert stream["ids"] == expected
    print("PASS: sampled serial and speculative completions repeat their own seeded output", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/engine/crucible")
    args = parser.parse_args()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0, "Port 8790 already has a listener"
    out = ROOT / "results" / ("sampling-live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    tokenizer = ModelTokenizer(args.checkpoint)
    body = dict(model="crucible", messages=[dict(role="user", content="Write a Python function for the nth Fibonacci number.")],
                max_tokens=64, spec=2, temperature=0.7, top_k=50, top_p=0.95, min_p=0.01,
                seed=(1 << 64) - 1, return_token_ids=True)
    ids = tokenizer.encode(validate_request(body, "chat", "crucible"))
    ids_file = out / "chat.ids"
    ids_file.write_text(",".join(map(str, ids)))
    with (out / "gen.log").open("w") as log:
        subprocess.run([str(args.engine), "gen", str(args.checkpoint), "--ids-file", str(ids_file),
                        "--n", "64", "--spec", "2", "--temperature", "0.7", "--top-k", "50",
                        "--top-p", "0.95", "--min-p", "0.01", "--seed", str(body["seed"])],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    line = next(line for line in (out / "gen.log").read_text().splitlines() if line.startswith("ids:"))
    expected = stop_prefix([int(token) for token in line[4:].split()])
    begin = time.monotonic()
    backend = EngineBackend(args.engine, args.checkpoint)
    readiness = time.monotonic() - begin
    server = ThreadingHTTPServer(("127.0.0.1", 8790), handler_for(Service(backend, tokenizer)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        streamed = check_http(out, body, expected, ids)
        malformed_protocol(backend, ids)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda name: curl_plain(out, name, body), ("queued-a", "queued-b")))
        assert all(item["choices"][0]["token_ids"] == expected for item in results)
        stopping = body | dict(messages=[dict(role="user", content="Reply with only OK.")], temperature=0)
        stop = curl_plain(out, "greedy-stop", stopping)
        assert stop["choices"][0]["token_ids"][-1] in STOP_IDS
        after = curl_plain(out, "after-short-prompt", body)
        assert after["choices"][0]["token_ids"] == expected
        check_completions(out, body, expected, ids)
        summary = dict(output=str(out), model_ready_seconds=readiness, prompt_tokens=len(ids),
                       stream={key: value for key, value in streamed.items() if key not in ("ids", "text")})
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()
    assert backend.process.poll() is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0
    print("PASS: sampled API server and resident engine stopped", flush=True)


if __name__ == "__main__":
    main()
