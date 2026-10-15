"""Property test: random valid and invalid requests through the real HTTP server and both CPU fake engines.

Invariants for every request: the server answers in time, with no 500 for client input; an error is a JSON error
object with one crucible_error line; a stream is well-formed SSE that ends with [DONE]; a plain reply is a JSON
completion. After the run: no job, slot or per-client count is left, and no request text is in the log.
"""
import copy
import http.client
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from server.backend import EngineBackend
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for
from server.template import ChatTemplate

HERE = Path(__file__).parent
CONTEXT = 8192
CANARY = "FUZZCANARY8813"
SEEDS = int(os.environ.get("CRUCIBLE_FUZZ_REQUESTS", "240"))


class TemplateTokenizer:
    """The fixture chat template, one ID per UTF-8 byte. Its pieces decode in step with the IDs."""
    def __init__(self, root):
        self.template = ChatTemplate(root)

    def encode(self, request):
        if request.kind == "completion" and isinstance(request.prompt, tuple):
            return list(request.prompt)
        text = request.prompt if request.kind == "completion" else self.template.render(
            request.messages, request.enable_thinking, request.reasoning_effort, request.tools, request.tool_choice)
        return [1000 + byte for byte in text.encode("utf-8")]

    def stream(self):
        return None

    def step(self, stream, token):
        return chr(97 + token % 26)

    def decode(self, ids):
        return "".join(chr(97 + token % 26) for token in ids)


TOOL = {"type": "function", "function": {"name": "add", "description": "Add.", "parameters": {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
VALID = (
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": CANARY}], "max_tokens": 8,
     "stream": True, "stream_options": {"include_usage": True}},
    {"messages": [{"role": "user", "content": CANARY},
                  {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                   "function": {"name": "add", "arguments": "{\"a\":1,\"b\":2}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "3 " + CANARY}],
     "max_completion_tokens": 8, "tools": [TOOL], "tool_choice": "auto", "stream": True},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": CANARY}]}], "max_tokens": 6,
     "temperature": 0.6, "top_p": 0.95, "top_k": 20, "stop": ["zz"], "reasoning_effort": "high", "seed": 5},
    {"prompt": "Once " + CANARY, "max_tokens": 5, "stream": True},
)
ATOMS = (None, True, False, 0, 1, -1, 2 ** 70, 0.5, 1e308, float("nan"), "x", "", [], {}, [1], {"a": 1}, "auto",
         "none", "user", "tool", "function", CANARY)
FIELDS = ("model", "messages", "prompt", "max_tokens", "temperature", "n", "stream", "stream_options", "spec",
          "enable_thinking", "reasoning_effort", "chat_template_kwargs", "thinking_budget", "return_token_ids",
          "top_k", "top_p", "min_p", "seed", "cache_control", "tools", "tool_choice", "stop", "unknown_" + CANARY)


def atom(rng, depth=0):
    if depth < 3 and rng.random() < 0.3:
        if rng.random() < 0.5:
            return [atom(rng, depth + 1) for _ in range(rng.randint(0, 3))]
        return {rng.choice(("type", "text", "role", "content", "name", "arguments", "id")): atom(rng, depth + 1)
                for _ in range(rng.randint(0, 3))}
    return rng.choice(ATOMS)


def mutate(rng, value):
    if isinstance(value, dict) and value and rng.random() < 0.7:
        key, value = rng.choice(list(value)), dict(value)
        roll = rng.random()
        if roll < 0.15:
            del value[key]
        elif roll < 0.3:
            value[rng.choice(FIELDS)] = atom(rng)
        else:
            value[key] = mutate(rng, value[key])
        return value
    if isinstance(value, list) and value and rng.random() < 0.7:
        value, index = list(value), rng.randrange(len(value))
        roll = rng.random()
        if roll < 0.15:
            del value[index]
        elif roll < 0.3:
            value.insert(index, copy.deepcopy(value[index]))
        else:
            value[index] = mutate(rng, value[index])
        return value
    return atom(rng)


def case(seed, batch):
    """(path, body bytes, disconnect after the first SSE event)."""
    rng = random.Random(seed)
    body = copy.deepcopy(rng.choice(VALID))
    for _ in range(rng.choice((0, 0, 1, 1, 2, 3))):
        body = mutate(rng, body)
    if batch and isinstance(body, dict):
        body.setdefault("spec", 0)
    path = "/v1/completions" if isinstance(body, dict) and "prompt" in body else "/v1/chat/completions"
    if rng.random() < 0.05:
        path = rng.choice(("/v1/completions", "/v1/chat/completions", "/v1/" + CANARY))
    raw = json.dumps(body, allow_nan=True).encode("utf-8", "surrogatepass")
    if rng.random() < 0.05:
        raw = raw[:rng.randrange(len(raw) + 1)]  # cut JSON
    return path, raw, rng.random() < 0.15


# Requests that BaseHTTPRequestHandler refuses before a method handler runs (send_error), and one upload above the
# body limit (discard_body): (request bytes, the status). A request line that the parser reads as HTTP/0.9 gets no
# status line, only the body.
LARGE = 24 << 20  # above the loopback socket buffers, so an unread body resets the connection


def raw_cases(port):
    host = b"Host: 127.0.0.1:%d\r\n" % port
    return [
        (b"PUT /v1/models HTTP/1.1\r\n" + host + b"\r\n", 501),
        (b"DELETE /v1/" + CANARY.encode() + b" HTTP/1.1\r\n" + host + b"\r\n", 501),
        (b"GARBAGE " + CANARY.encode() + b"\r\n\r\n", 400),
        (b"GET /v1/models HTTP/1.1\r\n" + b"X-A: b\r\n" * 120 + b"\r\n", 431),
        (b"GET /v1/models HTTP/1.1\r\nX-A: " + b"b" * 70000 + b"\r\n\r\n", 431),
        (b"GET /v1/models HTTP/2.0\r\n" + host + b"\r\n", 505),
        (b"POST /v1/chat/completions HTTP/1.1\r\n" + host + b"Content-Type: application/json\r\n"
         b"Content-Length: %d\r\n\r\n" % LARGE + b"x" * LARGE, 413),
    ]


def send_raw(port, data, expected):
    client = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        client.sendall(data)  # a reset here (EPIPE) fails the case: the client never sees the error
        reply = b""
        while chunk := client.recv(65536):
            reply += chunk
    finally:
        client.close()
    head, separator, body = reply.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1]) if reply.startswith(b"HTTP/1.") else expected
    assert status == expected, reply[:200]
    error = json.loads(body if separator else reply)["error"]
    assert error["message"] and error["type"] == ("invalid_request_error" if status < 500 else "server_error")
    return status


def check_reply(status, headers, data):
    """One reply against the invariants. Returns the status for the error-line count."""
    assert status in (200, 400, 404, 413, 429, 503), (status, data[:200])
    if status != 200:
        error = json.loads(data)["error"]
        assert isinstance(error["message"], str) and error["message"]
        assert error["type"] == ("invalid_request_error" if status < 500 else "server_error")
        return status
    if headers.get("Content-Type") == "text/event-stream":
        assert data.endswith(b"data: [DONE]\n\n"), data[-200:]
        events = [part[6:] for part in data.decode().split("\n\n") if part.startswith("data: ")]
        payloads = [json.loads(event) for event in events[:-1]]
        ends = [choice["finish_reason"] for payload in payloads for choice in payload.get("choices", [])
                if choice.get("finish_reason")]
        errors = [payload for payload in payloads if "error" in payload]
        assert len(ends) + len(errors) == 1, payloads[-3:]
        return 200 if ends else "stream_error"
    reply = json.loads(data)
    assert reply["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls")
    assert reply["usage"]["total_tokens"] == reply["usage"]["prompt_tokens"] + reply["usage"]["completion_tokens"]
    return 200


def send(port, path, raw, disconnect):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.request("POST", path, raw, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if disconnect and response.getheader("Content-Type") == "text/event-stream":
            response.fp.readline()
            return None  # the close below cancels the request mid-stream
        return check_reply(response.status, dict(response.getheaders()), response.read())
    finally:
        connection.close()


def start_backend(batch):
    popen = subprocess.Popen
    fake = HERE / ("fake_batch_engine.py" if batch else "fake_engine.py")
    def start(*args, **kwargs):
        env = dict(kwargs.pop("env", os.environ), FAKE_CONTEXT=str(CONTEXT), FAKE_DELAY="0.001",
                   FAKE_SLOT_CONTEXTS=f"{CONTEXT},{CONTEXT}")
        return popen([sys.executable, str(fake)], env=env, **kwargs)
    with patch("server.backend.subprocess.Popen", side_effect=start):
        if batch:
            return BatchedEngineBackend("unused", "unused", context=CONTEXT, streams=2, output_timeout=5)
        return EngineBackend("unused", "unused", context=CONTEXT)


def leftovers(backend, service):
    counts = {name: len(getattr(backend, name)) for name in ("owned_jobs", "active", "waiting") if hasattr(backend, name)}
    return {name: count for name, count in counts.items() if count} | (
        dict(client_requests=service.client_requests) if service.client_requests else {})


@pytest.mark.parametrize("batch", [False, True], ids=["serial", "batch"])
def test_random_requests_keep_the_server_invariants(batch, capsys):
    backend = start_backend(batch)
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "tokenizer_config.json").write_text(json.dumps(
            {"chat_template": (HERE / "fixtures/qwen_chat_template.jinja").read_text()}))
        service = Service(backend, TemplateTokenizer(root), max_requests_per_client=64)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        cases = [(send, case(seed, batch)) for seed in range(SEEDS)]
        for index, item in enumerate(raw_cases(port)):
            cases.insert(index * len(cases) // 7, (send_raw, item))
        started = time.monotonic()
        with ThreadPoolExecutor(8) as pool:
            statuses = list(pool.map(lambda item: item[0](port, *item[1]), cases))
        assert time.monotonic() - started < 120
        deadline = time.monotonic() + 15
        while leftovers(backend, service) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert leftovers(backend, service) == {}
        # The server still answers after the run.
        assert send(port, "/v1/chat/completions", json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "spec": 0}).encode(), False) == 200
    finally:
        server.shutdown()
        server.server_close()
        backend.close()
    err = capsys.readouterr().err
    assert CANARY not in err and "Traceback" not in err
    lines = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_error ")]
    refused = sorted(status for status in statuses if status not in (None, 200, "stream_error"))
    assert sorted(line["status"] for line in lines if not line["streamed"]) == refused
    assert len([line for line in lines if line["streamed"] and line["status"] != 200]) == statuses.count("stream_error")
    assert statuses.count(200) > SEEDS // 10  # the generator still reaches the engine
