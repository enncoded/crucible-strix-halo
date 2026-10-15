"""Headers at admission, SSE keepalives, and cancellation of a request whose client goes away before output."""
import json
import os
import queue
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from server import http
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for
from server.tests.serve_fixtures import PiecesTokenizer, chat, chunks, invoke, token_events

FAKE = Path(__file__).parent / "fake_batch_engine.py"


class SilentBackend:
    """A job that gets its events only when the test releases them."""
    def __init__(self):
        self.jobs, self.trace = [], None

    def submit(self, ids, request, **options):
        job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
        self.jobs.append(job)
        return job


def closed_peer():
    server, client = socket.socketpair()
    client.close()
    return server


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(http, "POLL_SECONDS", 0.01)


@pytest.mark.parametrize("stream", [False, True])
def test_disconnect_before_output_cancels_the_job(stream, fast_polls, capsys):
    backend = SilentBackend()
    connection = closed_peer()
    try:
        response = invoke(backend, PiecesTokenizer(["x"]), chat(stream=stream), connection=connection)
    finally:
        connection.close()
    assert backend.jobs[0].cancelled.is_set()
    assert response.handler.close_connection
    # A stream sent its headers at admission; a plain request sends nothing to a closed client.
    assert response.statuses == ([200] if stream else [])
    assert b"data:" not in response.body
    line = [line for line in capsys.readouterr().err.splitlines() if line.startswith("crucible_cancel ")]
    assert json.loads(line[0].split(" ", 1)[1]) == dict(request_id=None, streamed=stream, completion_tokens=0)


def test_connected_client_keeps_waiting(fast_polls):
    backend = SilentBackend()
    server, client = socket.socketpair()
    timer = threading.Timer(0.2, lambda: [backend.jobs[0].events.put(event) for event in token_events(["hi"])])
    timer.start()
    try:
        response = invoke(backend, PiecesTokenizer(["hi"]), chat(stream=True), connection=server)
    finally:
        timer.cancel(); server.close(); client.close()
    assert response.statuses == [200]
    assert "".join(choice["delta"].get("content", "") for event in chunks(response)
                   for choice in event["choices"]) == "hi"


def test_keepalive_comments_while_waiting(fast_polls, monkeypatch):
    monkeypatch.setattr(http, "KEEPALIVE_SECONDS", 0.03)
    backend = SilentBackend()
    timer = threading.Timer(0.2, lambda: [backend.jobs[0].events.put(event) for event in token_events(["hi"])])
    timer.start()
    try:
        response = invoke(backend, PiecesTokenizer(["hi"]), chat(stream=True))
    finally:
        timer.cancel()
    head, _, rest = response.body.partition(b"data:")
    assert head.count(b": keepalive\n\n") >= 2 and head == b": keepalive\n\n" * head.count(b": keepalive\n\n")
    assert response.body.endswith(b"data: [DONE]\n\n")


def test_plain_requests_get_no_keepalive(fast_polls, monkeypatch):
    monkeypatch.setattr(http, "KEEPALIVE_SECONDS", 0.01)
    backend = SilentBackend()
    timer = threading.Timer(0.1, lambda: [backend.jobs[0].events.put(event) for event in token_events(["hi"])])
    timer.start()
    try:
        response = invoke(backend, PiecesTokenizer(["hi"]), chat())
    finally:
        timer.cancel()
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "hi"


@pytest.mark.parametrize("stream", [False, True])
def test_first_event_timeout(stream, fast_polls, monkeypatch):
    monkeypatch.setattr(http, "FIRST_EVENT_SECONDS", 0.05)
    backend = SilentBackend()
    response = invoke(backend, PiecesTokenizer(["x"]), chat(stream=stream))
    assert response.statuses == ([200] if stream else [503])
    assert b"The queued request timed out" in response.body
    assert backend.jobs[0].cancelled.is_set()
    if stream: assert response.body.endswith(b"data: [DONE]\n\n")


def test_first_event_limit_is_longer_than_the_token_gap():
    assert http.FIRST_EVENT_SECONDS >= 3600 and http.NEXT_EVENT_SECONDS == 360
    assert http.KEEPALIVE_SECONDS <= 10


class LongPrefillTokenizer:
    def encode(self, request): return list(range(1, 161))
    def stream(self): return None
    def step(self, stream, token): return "x"
    def decode(self, ids): return "x" * len(ids)


class DisconnectDuringPrefillTest(unittest.TestCase):
    """A client that closes during a long prefill stops the batch worker within one chunk."""

    def setUp(self):
        self.log = tempfile.NamedTemporaryFile(delete=False).name
        env = dict(os.environ, FAKE_PREFILL_ROWS="4", FAKE_DELAY="0.05", FAKE_COMMAND_LOG=self.log,
                   FAKE_SLOT_CONTEXTS="1024,1024")
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)],
                                                             env=dict(kwargs.pop("env", {}), **env), **kwargs)):
            self.backend = BatchedEngineBackend("unused", "unused", context=1024, streams=2, output_timeout=5)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(self.backend, LongPrefillTokenizer())))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown(); self.server.server_close()
        self.backend.close()
        os.unlink(self.log)

    def post(self, body):
        encoded = json.dumps(body).encode()
        client = socket.create_connection(("127.0.0.1", self.port))
        client.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(encoded)}\r\n\r\n".encode() + encoded)
        return client

    def commands(self):
        with open(self.log) as file:
            return [json.loads(line) for line in file]

    def test_stream_headers_come_at_admission_and_a_close_stops_the_prefill(self):
        start = time.monotonic()
        client = self.post(chat(stream=True, max_tokens=4))
        client.settimeout(5)
        head = client.recv(4096)
        # 40 chunks at 50 ms: the headers come long before the 2 s prefill ends.
        self.assertTrue(head.startswith(b"HTTP/1.1 200"), head)
        self.assertLess(time.monotonic() - start, 1.0)
        time.sleep(0.3)
        client.close()
        closed = time.monotonic()
        while time.monotonic() - closed < 3 and not any(c["op"] == "close" for c in self.commands()):
            time.sleep(0.02)
        ops = [c["op"] for c in self.commands()]
        self.assertIn("close", ops)
        stop = time.monotonic() - closed
        # The poll (0.25 s) plus one 50 ms chunk, with room for a slow CI host.
        self.assertLess(stop, 1.0)
        after = ops[ops.index("close") + 1:]
        self.assertNotIn("prefill", after)
        self.assertLess(ops.count("prefill"), 30)

    def test_plain_request_close_also_stops_the_prefill(self):
        client = self.post(chat(max_tokens=4))
        time.sleep(0.3)
        client.close()
        closed = time.monotonic()
        while time.monotonic() - closed < 3 and not any(c["op"] == "close" for c in self.commands()):
            time.sleep(0.02)
        ops = [c["op"] for c in self.commands()]
        self.assertIn("close", ops)
        self.assertLess(ops.count("prefill"), 30)


class LiveHandlerServer:
    """A real loopback server with a backend that never answers."""
    def __init__(self, **options):
        self.backend = SilentBackend()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(self.backend, PiecesTokenizer(["x"]),
                                                                                **options)))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def close(self):
        self.server.shutdown(); self.server.server_close()

    def connect(self):
        client = socket.create_connection(("127.0.0.1", self.port))
        client.settimeout(5)
        return client


def closed_by_server(client, limit):
    start = time.monotonic()
    while time.monotonic() - start < limit:
        try:
            data = client.recv(4096)
        except socket.timeout:
            return False
        if not data:
            return True
    return False


def test_a_slow_request_head_is_closed_at_the_read_deadline(monkeypatch):
    # RA-7: one header byte every 0.1 s never trips the 30 s socket timeout; the request deadline closes it.
    monkeypatch.setattr(http, "READ_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(http, "WATCH_SECONDS", 0.05)
    live = LiveHandlerServer()
    try:
        client = live.connect()
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")
        start = time.monotonic()
        closed = False
        for _ in range(40):
            try:
                client.sendall(b"X")
            except OSError:
                closed = True
                break
            readable = select.select([client], [], [], 0.1)[0]
            if readable and client.recv(4096) == b"":
                closed = True
                break
        assert closed and time.monotonic() - start < 2.0
    finally:
        live.close()


def test_connections_above_the_cap_get_503(monkeypatch):
    monkeypatch.setattr(http, "MAX_CONNECTIONS", 2)
    live = LiveHandlerServer()
    try:
        held = [live.connect() for _ in range(2)]
        time.sleep(0.2)
        extra = live.connect()
        reply = extra.recv(4096)
        head, _, body = reply.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 503") and b"Content-Length: %d" % len(body) in head
        assert json.loads(body)["error"]["message"] == "Too many connections. Try again later."
        for client in held: client.close()
        time.sleep(0.2)
        again = live.connect()
        again.sendall(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % live.port)
        assert again.recv(4096).startswith(b"HTTP/1.1 200")
    finally:
        live.close()


class StepFailingTokenizer(PiecesTokenizer):
    def step(self, stream, token):
        raise RuntimeError("tokenizer fault")


def test_unexpected_error_after_stream_headers_ends_the_stream_with_an_sse_error(capsys):
    # Headers go out at admission, so an unexpected fault after them must still end the stream as SSE.
    from server.tests.serve_fixtures import ScriptBackend
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(ScriptBackend(token_events(["a"])),
                                                                       StepFailingTokenizer(["a"]))))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = socket.create_connection(("127.0.0.1", server.server_address[1]))
        client.settimeout(5)
        body = json.dumps(chat(stream=True)).encode()
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                       b"Content-Length: %d\r\n\r\n" % (server.server_address[1], len(body)) + body)
        raw = b""
        while chunk := client.recv(65536):
            raw += chunk
    finally:
        server.shutdown(); server.server_close()
    assert raw.startswith(b"HTTP/1.1 200") and raw.count(b"HTTP/1.1") == 1
    events = [part[6:] for part in raw.split(b"\r\n\r\n", 1)[1].decode().split("\n\n") if part.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["error"]["message"] == "The server could not process the request."
    assert "tokenizer fault" not in raw.decode()


def post_stream(live):
    client = live.connect()
    body = json.dumps(chat(stream=True)).encode()
    client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                   b"Content-Length: %d\r\n\r\n" % (live.port, len(body)) + body)
    return client


def test_requests_per_client_above_the_limit_get_429():
    # RA-10: one client address cannot fill the shared queue. Loopback clients share one address, so it is opt-in.
    live = LiveHandlerServer(max_requests_per_client=2)
    try:
        held = [post_stream(live) for _ in range(2)]
        for client in held:
            assert client.recv(4096).startswith(b"HTTP/1.1 200")
        extra = post_stream(live)
        reply = b""
        while chunk := extra.recv(4096):  # an error reply closes the connection
            reply += chunk
        head, _, body = reply.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 429")
        assert json.loads(body)["error"]["message"] == "This client has too many open requests. Try again later."
        assert len(live.backend.jobs) == 2
        held[0].close()
        deadline = time.monotonic() + 3
        while not any(job.cancelled.is_set() for job in live.backend.jobs):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        again = post_stream(live)
        assert again.recv(4096).startswith(b"HTTP/1.1 200")
    finally:
        live.close()


def test_requests_per_client_are_not_limited_by_default():
    live = LiveHandlerServer()
    try:
        clients = [post_stream(live) for _ in range(12)]
        assert all(client.recv(4096).startswith(b"HTTP/1.1 200") for client in clients)
    finally:
        live.close()
