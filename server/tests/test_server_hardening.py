"""HTTP transport edges found by the server-hardening pass."""
import json
import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from server.http import Service, handler_for
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, invoke, token_events

CANARY = "CANARY5521"


@pytest.fixture
def server():
    backend = ScriptBackend(token_events(["hi"]))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(backend, PiecesTokenizer(["hi"]))))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def exchange(port, head, body=b""):
    """Send the whole request, then read the whole reply. A send failure is the result, as a client sees it."""
    client = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        client.sendall(head + body)
        reply = b""
        while chunk := client.recv(65536):
            reply += chunk
        return reply
    except OSError as error:
        return type(error).__name__
    finally:
        client.close()


def post(port, body, path=b"/v1/chat/completions", content_type=b"application/json"):
    return (b"POST %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: %s\r\nContent-Length: %d\r\n\r\n"
            % (path, port, content_type, len(body)))


def test_an_empty_body_is_a_bad_request_not_a_size_error():
    # SH-1: Content-Length 0 got 413 "must fit the 1 MiB limit".
    response = invoke(ScriptBackend([]), PiecesTokenizer([]), raw=b"")
    assert response.statuses == [400]
    assert "empty" in json.loads(response.body)["error"]["message"]


@pytest.mark.parametrize("path, content_type, status", [
    (b"/v1/chat/completions", b"application/json", 413),
    (b"/v1/chat/completions", b"text/plain", 413),
    (b"/v1/other", b"application/json", 404),
])
def test_a_refused_body_is_read_so_the_client_gets_the_error(server, path, content_type, status):
    # SH-2: the server closed the socket with the body unread. The client's upload then failed (Node fetch: EPIPE)
    # and it never saw the status or the message.
    port = server.server_address[1]
    body = b"x" * (24 << 20)  # larger than the loopback socket buffers, so an unread body cannot hide in them
    reply = exchange(port, post(port, body, path, content_type), body)
    assert isinstance(reply, bytes), reply
    assert reply.startswith(b"HTTP/1.1 %d " % status)
    assert json.loads(reply.partition(b"\r\n\r\n")[2])["error"]["message"]


@pytest.mark.parametrize("request_bytes, status", [
    (b"PUT /v1/models HTTP/1.1\r\nHost: x\r\n\r\n", 501),
    (b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n", 501),
    (b"GET /v1/models HTTP/1.1\r\n" + b"X-A: b\r\n" * 120 + b"\r\n", 431),
    (b"GET /v1/models HTTP/1.1\r\nX-A: " + b"b" * 70000 + b"\r\n\r\n", 431),
    (b"G" * 70000 + b" /v1/models HTTP/1.1\r\n\r\n", 414),
])
def test_protocol_errors_are_json_with_an_error_line(server, request_bytes, status, capsys):
    # SH-3: the standard error page was HTML, and no crucible_error line recorded the refusal.
    reply = exchange(server.server_address[1], request_bytes)
    head, _, body = reply.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 %d " % status)
    assert json.loads(body)["error"]["type"] == ("invalid_request_error" if status < 500 else "server_error")
    err = capsys.readouterr().err
    lines = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_error ")]
    assert [line["status"] for line in lines] == [status]


@pytest.mark.parametrize("request_bytes", [
    b"GARBAGE " + CANARY.encode() + b"\r\n\r\n",
    b"POST /v1/" + CANARY.encode() + b" HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
    b"GET /v1/models?q=" + CANARY.encode() + b" HTTP/1.1\r\nHost: x\r\n\r\n",
    b"BREW" + CANARY.encode() + b" /v1/models HTTP/1.1\r\nHost: x\r\n\r\n",
    b"GET /" + CANARY.encode() + b" HTTP/9.9\r\n\r\n",
])
def test_the_access_log_never_records_request_text(server, request_bytes, capsys):
    # SH-3: the access line and the error line quoted the raw request line (privacy rule: never record text).
    reply = exchange(server.server_address[1], request_bytes)
    assert CANARY.encode() not in reply
    err = capsys.readouterr().err
    assert err and CANARY not in err


def test_the_access_log_keeps_known_endpoints(server, capsys):
    port = server.server_address[1]
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}).encode()
    reply = exchange(port, post(port, body), body)
    assert reply.startswith(b"HTTP/1.1 200 ")
    assert '"POST /v1/chat/completions HTTP/1.1" 200' in capsys.readouterr().err


@pytest.mark.parametrize("path, body", [
    ("/v1/messages/count_tokens", {"model": "crucible", "messages": [{"role": "user", "content": "hello"}]}),
    ("/tokenize", {"content": "hello"}),
    ("/v1/tokenize", {"content": "hello"}),
    ("/apply-template", {"messages": [{"role": "user", "content": "hello"}]}),
])
def test_the_access_log_keeps_the_utility_endpoints(server, capsys, path, body):
    # Train 15: the token-count change adds these paths; with SH-3 an unknown path logs as "<other>".
    port = server.server_address[1]
    data = json.dumps(body).encode()
    reply = exchange(port, post(port, data, path.encode()), data)
    # PiecesTokenizer has no chat template, so /apply-template fails here; the access line must still name the path.
    status = reply.split(b" ", 2)[1].decode()
    assert f'"POST {path} HTTP/1.1" {status}' in capsys.readouterr().err
