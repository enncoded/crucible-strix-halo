"""Release audit: input-boundary failures must give a fixed error, never a traceback."""
import contextlib
import http.client
import io
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from server.api import ApiError, validate_request
from server.http import Service, handler_for
from server.tests.serve_fixtures import ScriptBackend, token_events

CANARY = "CANARY-release-audit-7731"


class ReasoningEffortTypeTests(unittest.TestCase):
    """RA-1: a list or object effort raised TypeError (unhashable), so the client got no response."""

    def test_unhashable_effort_is_a_400(self):
        for body in (dict(reasoning_effort=[]), dict(reasoning_effort={}),
                     dict(chat_template_kwargs=dict(reasoning_effort=[CANARY]))):
            with self.subTest(body=body), self.assertRaises(ApiError) as caught:
                validate_request(dict(messages=[dict(role="user", content="x")], **body), "chat", "crucible")
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(caught.exception.param, "reasoning_effort")
            self.assertNotIn(CANARY, str(caught.exception))


class FailingTokenizer:
    """An unexpected fault whose text contains request content."""

    def encode(self, request):
        raise RuntimeError("tokenizer fault near " + CANARY)


class StreamFailingTokenizer:
    """Encodes, then fails when the response starts: the engine job exists by then."""

    def encode(self, request):
        return [10]

    def stream(self):
        raise RuntimeError("decoder fault near " + CANARY)


class TurnBackend(ScriptBackend):
    """Jobs carry a turn with a request ID, as the real backends do."""

    def submit(self, ids, request, **options):
        job = super().submit(ids, request, **options)
        job.turn = dict(request_id=41)
        return job


class LiveServer:
    def __init__(self, tokenizer, backend=None):
        backend = backend or ScriptBackend(token_events(["a"]))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(backend, tokenizer)))
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def post(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(body),
                               {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()


class UnexpectedErrorTests(unittest.TestCase):
    """RA-1: an exception outside ApiError must not drop the connection or print a traceback with its text."""

    def run_failure(self, tokenizer, backend=None):
        stderr = io.StringIO()
        with LiveServer(tokenizer, backend) as live, patch("sys.stderr", stderr), contextlib.redirect_stderr(stderr):
            status, _, body = live.post(dict(messages=[dict(role="user", content=CANARY)]))
        log = stderr.getvalue()
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"]["type"], "server_error")
        self.assertNotIn(CANARY, body.decode())
        self.assertNotIn(CANARY, log)
        self.assertNotIn("Traceback", log)
        lines = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines() if line.startswith("crucible_error ")]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["status"], 500)
        return lines[0]

    def frames(self, line):
        """The error text is the exception type, then up to three basename:line:function frames, innermost last."""
        kind, _, frames = line["error"].partition(" at ")
        self.assertEqual(kind, "unexpected RuntimeError")
        frames = frames.split(" > ")
        self.assertTrue(1 <= len(frames) <= 3)
        for frame in frames:
            name, number, function = frame.split(":")
            self.assertNotIn("/", name)
            self.assertTrue(number.isdecimal() and function.isidentifier())
        return frames

    def test_unexpected_fault_is_a_fixed_500_and_a_content_free_line(self):
        line = self.run_failure(FailingTokenizer())
        self.assertIsNone(line["request_id"])
        frames = self.frames(line)
        self.assertEqual(frames[-1].split(":")[::2], ["test_release_audit.py", "encode"])

    def test_unexpected_fault_after_submit_names_the_request(self):
        line = self.run_failure(StreamFailingTokenizer(), TurnBackend(token_events(["a"])))
        self.assertEqual(line["request_id"], 41)
        self.assertEqual(self.frames(line)[-1].split(":")[::2], ["test_release_audit.py", "stream"])

    def test_a_fault_while_naming_the_frames_still_gives_the_fixed_line(self):
        # A raise inside the 500 path would reach socketserver, which prints the chained original exception text.
        with patch("server.http.extract_tb", side_effect=ValueError("frame fault")):
            line = self.run_failure(FailingTokenizer())
        self.assertEqual(line["error"], "unexpected RuntimeError")
        self.assertIsNone(line["request_id"])

    def test_server_header_does_not_name_the_python_version(self):
        with LiveServer(FailingTokenizer()) as live, contextlib.redirect_stderr(io.StringIO()):
            _, headers, _ = live.post(dict(messages=[dict(role="user", content="x")]))
        self.assertEqual(headers.get("Server"), "Crucible/0.1")


if __name__ == "__main__":
    unittest.main()
