"""The batch backend behind the cache-aware HTTP handler."""
import io
import json
import subprocess
import sys
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for

FAKE = Path(__file__).parent / "fake_batch_engine.py"


class CacheAwareTokenizer:
    """The same surface as ModelTokenizer, including cache_prefix_tokens."""
    def encode(self, request):
        return [10]
    def cache_prefix_tokens(self, request, ids):
        return 0
    def stream(self):
        return None
    def step(self, stream, token):
        return ""
    def decode(self, ids):
        return ""


class BatchHttpTest(unittest.TestCase):
    def setUp(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3)

    def tearDown(self):
        self.backend.close()

    def handler(self, path, payload=None, method="POST"):
        handler = handler_for(Service(self.backend, CacheAwareTokenizer()))
        h = handler.__new__(handler)
        h.path = path
        body = json.dumps(payload or {}).encode()
        h.headers = Message()
        for key, value in {"Host": "127.0.0.1:8790", "Content-Length": str(len(body)),
                           "Content-Type": "application/json"}.items():
            h.headers[key] = value
        h.server = SimpleNamespace(server_address=("127.0.0.1", 8790))
        h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
        status = []
        h.send_response = status.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        h.do_POST() if method == "POST" else h.do_GET()
        return status, h.wfile.getvalue()

    def test_chat_request_reaches_the_batch_backend(self):
        status, body = self.handler("/v1/chat/completions",
                                    dict(messages=[dict(role="user", content="hi")], max_tokens=2, spec=0))
        self.assertEqual(status, [200])
        self.assertEqual(json.loads(body)["usage"]["completion_tokens"], 2)
        self.assertNotIn("crucible_metrics", json.loads(body)["usage"])

    def test_opt_in_metrics_and_trace(self):
        request = dict(messages=[dict(role="user", content="hi")], max_tokens=2, spec=0, crucible_metrics=True,
                       stream=True, stream_options=dict(include_usage=True), return_token_ids=True)
        status, body = self.handler("/v1/chat/completions", request)
        self.assertEqual(status, [200])
        chunks = [json.loads(line[6:]) for line in body.decode().split("\n\n") if line.startswith("data: {")]
        stamped = [chunk for chunk in chunks if "crucible_t" in chunk]
        self.assertEqual(len(stamped), 2)
        metrics = chunks[-1]["usage"]["crucible_metrics"]
        self.assertEqual(metrics["t_first_token"], stamped[0]["crucible_t"])
        status, body = self.handler("/v1/crucible/trace?since=0", method="GET")
        self.assertEqual(status, [200])
        ops = [entry["op"] for entry in json.loads(body)["data"] if metrics["request_id"] in entry["request_ids"]]
        self.assertEqual(ops[:2], ["open", "prefill"])
        status, _ = self.handler("/v1/crucible/trace?since=nan", method="GET")
        self.assertEqual(status, [400])


if __name__ == "__main__":
    unittest.main()
