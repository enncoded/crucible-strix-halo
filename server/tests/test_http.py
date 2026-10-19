import io
import json
import queue
import threading
from email.message import Message
from types import SimpleNamespace
import unittest
from server.http import Service, handler_for

class FakeTokenizer:
    def encode(self, request):
        return [10]
    def stream(self):
        return None
    def step(self, stream, token):
        return "hello" if token == 20 else ""
    def decode(self, ids):
        return "hello"

class FakeBackend:
    def __init__(self):
        self.jobs, self.requests = [], []
    def submit(self, ids, request):
        self.requests.append(request)
        events = queue.Queue()
        for event in (dict(event="token", id=20), dict(event="token", id=248046),
                      dict(event="done", prompt_tokens=1, completion_tokens=2, finish_reason="stop")):
            events.put(event)
        job = SimpleNamespace(events=events, cancelled=threading.Event())
        self.jobs.append(job)
        return job

class HttpTests(unittest.TestCase):
    def handle(self, body, path="/v1/chat/completions", host="127.0.0.1:8790", raw=None, content_length=None,
               **service_options):
        backend = FakeBackend()
        handler = handler_for(Service(backend, FakeTokenizer(), **service_options))
        instance = handler.__new__(handler)
        instance.path = path
        encoded = raw if raw is not None else json.dumps(body).encode()
        instance.headers = Message()
        for key, value in {"Host": host, "Content-Length": str(len(encoded)), "Content-Type": "application/json"}.items():
            instance.headers[key] = value
        if content_length is not None:
            instance.headers.replace_header("Content-Length", content_length)
        instance.server = SimpleNamespace(server_address=("127.0.0.1", 8790))
        instance.rfile, instance.wfile = io.BytesIO(encoded), io.BytesIO()
        status, headers = [], {}
        instance.send_response = status.append
        instance.send_header = lambda name, value: headers.update({name: value})
        instance.end_headers = lambda: None
        instance.do_POST()
        self.instance = instance
        return status[0], headers, instance.wfile.getvalue(), backend

    def chat(self, **options):
        return dict(messages=[dict(role="user", content="hello")], max_tokens=2) | options

    def test_draft_capture_gets_each_finished_turn(self):
        # serve --draft-capture: the class and the IDs of a finished turn, nothing else.
        seen = []
        capture = SimpleNamespace(record=lambda **kwargs: seen.append(kwargs))
        status, _, _, _ = self.handle(self.chat(), capture=capture)
        self.assertEqual(status, 200)
        self.assertEqual(seen, [dict(tools=False, thinking=False, prompt_ids=[10], output_ids=[20, 248046])])

    def test_draft_capture_runs_after_the_job_cancel_and_any_fault_keeps_the_reply(self):
        # the capture runs at the end of the request, after the job cancel; a fault of any type is
        # logged by type only, and the reply and the connection stay.
        cancelled = []
        def record(**kwargs):
            cancelled.append(all(job.cancelled.is_set() for job in self.backend_jobs))
            raise KeyError("secret")
        capture = SimpleNamespace(record=record)
        original = FakeBackend.submit
        def submit(backend, ids, request):
            job = original(backend, ids, request)
            self.backend_jobs = backend.jobs
            return job
        FakeBackend.submit = submit
        try:
            status, _, raw, _ = self.handle(self.chat(), capture=capture)
        finally:
            FakeBackend.submit = original
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "hello")
        self.assertEqual(cancelled, [True])
        faulted = getattr(self.instance, "close_connection", False)
        self.handle(self.chat())
        self.assertEqual(faulted, getattr(self.instance, "close_connection", False))

    def test_draft_stats_get_the_done_events_of_the_turn_and_a_fault_keeps_the_reply(self):
        # serve --draft-stats: the class and the done events that carry draft_event, nothing else.
        seen = []
        capture = SimpleNamespace(record=lambda **kwargs: None, stats=object(),
                                  record_draft=lambda **kwargs: seen.append(kwargs))
        original = FakeBackend.submit
        def submit(backend, ids, request):
            job = original(backend, ids, request)
            job.turn = dict(draft_event=dict(event="done", draft_tally=[]))
            return job
        FakeBackend.submit = submit
        try:
            status, _, _, _ = self.handle(self.chat(), capture=capture)
            self.assertEqual(status, 200)
            self.assertEqual(seen, [dict(tools=False, thinking=False, events=[dict(event="done", draft_tally=[])])])
            def broken(**kwargs):
                raise KeyError("secret")
            capture.record_draft = broken
            status, _, raw, _ = self.handle(self.chat(), capture=capture)
        finally:
            FakeBackend.submit = original
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "hello")

    def test_nonstream_chat_usage_and_ids(self):
        status, headers, raw, backend = self.handle(self.chat(return_token_ids=True, seed=18446744073709551615))
        response = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(response["seed"], 18446744073709551615)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(response["choices"][0]["token_ids"], [20,248046])
        self.assertEqual(response["usage"], dict(prompt_tokens=1, completion_tokens=2, total_tokens=3))
        self.assertTrue(backend.jobs[0].cancelled.is_set())

    def test_server_mtp_window_reaches_the_backend(self):
        # serve --mtp-window 0: requests without mtp_prefix use the full MTP window; a client value wins.
        for path, body in (("/v1/chat/completions", self.chat()), ("/v1/completions", dict(prompt="hi", max_tokens=2))):
            _, _, _, backend = self.handle(body, path, default_mtp_prefix=0)
            self.assertEqual(backend.requests[0].mtp_prefix, 0)
            _, _, _, backend = self.handle(body | dict(mtp_prefix=256), path, default_mtp_prefix=0)
            self.assertEqual(backend.requests[0].mtp_prefix, 256)
            _, _, _, backend = self.handle(body, path)
            self.assertEqual(backend.requests[0].mtp_prefix, 256)

    def test_stream_chat(self):
        status, headers, raw, backend = self.handle(self.chat(stream=True, return_token_ids=True,
                                                             stream_options=dict(include_usage=True), seed=987))
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
        self.assertTrue(all(event["seed"] == 987 for event in events))
        ids = [choice["token_id"] for event in events for choice in event["choices"] if "token_id" in choice]
        self.assertEqual(ids, [20,248046])
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 2)
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))

    def test_stream_completion(self):
        status, _, raw, _ = self.handle(dict(prompt="hello", max_tokens=2, stream=True), "/v1/completions")
        self.assertEqual(status, 200)
        self.assertIn(b'"text": "hello"', raw)

    def test_invalid_sampling_before_submission(self):
        status, _, raw, backend = self.handle(self.chat(temperature=10**400))
        self.assertEqual(status, 400)
        self.assertIn("temperature must be a finite number", json.loads(raw)["error"]["message"])
        self.assertEqual(backend.jobs, [])

    def test_invalid_json(self):
        status, _, raw, backend = self.handle(None, raw=b'{"broken":')
        self.assertEqual(status, 400)
        self.assertEqual(backend.jobs, [])

    def test_duplicate_fields(self):
        raw = b'{"messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0,"temperature":0}'
        status, _, _, backend = self.handle(None, raw=raw)
        self.assertEqual(status, 400)
        self.assertEqual(backend.jobs, [])

    def test_surrogate_error_key(self):
        status, _, raw, backend = self.handle(self.chat(**{"\ud800": 1}))
        self.assertEqual(status, 400)
        self.assertIn("not supported", json.loads(raw)["error"]["message"])
        self.assertEqual(backend.jobs, [])

    def test_content_length_boundary(self):
        for value, expected in (("1" * 10000, 413), ("²", 411), ("٠", 411), ("-1", 411)):
            status, _, raw, backend = self.handle(self.chat(), content_length=value)
            self.assertEqual(status, expected)
            self.assertEqual(backend.jobs, [])

    def test_host_guard(self):
        status, _, raw, backend = self.handle(self.chat(), host="malicious.example:8790")
        self.assertEqual(status, 403)
        self.assertEqual(backend.jobs, [])

if __name__ == "__main__":
    unittest.main()
