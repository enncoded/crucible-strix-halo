import json
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from server.tests import test_http, test_tools
from server.tests.serve_fixtures import joined_calls
from server.tests.test_tools import TOOLS

class CharacterTokenizer:
    def __init__(self, text): self.text = text
    def encode(self, request): return [10]
    def stream(self): return None
    def step(self, stream, token): return self.text[token - 20] if token != 248046 else ""
    def decode(self, ids): return self.text

class CharacterBackend:
    def __init__(self, text, cut=False): self.text, self.cut, self.jobs = text, cut, []
    def submit(self, ids, request):
        events = queue.Queue()
        for index in range(len(self.text)):
            events.put(dict(event="token", id=index + 20))
        if not self.cut:  # cut: max_tokens ends the output, with no end token
            events.put(dict(event="token", id=248046))
        count = len(self.text) + (0 if self.cut else 1)
        events.put(dict(event="done", prompt_tokens=1, completion_tokens=count,
                        finish_reason="length" if self.cut else "stop"))
        job = SimpleNamespace(events=events, cancelled=threading.Event())
        self.jobs.append(job)
        return job

class ToolHttpTests(unittest.TestCase):
    def handle(self, text, cut=False, **options):
        request = dict(messages=[dict(role="user", content="Read a.py")], tools=TOOLS,
                       max_tokens=len(text) + (0 if cut else 1), return_token_ids=True, seed=42) | options
        with patch("server.tests.test_http.FakeTokenizer", return_value=CharacterTokenizer(text)), \
             patch("server.tests.test_http.FakeBackend", side_effect=lambda: CharacterBackend(text, cut)):
            return test_http.HttpTests().handle(request)

    def test_nonstream_tool_response_and_ids(self):
        text = test_tools.ToolOutputTests().block()
        status, _, raw, backend = self.handle(text)
        response = json.loads(raw)
        self.assertEqual(status, 200)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["offset"], 4)
        self.assertEqual(choice["token_ids"], list(range(20, 20 + len(text))) + [248046])
        self.assertEqual(response["usage"]["completion_tokens"], len(text) + 1)
        self.assertTrue(backend.jobs[0].cancelled.is_set())

    def test_stream_openai_delta_shape(self):
        text = "Before.\n" + test_tools.ToolOutputTests().block()
        status, _, raw, backend = self.handle(text, stream=True)
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
        choices = [event["choices"][0] for event in events]
        self.assertEqual("".join(choice["delta"].get("content", "") for choice in choices), "Before.\n")
        deltas = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
        # The id and name come first, before any argument text (item 4, streamed calls).
        self.assertEqual(deltas[0], dict(index=0, id=deltas[0]["id"], type="function",
                                         function=dict(name="read", arguments="")))
        self.assertGreater(len(deltas), 2)
        calls = joined_calls(deltas)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), dict(path="a.py", offset=4, flags=dict(full=True)))
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")
        self.assertEqual(len([choice for choice in choices if "token_id" in choice]), len(text) + 1)
        self.assertNotIn(b"<tool_call>", raw)
        self.assertTrue(backend.jobs[0].cancelled.is_set())

    def malformed(self, text, cut=False, **options):
        lines = []
        with patch("server.metrics.emit", lines.append):
            result = self.handle(text, cut, **options)
        return result, [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_error ")]

    def test_malformed_call_is_content_nonstream_and_stream(self):
        # Design decision (Oct 6), as other engines: no error after HTTP 200; the text is the answer, finish_reason stop.
        # A structural error (junk before the function end). A schema-only failure now goes out as a call.
        text = "Before.\n" + test_tools.ToolOutputTests().block().replace("</parameter>\n</function>",
                                                                          "</parameter>\njunk</function>")
        (status, _, raw, backend), errors = self.malformed(text)
        self.assertEqual(status, 200)
        response = json.loads(raw)
        choice = response["choices"][0]
        self.assertEqual((choice["message"]["content"], choice["finish_reason"]), (text, "stop"))
        self.assertNotIn("tool_calls", choice["message"])
        self.assertIs(response["usage"]["tool_call_fallback"], True)
        self.assertEqual(errors, [dict(request_id=None, status=200, streamed=False, error="tool_call_fallback",
                                       completion_tokens=len(text) + 1, invalid="parameter_tag")])
        self.assertTrue(backend.jobs[0].cancelled.is_set())
        (status, _, raw, backend), errors = self.malformed(text, stream=True, stream_options=dict(include_usage=True))
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
        self.assertFalse([event for event in events if "error" in event])
        choices = [event["choices"][0] for event in events if event["choices"]]
        self.assertEqual("".join(choice["delta"].get("content") or "" for choice in choices), text)
        # The call's id and name went out before the junk, so the stream ends with "length" (design decision, Oct 7).
        self.assertEqual(choices[-1]["finish_reason"], "length")
        self.assertIs(events[-1]["usage"]["tool_call_fallback"], True)
        self.assertEqual(events[-1]["usage"]["tool_call_invalid"], "parameter_tag")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["streamed"])
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))

    def test_partial_call_at_max_tokens_keeps_length(self):
        text = test_tools.ToolOutputTests().block()[:40]
        for stream in (False, True):
            (status, _, raw, _), errors = self.malformed(text, cut=True, stream=stream)
            self.assertEqual(status, 200)
            if stream:
                choices = [json.loads(line[6:])["choices"][0] for line in raw.decode().splitlines()
                           if line.startswith("data: {")]
                self.assertEqual("".join(c["delta"].get("content") or "" for c in choices), text)
                self.assertEqual(choices[-1]["finish_reason"], "length")
            else:
                choice = json.loads(raw)["choices"][0]
                self.assertEqual((choice["message"]["content"], choice["finish_reason"]), (text, "length"))
            self.assertEqual(len(errors), 1)

if __name__ == "__main__": unittest.main()
