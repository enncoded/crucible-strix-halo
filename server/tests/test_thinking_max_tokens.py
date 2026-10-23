"""Guard: with thinking on, no reply holds more than max_tokens tokens (Discord report, "thinking ignores max tokens").

The server stops a think block early (think_budget) and resumes with a second engine job. These tests drive the real
HTTP handler and the real backends against the fake engines. Each case checks the client's counts: usage,
the streamed token IDs, and the finish reason. They found no failure, so they are a regression guard.
"""
import itertools
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from server.backend import EngineBackend
from server.batch_backend import BatchedEngineBackend
from server.tests.serve_fixtures import chunks, invoke

TESTS = Path(__file__).parent
CONTEXT = 70000
# Large IDs, so that "the prompt's last ID is END - k" tells the fake engine to close the think block after k tokens.
SENT, END, BRK = 100006, 100007, 100008


class Tokenizer:
    def __init__(self, close_after):
        self.close_after = close_after

    def encode(self, request):
        return [END - self.close_after]

    def stream(self):
        return None

    def step(self, state, token):
        return "</think>" if token == END else ("\n\nTime.\n" if token == SENT else "\n\n" if token == BRK else "r ")

    def decode(self, ids):
        return "".join(self.step(None, token) for token in ids)

    def think_close_ids(self):
        return (SENT, END, BRK)


def start_batch(streams):
    popen = subprocess.Popen
    env = dict(os.environ, FAKE_SLOT_CONTEXTS=",".join([str(CONTEXT)] * streams), FAKE_DELAY="0")
    with patch("server.backend.subprocess.Popen", side_effect=lambda *a, **k: popen(
            [sys.executable, str(TESTS / "fake_batch_engine.py")], **{**k, "env": env})):
        return BatchedEngineBackend("unused", "unused", context=CONTEXT, streams=streams)


def start_serial():
    popen = subprocess.Popen
    env = dict(os.environ, FAKE_CONTEXT=str(CONTEXT), FAKE_DELAY="0")
    with patch("server.backend.subprocess.Popen", side_effect=lambda *a, **k: popen(
            [sys.executable, str(TESTS / "fake_engine.py")], **{**k, "env": env})):
        return EngineBackend("unused", "unused", context=CONTEXT)


def ask(backend, close_after, **fields):
    """One thinking request. Returns (streamed or listed token IDs, usage, finish reasons)."""
    body = dict(messages=[dict(role="user", content="hi")], enable_thinking=True, return_token_ids=True,
                stream_options=dict(include_usage=True)) | fields
    response = invoke(backend, Tokenizer(close_after), body)
    assert response.statuses[0] == 200, response.body
    if body.get("stream"):
        events = chunks(response)
        count = sum(1 for event in events for choice in event["choices"] if "token_id" in choice)
        usage = next(event["usage"] for event in events if event.get("usage"))
        return count, usage, [c["finish_reason"] for e in events for c in e["choices"] if c["finish_reason"]]
    reply = json.loads(response.body)
    choice = reply["choices"][0]
    return len(choice["token_ids"]), reply["usage"], [choice["finish_reason"]]


class CountGuard:
    """Mixin: each case needs the client's counts to equal each other and to stay within max_tokens."""

    def check(self, backend, close_after, max_tokens, **fields):
        count, usage, finish = ask(backend, close_after, max_tokens=max_tokens, **fields)
        label = f"close_after={close_after} max_tokens={max_tokens} {fields}"
        self.assertLessEqual(usage["completion_tokens"], max_tokens, label)
        self.assertEqual(count, usage["completion_tokens"], label)
        self.assertEqual(finish, ["length"], label)
        self.assertLessEqual(usage["completion_tokens_details"]["reasoning_tokens"], max_tokens, label)

    def cases(self, sizes, everywhere=True, specs=(0, 7)):
        for max_tokens in sizes:
            cap = max(1, max_tokens - max(1024, max_tokens * 15 // 100))
            # The model closes its own block well before, at, and after the server's cap, and at the very end.
            points = {5, cap - 1, cap, cap + 1, cap + 2, cap + 3, max_tokens - 1, max_tokens, max_tokens + 1}
            if not everywhere:
                points = {cap, cap + 1, max_tokens + 1}
            for close_after, stream, spec in itertools.product(sorted(p for p in points if p > 0), (False, True), specs):
                yield max_tokens, close_after, dict(stream=stream, spec=spec)


class BatchedMaxTokensTest(CountGuard, unittest.TestCase):
    def run_streams(self, streams, sizes):
        backend = start_batch(streams)
        self.addCleanup(backend.close)
        for max_tokens, close_after, fields in self.cases(sizes):
            with self.subTest(max_tokens=max_tokens, close_after=close_after, **fields):
                self.check(backend, close_after, max_tokens, **fields)

    def test_one_stream_never_exceeds_max_tokens(self):
        self.run_streams(1, (2000, 3000))

    def test_two_streams_never_exceed_max_tokens(self):
        self.run_streams(2, (2000,))

    def test_small_max_tokens_never_exceed(self):
        backend = start_batch(1)
        self.addCleanup(backend.close)
        for max_tokens, field, stream in itertools.product((1, 2, 16, 1024, 1025), ("max_tokens", "max_completion_tokens"),
                                                          (False, True)):
            with self.subTest(max_tokens=max_tokens, field=field, stream=stream):
                count, usage, finish = ask(backend, 50, **{field: max_tokens}, stream=stream)
                self.assertEqual((count, usage["completion_tokens"], finish), (max_tokens, max_tokens, ["length"]))

    def test_the_first_job_stops_at_the_cap(self):
        """The batch worker does not generate past the cap: the continuation job does the rest."""
        log = Path(os.environ.get("TMPDIR", "/tmp")) / f"crucible-cmd-{os.getpid()}.log"
        self.addCleanup(lambda: log.unlink(missing_ok=True))
        log.write_text("")
        with patch.dict(os.environ, FAKE_COMMAND_LOG=str(log)):
            backend = start_batch(1)
        self.addCleanup(backend.close)
        count, usage, _ = ask(backend, 3000, max_tokens=3000, spec=0)
        self.assertEqual(count, 3000)
        steps = {}
        for line in log.read_text().splitlines():
            command = json.loads(line)
            if command["op"] == "step":
                for request_id in command.get("request_ids", [command.get("request_id")]):
                    steps[request_id] = steps.get(request_id, 0) + 1
        first, second = sorted(steps)
        cap = 3000 - 1024
        self.assertEqual(steps[first], cap)
        self.assertLessEqual(steps[second], 3000 - cap)


class SerialMaxTokensTest(CountGuard, unittest.TestCase):
    def test_serial_never_exceeds_max_tokens(self):
        backend = start_serial()
        self.addCleanup(backend.close)
        # The serial fake engine is slow (about 3 ms a token), so this runs fewer cases.
        for max_tokens, close_after, fields in self.cases((1100,), everywhere=False, specs=(0,)):
            with self.subTest(max_tokens=max_tokens, close_after=close_after, **fields):
                self.check(backend, close_after, max_tokens, **fields)


if __name__ == "__main__":
    unittest.main()
