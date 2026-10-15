"""CPU tests: always-on per-turn stderr lines, opt-in max_tokens clamp, and the overflow message."""
import contextlib
import io
import json
import re
import unittest
from server import metrics
from server.api import ApiError, validate_request
from server.backend import EngineBackend, Job
from server.http import Service


def job(enabled):
    item = Job(dict(ids=[1, 2, 3], max_tokens=8))
    item.turn["enabled"] = enabled
    item.turn["request_id"] = 7
    item.turn["t_admit"] = item.turn["t_first_gpu"] = item.turn["t_submit"] + 0.01
    return item


def turn_lines(text):
    return [json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("crucible_turn ")]


class TurnLogTest(unittest.TestCase):
    def run_turn(self, enabled):
        item, err = job(enabled), io.StringIO()
        with contextlib.redirect_stderr(err):
            token = EngineBackend._stamp(item, dict(event="token", id=5))
            done = EngineBackend._stamp(item, dict(event="done", prompt_tokens=3, completion_tokens=1,
                                                   finish_reason="stop", cached_tokens=0, prefill_seconds=0.002))
        return token, done, turn_lines(err.getvalue())

    def test_line_without_opt_in_and_unchanged_events(self):
        token, done, lines = self.run_turn(False)
        self.assertEqual(token, dict(event="token", id=5))
        self.assertNotIn("crucible_metrics", done)
        [line] = lines
        self.assertEqual(line["request_id"], 7)
        self.assertEqual((line["prompt_tokens"], line["cached_tokens"], line["completion_tokens"]), (3, 0, 1))
        self.assertEqual(line["finish"], "stop")
        self.assertEqual(line["session_label"], "miss")
        self.assertGreaterEqual(line["ttft_s"], 0)
        self.assertGreaterEqual(line["decode_s"], 0)
        self.assertEqual(line["prefill_s"], 0.002)

    def test_opt_in_keeps_response_metrics(self):
        token, done, lines = self.run_turn(True)
        self.assertIn("crucible_t", token)
        self.assertEqual(done["crucible_metrics"]["request_id"], 7)
        self.assertEqual(len(lines), 1)

    def test_prefill_line(self):
        turn = metrics.new_turn(request_id=3)
        turn["t_first_gpu"] = 10.0
        line = metrics.prefill_line(turn, 4096, 12163, 12.5)
        self.assertEqual(json.loads(line.split(" ", 1)[1]), dict(request_id=3, done=4096, total=12163, elapsed_s=2.5))

    def test_line_survives_nan(self):
        line = metrics.turn_line(dict(prefill_s=float("nan"), request_id=4), dict(prompt_tokens=1))
        self.assertEqual(json.loads(line.split(" ", 1)[1]), dict(request_id=4, error="unloggable metrics"))

    def test_cache_fields(self):
        """Oct 5 incident: a cache that stops saving must show in the turn line."""
        done = dict(prompt_tokens=94222, cached_tokens=53248, cache_saved_tokens=0, cache_tail_tokens=0,
                    cache_save_refused="host", cache_entries=9, cache_bytes=10 << 30, cache_evictions=3,
                    host_free_mib=9830, slot_context=262144, session_replay=False, cache_class="session")
        line = json.loads(metrics.turn_line(dict(request_id=1), done).split(" ", 1)[1])
        for key in ("cache_saved_tokens", "cache_tail_tokens", "cache_save_refused", "cache_entries", "cache_bytes",
                    "cache_evictions", "host_free_mib", "slot_context"):
            self.assertEqual(line[key], done[key])
        self.assertIs(line["replay_hit"], False)
        self.assertIs(json.loads(metrics.turn_line({}, dict(cached_tokens=9, session_replay=True)).split(" ", 1)[1])
                      ["replay_hit"], True)
        # A replay state that was still on the slot.
        resumed = json.loads(metrics.turn_line({}, dict(cached_tokens=9, session_replay=True, slot_resume=True))
                             .split(" ", 1)[1])
        self.assertIs(resumed["slot_resume"], True)
        self.assertNotIn("slot_resume", line)
        # A resume from an in-slot checkpoint (soak405: the prompt left the previous output).
        point = json.loads(metrics.turn_line({}, dict(cached_tokens=9, session_replay=True, checkpoint_resume=True))
                           .split(" ", 1)[1])
        self.assertIs(point["checkpoint_resume"], True)
        self.assertNotIn("checkpoint_resume", line)
        # Older engines omit the fields; a wrong type never reaches the line.
        old = json.loads(metrics.turn_line({}, dict(cached_tokens=0, cache_entries="9")).split(" ", 1)[1])
        self.assertNotIn("cache_entries", old)
        self.assertNotIn("cache_saved_tokens", old)

    def test_line_survives_missing_metrics(self):
        self.assertTrue(metrics.turn_line(None, dict(prompt_tokens=1)).startswith("crucible_turn {"))


class FakeBackend:
    context = 100


class ClampTest(unittest.TestCase):
    def request(self, max_tokens):
        return validate_request(dict(messages=[dict(role="user", content="hi")], max_tokens=max_tokens), "chat", "crucible")

    def test_clamp_is_opt_in(self):
        self.assertEqual(Service(FakeBackend(), None).fit(self.request(64), 90).max_tokens, 64)

    def test_clamp_lowers_to_room(self):
        service = Service(FakeBackend(), None, clamp_max_tokens=True)
        self.assertEqual(service.fit(self.request(64), 90).max_tokens, 11)
        self.assertEqual(service.fit(self.request(5), 90).max_tokens, 5)
        self.assertEqual(service.fit(self.request(64), 101).max_tokens, 64)  # prompt alone overflows: unchanged

    def test_overflow_message_matches_client_patterns(self):
        backend = EngineBackend.__new__(EngineBackend)
        backend.context = 100
        with self.assertRaises(ApiError) as caught:
            backend.submit(list(range(90)), self.request(64))
        # Pi's overflow detection (pi-ai utils/overflow.js) uses this phrase to start compaction.
        self.assertRegex(str(caught.exception), re.compile("exceeds the context window", re.I))
        self.assertIn("90 tokens", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
