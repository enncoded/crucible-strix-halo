"""ab.py must reuse the cached prompt prefix. Without cache_prefix_tokens the batched engine caches only the last
grid point of the prompt, which no later request can restore, so every turn prefilled the whole prompt again
(night2, Oct 9: 50 turns, cached_tokens 0 each, two step time limits)."""
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab  # noqa: E402


class FakeBackend:
    def __init__(self, cached=0):
        self.cached, self.calls = cached, []

    def submit(self, ids, request, cache_prefix_tokens=0, **options):
        self.calls.append(dict(length=len(ids), cache_prefix_tokens=cache_prefix_tokens))
        events = queue.Queue()
        events.put({"event": "token", "id": 7})
        events.put({"event": "token", "id": 8})
        events.put({"event": "done", "cached_tokens": self.cached})
        return SimpleNamespace(events=events)


class CachePrefixTests(unittest.TestCase):
    def test_prefix_stops_where_the_draft_warmup_starts(self):
        # The engine restores only a snapshot at or before the warmup start (prompt minus mtp_prefix).
        self.assertEqual(ab.cache_prefix_tokens(list(range(16384)), SimpleNamespace(mtp_prefix=256)), 16128)

    def test_short_prompt_has_no_prefix(self):
        self.assertEqual(ab.cache_prefix_tokens([1, 2, 3], SimpleNamespace(mtp_prefix=256)), 0)

    def test_run_sends_the_prefix_and_reports_cached_tokens(self):
        backend, result = FakeBackend(cached=12288), []
        ab.run(backend, list(range(16384)), SimpleNamespace(mtp_prefix=256), result)
        self.assertEqual(backend.calls, [dict(length=16384, cache_prefix_tokens=16128)])
        self.assertEqual(result[0]["cached_tokens"], 12288)
        self.assertEqual(result[0]["ids"], [7, 8])


class RequireCachedTests(unittest.TestCase):
    def test_warm_round_may_miss(self):
        ab.require_cached([{"cached_tokens": 0}], "warm")

    def test_timed_round_must_hit_for_every_session(self):
        ab.require_cached([{"cached_tokens": 12288}] * 4, "ab")
        with self.assertRaises(RuntimeError):
            ab.require_cached([{"cached_tokens": 12288}, {"cached_tokens": 0}], "ab")


if __name__ == "__main__":
    unittest.main()
