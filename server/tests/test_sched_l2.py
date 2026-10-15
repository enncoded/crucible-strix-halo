"""Train 13 audit L2, and the think-cap line's prefill rows. CPU only, scripted engine."""
import io
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server import metrics
from server.backend import Job
from server.batch_backend import BatchedEngineBackend, LiveRequest
from server.batch_scheduler import choose


def backend_with(*lives, rows=4096, interleave=10):
    backend = object.__new__(BatchedEngineBackend)
    backend.prefill_rows = rows; backend.prefill_rate = 1000.0; backend.fatal = None
    backend.prefill_interleave = interleave; backend.prefill_cursor = None; backend.last_step = float("-inf")
    backend.trace = metrics.Trace(); backend.script = []
    backend.process = SimpleNamespace(stdin=io.StringIO())
    backend._next = lambda: backend.script.pop(0)
    backend.ownership_lock, backend.owned_jobs = threading.Lock(), {}
    backend.closed = SimpleNamespace(is_set=lambda: False)
    backend.kept, backend.free_contexts, backend.pending_tokens = {}, [], 0
    backend.admission_lock = threading.Lock()
    backend.active = {}
    for id, prompt, max_tokens in lives:
        # The worker's queue size with the interleave: 32 plus 9 events for each of the chunk's passes (77 at 10).
        job = Job(dict(ids=list(range(prompt)), max_tokens=max_tokens, stop_ids=[99], spec=0),
                  events=queue.Queue(maxsize=77))
        backend.active[id] = LiveRequest(job, slot_context=262144)
    return backend


def opened(backend, id, cached=0):
    backend.script += [dict(event="opened", request_id=id, slot_generation=id, slot_context=262144,
                            cached_tokens=cached), dict(event="batch_end", op="open")]
    backend._command(dict(op="open", request_id=id, request={}))



class PartialRoomTest(unittest.TestCase):
    """L2: a ready decoder without room for every pass of a chunk still gets a step once decode_max has passed."""

    def test_excluded_decoder_keeps_its_own_step_clock(self):
        backend = backend_with((1, 100_000, 100), (2, 10, 100), (3, 10, 100))
        opened(backend, 1, 0); opened(backend, 2); opened(backend, 3)
        for id in (2, 3):
            backend.active[id].ready, backend.active[id].count = True, 1
        backend.active[2].last_step = backend.active[3].last_step = 0.0
        slow = backend.active[3].job.events
        for _ in range(40): slow.put_nowait(dict(event="token"))  # 40 queued: ready, but no room for 45
        self.assertEqual(backend._ready(), [2, 3])
        self.assertEqual(backend._interleave_decoders(1, 4096), [2])
        backend.script += [dict(event="token", request_id=2, slot_generation=2, id=5),
                           dict(event="batch_part", op="verify_step"),
                           dict(event="batch_end", op="prefill", cursor=4096, engine_s=0.5)]
        with patch("server.batch_backend.time.monotonic", return_value=0.3):
            backend._command(dict(op="prefill", request_id=1, request_ids=[2], every=10), 4096)
        self.assertEqual(backend.active[2].last_step, 0.3)
        self.assertEqual(backend.active[3].last_step, 0.0)
        # Decoder 2 just ran a pass, but decoder 3 waited 0.31 s: the step comes before the next chunk.
        self.assertEqual(backend._step_clock([2, 3]), 0.0)
        self.assertEqual(choose(0.31, [2, 3], backend._pending_prefills(), backend._step_clock([2, 3]), 1000.0)[0],
                         "step")

    def test_every_decoder_in_the_passes_keeps_todays_choice(self):
        backend = backend_with((1, 100_000, 100), (2, 10, 100))
        opened(backend, 1, 0); opened(backend, 2)
        backend.active[2].ready, backend.active[2].count, backend.active[2].last_step = True, 1, 0.0
        backend.script += [dict(event="token", request_id=2, slot_generation=2, id=5),
                           dict(event="batch_part", op="verify_step"),
                           dict(event="batch_end", op="prefill", cursor=4096, engine_s=0.5)]
        with patch("server.batch_backend.time.monotonic", return_value=0.3):
            backend._command(dict(op="prefill", request_id=1, request_ids=[2], every=10), 4096)
        self.assertEqual(backend._step_clock([2]), 0.3)
        self.assertEqual(choose(0.31, [2], backend._pending_prefills(), backend._step_clock([2]), 1000.0)[0], "prefill")

    def test_a_new_decoder_starts_its_clock_at_its_first_token(self):
        backend = backend_with((1, 10, 100))
        opened(backend, 1, 0)
        backend.script += [dict(event="token", request_id=1, slot_generation=1, id=5),
                           dict(event="batch_end", op="prefill", cursor=10, engine_s=0.1)]
        with patch("server.batch_backend.time.monotonic", return_value=7.0):
            backend._command(dict(op="prefill", request_id=1), 10)
        self.assertEqual(backend.active[1].last_step, 7.0)


class ThinkCapPrefillRowsTest(unittest.TestCase):
    """The think-cap crucible_turn line counts the rows that ran, not the chunk sizes (8192 where 682 ran)."""

    def test_resumed_first_job_counts_its_rows(self):
        backend = backend_with((1, 14_163, 200))
        opened(backend, 1, 13_481)
        for cursor in (13_489, 14_163):  # an 8-row head chunk, then the bulk rest
            backend.script += [dict(event="batch_end", op="prefill", cursor=cursor, engine_s=0.1)]
            backend._prefill(1)
        first = backend.active[1].job.turn
        cont = metrics.new_turn(2)
        line = metrics.request_line(first, dict(prompt_tokens=14_163, completion_tokens=200, finish_reason="stop"),
                                    SimpleNamespace(tokens=128, by="max_thinking_tokens"), cont,
                                    dict(prompt_tokens=14_316, cached_tokens=14_163, completion_tokens=43,
                                         prefill_tokens=153))
        fields = json.loads(line.split(" ", 1)[1])
        self.assertEqual(fields["prefill_tokens"], 682)
        self.assertEqual(fields["cont_prefill_tokens"], 153)

    def test_the_line_keeps_the_cache_totals_of_the_continuation(self):
        # The first job is cancelled before its done event, so only the continuation's done event names the cache
        # after the turn (its tail save, totals, host memory). Without them a closed think turn has no cache stats.
        first, cont = metrics.new_turn(1), metrics.new_turn(2)
        line = metrics.request_line(first, dict(prompt_tokens=100, completion_tokens=20, finish_reason="stop"),
                                    SimpleNamespace(tokens=1, by="answer_room"), cont,
                                    dict(prompt_tokens=103, cached_tokens=100, completion_tokens=19,
                                         cache_tail_tokens=121, cache_entries=7, cache_bytes=2 << 30,
                                         cache_evictions=3, host_free_mib=6616, cache_save_refused="host"))
        fields = json.loads(line.split(" ", 1)[1])
        self.assertEqual((fields["cache_tail_tokens"], fields["cache_entries"], fields["cache_bytes"],
                          fields["cache_evictions"], fields["host_free_mib"], fields["cache_save_refused"]),
                         (121, 7, 2 << 30, 3, 6616, "host"))
        self.assertEqual(fields["cached_tokens"], None)  # the client's restore count stays the first job's


if __name__ == "__main__": unittest.main()
