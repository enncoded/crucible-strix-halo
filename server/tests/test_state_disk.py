"""The disk tier for conversation states: the server stages a disk read before an
open, and a stop lets the engine write its states, so a restarted server continues a session without a cold prefill."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from server.api import validate_request
from server import batch_backend
from server.batch_backend import (BatchedEngineBackend, STAGE_AHEAD_POLL_SECONDS, STAGE_MIN_TOKENS, STAGE_POLL_SECONDS,
                                  STAGE_REFRESH_SECONDS, STATE_FLUSH_SECONDS)

FAKE = Path(__file__).parent / "fake_batch_engine.py"


class StateDiskTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.disk = Path(self.dir.name) / "disk.json"
        self.commands = Path(self.dir.name) / "commands.jsonl"
        self.request = validate_request(dict(prompt="x", max_tokens=4, spec=0), "completion", "crucible")
        self.backends = []

    def tearDown(self):
        for backend in self.backends: backend.close()
        self.dir.cleanup()

    def start(self, state_disk="1", flush_delay="0", ignore_term="0", **options):
        popen = subprocess.Popen
        env = dict(FAKE_STATE_DISK=state_disk, FAKE_DISK_FILE=str(self.disk), FAKE_COMMAND_LOG=str(self.commands),
                   FAKE_FLUSH_DELAY=flush_delay, FAKE_SLOT_CONTEXTS="40000,40000", FAKE_IGNORE_TERM=ignore_term)
        with patch.dict(os.environ, env), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            backend = BatchedEngineBackend("unused", "unused", context=40000, streams=2, conversation_cache_mib=64,
                                           session_replay=True, state_dir=str(Path(self.dir.name) / "states"),
                                           **options)
        self.backends.append(backend)
        return backend

    def run_job(self, backend, ids):
        job = backend.submit(ids, self.request)
        while True:
            event = job.events.get(timeout=10)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done": return job

    def ops(self):
        if not self.commands.exists(): return []
        return [json.loads(line)["op"] for line in self.commands.read_text().splitlines()]

    def test_restart_resumes_from_disk(self):
        ids = list(range(1, STAGE_MIN_TOKENS + 100))
        first = self.start(flush_delay="0.4")
        self.assertTrue(first.state_disk)
        self.run_job(first, ids)
        start = time.monotonic()
        first.close()
        # The server closed stdin and waited for the engine to write its states; it did not stop it with a signal.
        self.assertGreaterEqual(time.monotonic() - start, 0.4)
        self.assertEqual(first.process.returncode, 0)
        self.assertEqual(json.loads(self.disk.read_text()), [ids])
        self.commands.unlink()
        second = self.start()
        job = self.run_job(second, ids + [7, 8, 9])
        self.assertEqual(job.turn.get("opened_cached"), len(ids))
        ops = self.ops()
        # The read ran before the open: two polls while it read, then the ready reply.
        self.assertEqual(ops[:ops.index("open")], ["stage", "stage", "stage"])
        self.assertEqual(job.turn.get("state_disk"), "ready")

    def test_short_prompts_skip_the_stage_command(self):
        backend = self.start()
        self.run_job(backend, [1, 2, 3])
        self.assertNotIn("stage", self.ops())

    def test_an_engine_without_the_tier_gets_no_stage_command(self):
        backend = self.start(state_disk="0")
        self.assertFalse(backend.state_disk)
        self.run_job(backend, list(range(1, STAGE_MIN_TOKENS + 10)))
        self.assertNotIn("stage", self.ops())

    def test_options_are_forwarded(self):
        with patch("server.batch_backend.EngineBackend.__init__") as initialize:
            BatchedEngineBackend("unused", "unused", streams=2, conversation_cache_mib=64, session_replay=True,
                                 state_dir="/home/u/states", state_disk_gib=100, state_write_gib_per_day=0)
        options = initialize.call_args.kwargs["engine_options"]
        self.assertEqual(options[options.index("--state-disk-write-gib-per-day") + 1], "0")
        self.assertEqual(options[options.index("--state-dir") + 1], "/home/u/states")
        self.assertEqual(options[options.index("--state-disk-gib") + 1], "100")
        with patch("server.batch_backend.EngineBackend.__init__") as initialize:
            BatchedEngineBackend("unused", "unused", streams=2, conversation_cache_mib=64, cache_probation=0,
                                 cache_idle_minutes=5)
        options = initialize.call_args.kwargs["engine_options"]
        self.assertEqual(options[options.index("--cache-probation") + 1], "0")
        self.assertEqual(options[options.index("--cache-idle-minutes") + 1], "5")
        for bad in (dict(state_dir="relative"), dict(cache_probation=2), dict(cache_probation=1, conversation_cache_mib=0, session_replay=False), dict(state_dir="/x", session_replay=False),
                    dict(state_disk_gib=10), dict(state_dir="/x", state_disk_gib=0),
                    dict(state_write_gib_per_day=10), dict(state_dir="/x", state_write_gib_per_day=-1),
                    dict(cache_idle_minutes=0), dict(cache_idle_minutes=10081),
                    dict(cache_idle_minutes=5, conversation_cache_mib=0, session_replay=False)):
            arguments = dict(streams=2, conversation_cache_mib=64, session_replay=True) | bad
            with self.assertRaises(ValueError), patch("server.batch_backend.EngineBackend.__init__"):
                BatchedEngineBackend("unused", "unused", **arguments)


    def test_a_ready_read_is_kept_while_the_turn_waits_for_a_slot(self):
        """the engine drops a ready read that no poll touched for 60 s, so a turn that waits for a
        slot polls its ready read again; a read that the engine started again counts as a new wait."""
        backend = object.__new__(BatchedEngineBackend)
        backend.staging = {}
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto")
        job.cancelled = type("Event", (), {"is_set": lambda self: False})()
        job.turn = {}
        replies = iter(["ready", "ready", "reading", "ready"])
        backend._stage = lambda payload: next(replies)
        clock = [0.0]
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertTrue(backend._staged(job))
            clock[0] = STAGE_REFRESH_SECONDS / 2
            self.assertTrue(backend._staged(job))           # no poll yet
            clock[0] = STAGE_REFRESH_SECONDS + 1
            self.assertTrue(backend._staged(job))           # a poll keeps the read state ("ready")
            clock[0] = 2 * STAGE_REFRESH_SECONDS + 2
            self.assertFalse(backend._staged(job))          # the engine reads it again: the turn waits
            self.assertEqual(job.turn["state_disk_wait_s"], 0)
            clock[0] += 1
            self.assertTrue(backend._staged(job))
        self.assertLess(STAGE_REFRESH_SECONDS, 60 / 2)  # engine/runtime/state_disk.h stage_seconds

    def test_a_turn_that_waits_for_a_slot_keeps_its_read(self):
        """the exact path. A turn with no free slot never reaches _staged in _admit, so only
        _stage_ahead can touch its read. The model engine drops a ready read that no stage touched for 60 s, and a
        stage of a dropped read starts a new one. The turn waits 300 s for a slot: one read only, and it opens on it."""
        import queue, threading
        from collections import deque
        clock = [0.0]

        class Engine:
            def __init__(self): self.reads, self.started, self.touched = 0, None, None
            def state(self):
                if self.started is not None and clock[0] - self.started >= 1 and clock[0] - self.touched > 60:
                    self.started = None  # the engine dropped the ready read (state_disk.h Expire)
                if self.started is None:
                    self.reads += 1
                    self.started = clock[0]
                self.touched = clock[0]
                return "ready" if clock[0] - self.started >= 1 else "reading"

        engine = Engine()
        backend = object.__new__(BatchedEngineBackend)
        backend.state_disk, backend.staging, backend.kept = True, {}, {}
        backend.waiting, backend.jobs, backend.free_contexts = deque(), queue.Queue(), []
        backend.admission_lock, backend.pending_tokens = threading.Lock(), 0
        backend._stage = lambda payload: engine.state()
        opened = []
        backend._open = lambda job, slot_context: opened.append((clock[0], engine.state())) or True
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto", max_tokens=8)
        job.cancelled = threading.Event()
        job.after, job.turn = None, {}
        backend.jobs.put(job)
        backend.pending_tokens = len(job.payload["ids"])
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 300:  # the worker loop runs _admit all the time (here every 0.5 s); no slot is free
                self.assertFalse(backend._admit())
                clock[0] += 0.5
            backend.free_contexts.append(40000)
            self.assertTrue(backend._admit())
        self.assertEqual(engine.reads, 1)
        self.assertEqual(opened, [(300.0, "ready")])

    def test_turns_without_a_slot_are_polled_less_often(self):
        """each stage poll costs the engine command thread about 8 ms while other sessions
        decode. A reading turn that has no slot is polled every STAGE_AHEAD_POLL_SECONDS, not every 50 ms."""
        import queue, threading
        from collections import deque
        clock = [0.0]
        polls = []
        backend = object.__new__(BatchedEngineBackend)
        backend.state_disk, backend.staging, backend.kept = True, {}, {}
        backend.waiting, backend.jobs, backend.free_contexts = deque(), queue.Queue(), []
        backend.admission_lock, backend.pending_tokens = threading.Lock(), 0
        backend._stage = lambda payload: polls.append(clock[0]) or "reading"  # a slow read
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto", max_tokens=8)
        job.cancelled = threading.Event()
        job.after, job.turn = None, {}
        backend.jobs.put(job)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 10:
                self.assertFalse(backend._admit())
                clock[0] += 0.01
            self.assertLessEqual(len(polls), 10 / STAGE_AHEAD_POLL_SECONDS + 2)
            # With a free slot the turn waits for its read and polls it every STAGE_POLL_SECONDS again.
            backend.free_contexts.append(40000)
            before = len(polls)
            while clock[0] < 11:
                self.assertFalse(backend._admit())
                clock[0] += 0.01
            self.assertGreaterEqual(len(polls) - before, 0.5 / STAGE_POLL_SECONDS)  # 0.01 s steps: about 17

    def test_a_new_read_after_a_gap_is_a_new_wait(self):
        """A "reading" reply after a long gap (the engine dropped the ready read and reads it again) restarts the
        wait, so the turn does not open beside an unused full read."""
        backend = object.__new__(BatchedEngineBackend)
        backend.staging = {}
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto")
        job.cancelled = type("Event", (), {"is_set": lambda self: False})()
        job.turn = {}
        replies = iter(["reading", "reading", "ready"])
        backend._stage = lambda payload: next(replies)
        clock = [0.0]
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertFalse(backend._staged(job))
            clock[0] = 300.0
            self.assertFalse(backend._staged(job))
            self.assertEqual(job.turn["state_disk_wait_s"], 0)
            clock[0] += 1
            self.assertTrue(backend._staged(job))

    def test_stop_fits_the_service_stop_time(self):
        """A service unit stops with TimeoutStopSec=120. The engine flushes for at most 75 s, the server
        waits a little longer, then kills it (a tier engine waits after SIGTERM)."""
        header = (Path(__file__).parents[2] / "engine/serve/batched_serve.h").read_text()
        import re
        flush, grace = map(float, re.search(r"kStateFlushSeconds = (\d+), kStateTermGrace = (\d+);", header).groups())
        self.assertLessEqual(flush, 75)
        self.assertLess(flush, STATE_FLUSH_SECONDS)
        self.assertLessEqual(STATE_FLUSH_SECONDS + 10, 120 - 15)
        self.assertLessEqual(grace, 120 - 15)
        backend = self.start(flush_delay="30", ignore_term="1")
        backend.flush_seconds = 0.5
        start = time.monotonic()
        backend.close()
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(backend.process.returncode, -9)


if __name__ == "__main__":
    unittest.main()
