from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from server.api import ApiError, validate_request
from server.backend import EngineBackend

FAKE = Path(__file__).parent / "fake_engine.py"

class BackendTests(unittest.TestCase):
    def setUp(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen", side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = EngineBackend("unused", "unused", context=32)
        self.request = validate_request(dict(prompt="x", max_tokens=8), "completion", "crucible")

    def tearDown(self):
        self.backend.close()
        self.assertIsNotNone(self.backend.process.poll())

    def collect(self, job):
        ids = []
        while True:
            event = job.events.get(timeout=5)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done":
                self.assertEqual(event["completion_tokens"], len(ids))
                return ids
            ids.append(event["id"])

    def test_fifo_requests_and_reset(self):
        jobs = [self.backend.submit([id], self.request) for id in (1,9,1)]
        self.assertEqual(self.collect(jobs[0]), list(range(2,10)))
        self.assertEqual(self.collect(jobs[1]), list(range(10,18)))
        self.assertEqual(self.collect(jobs[2]), list(range(2,10)))

    def test_disconnect_drains_before_next_request(self):
        cancelled = self.backend.submit([1], self.request)
        cancelled.events.get(timeout=5)
        cancelled.cancelled.set()
        next_job = self.backend.submit([9], self.request)
        self.assertEqual(self.collect(next_job), list(range(10,18)))

    def test_sampling_options_reach_the_engine(self):
        request = validate_request(dict(prompt="x", max_tokens=8, temperature=0.7, top_k=50,
                                        top_p=0.9, min_p=0.05, seed=18446744073709551615),
                                   "completion", "crucible")
        job = self.backend.submit([1], request)
        self.assertEqual({key: job.payload[key] for key in ("temperature", "top_k", "top_p", "min_p", "seed")},
                         dict(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05, seed=18446744073709551615))
        self.collect(job)

    def test_decode_defaults_and_off_switches_reach_engine(self):
        for options, expected in (({}, dict(spec=7, draft_q=0.3, draft_head="64k", mtp_prefix=256)),
                                  (dict(spec=0, draft_q=0, draft_head="full", mtp_prefix=0, draft_adapt=0),
                                   dict(spec=0, draft_q=0, draft_head="full", mtp_prefix=0, draft_adapt=0))):
            request = validate_request(dict(prompt="x", max_tokens=8) | options, "completion", "crucible")
            job = self.backend.submit([1], request)
            self.assertEqual({key: job.payload[key] for key in expected}, expected)
            if "draft_adapt" not in options:
                self.assertNotIn("draft_adapt", job.payload)
            self.collect(job)

    def test_context_preflight(self):
        with self.assertRaises(ApiError):
            self.backend.submit([1] * 32, self.request)

if __name__ == "__main__":
    unittest.main()
