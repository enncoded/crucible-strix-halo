"""Prompt lookup is the serve default for speculative requests (depth 7, min 8); --lookup 0 turns it off.

Lookup changes only which drafts the engine verifies, never an output."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server.api import validate_request
from server.backend import EngineBackend, LOOKUP_MIN

FAKE = Path(__file__).parent / "fake_engine.py"


def start(argv, target):
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer"), patch(target) as backend, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener):
        assert startup.main() == 0
    return backend.call_args.kwargs


class StartupTest(unittest.TestCase):
    def test_serial_and_batched_serve_default_to_depth_seven(self):
        self.assertEqual(start(["--serial"], "server.__main__.EngineBackend")["lookup"], 7)
        self.assertEqual(start([], "server.batch_backend.BatchedEngineBackend")["lookup"], 7)
        self.assertEqual(start(["--streams", "4"], "server.batch_backend.BatchedEngineBackend")["lookup"], 7)

    def test_zero_turns_lookup_off(self):
        self.assertEqual(start(["--serial", "--lookup", "0"], "server.__main__.EngineBackend")["lookup"], 0)
        self.assertEqual(start(["--streams", "4", "--lookup", "0"], "server.batch_backend.BatchedEngineBackend")["lookup"], 0)

    def test_out_of_range_depth_is_refused_before_loading(self):
        for value in ("-1", "8", "x"):
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--lookup", value]), \
                    patch.object(startup, "ModelTokenizer") as load, self.assertRaises(SystemExit) as error:
                startup.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_batched_backend_keeps_the_depth(self):
        from server.batch_backend import BatchedEngineBackend
        with patch("server.backend.EngineBackend.__init__", return_value=None) as base:
            BatchedEngineBackend("engine", "checkpoint", 128, streams=2, lookup=7)
        self.assertEqual(base.call_args.kwargs["lookup"], 7)


class PayloadTest(unittest.TestCase):
    def backend(self, **options):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            backend = EngineBackend("unused", "unused", context=32, **options)
        self.addCleanup(backend.close)
        return backend

    def payload(self, backend, **options):
        request = validate_request(dict(prompt="x", max_tokens=8) | options, "completion", "crucible")
        job = backend.submit([1], request)
        while job.events.get(timeout=5)["event"] not in ("done", "error"): pass
        return job.payload

    def test_speculative_requests_carry_lookup(self):
        backend = self.backend(lookup=7)
        self.assertEqual(LOOKUP_MIN, 8)
        payload = self.payload(backend)
        self.assertEqual((payload["lookup"], payload["lookup_min"]), (7, 8))

    def test_plain_requests_and_the_off_switch_keep_the_old_payload(self):
        for backend, options in ((self.backend(lookup=7), dict(spec=0)), (self.backend(lookup=7), dict(max_tokens=1)),
                                 (self.backend(lookup=0), {})):
            payload = self.payload(backend, **options)
            self.assertNotIn("lookup", payload)
            self.assertNotIn("lookup_min", payload)

    def test_library_default_is_the_serve_default(self):
        # Bench, soak and gate tools build backends directly; they must measure what production runs.
        payload = self.payload(self.backend())
        self.assertEqual((payload["lookup"], payload["lookup_min"]), (7, 8))
        from server.batch_backend import BatchedEngineBackend
        with patch("server.backend.EngineBackend.__init__", return_value=None) as base:
            BatchedEngineBackend("engine", "checkpoint", 128, streams=2)
        self.assertEqual(base.call_args.kwargs["lookup"], 7)

    def test_invalid_depth_is_refused(self):
        for value in (-1, 8, True, 1.5):
            with self.assertRaises(ValueError):
                EngineBackend.check_lookup(value)


if __name__ == "__main__":
    unittest.main()
