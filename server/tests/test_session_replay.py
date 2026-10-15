"""G15 session replay: the turn label and the CLI guard."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from server import metrics


class SessionReplayLabel(unittest.TestCase):
    def label(self, event):
        return metrics.finalize(metrics.new_turn(), event)["session_label"]

    def test_labels(self):
        self.assertEqual(self.label(dict(cached_tokens=40, session_replay=True)), "hit_replay")
        self.assertEqual(self.label(dict(cached_tokens=40, session_replay=False)), "hit_session")
        self.assertEqual(self.label(dict(cached_tokens=0, session_replay=True)), "miss")
        self.assertEqual(self.label(dict(cached_tokens=40)), "hit_session")

    def test_batch_flag(self):
        from server.batch_backend import BatchedEngineBackend
        with self.assertRaises(ValueError):
            BatchedEngineBackend("/nonexistent", "/nonexistent", session_replay=True)
        fake = Path(__file__).parent / "fake_batch_engine.py"
        popen, seen = subprocess.Popen, []
        def start(arguments, **kwargs):
            seen.append(arguments)
            return popen([sys.executable, str(fake)], **kwargs)
        for replay in (False, True):
            with patch("server.backend.subprocess.Popen", side_effect=start):
                BatchedEngineBackend("unused", "unused", context=128, streams=2, conversation_cache_mib=64,
                                     session_replay=replay).close()
        self.assertNotIn("--session-replay", seen[0])
        self.assertEqual(seen[1][-2:], ["--session-replay", "1"])

    def startup(self, *extra):
        """Run server.__main__.main with mocked backends; return the backend constructor mocks."""
        from unittest.mock import MagicMock
        from server import __main__ as startup
        import server.batch_backend as batch
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, "argv", ["crucible", "checkpoint", *extra]), patch.object(startup.signal, "signal"), \
                patch.object(startup, "ModelTokenizer"), patch.object(startup, "EngineBackend") as serial, \
                patch.object(batch, "BatchedEngineBackend") as batched, \
                patch.object(startup, "ThreadingHTTPServer", return_value=listener):
            self.assertEqual(startup.main(), 0)
        return serial, batched

    def test_replay_is_the_default_with_a_cache(self):
        # Design decision (Oct 4): replay is the default; --session-replay 0 keeps the fresh-request path.
        serial, _ = self.startup("--serial", "--conversation-cache-mib", "32")
        self.assertEqual(serial.call_args.kwargs["engine_options"], ("--session-replay", "1"))
        serial, _ = self.startup("--serial", "--conversation-cache-mib", "32", "--session-replay", "0")
        self.assertEqual(serial.call_args.kwargs["engine_options"], ())
        serial, _ = self.startup("--serial")
        self.assertEqual(serial.call_args.kwargs["engine_options"], ())
        _, batched = self.startup("--streams", "2", "--conversation-cache-mib", "32")
        self.assertIs(batched.call_args.kwargs["session_replay"], True)
        _, batched = self.startup("--streams", "2", "--conversation-cache-mib", "32", "--session-replay", "0")
        self.assertIs(batched.call_args.kwargs["session_replay"], False)
        _, batched = self.startup("--streams", "2")
        self.assertIs(batched.call_args.kwargs["session_replay"], False)
        # No --streams flag is --streams 1 (the batched stack); replay is the default there too.
        serial, batched = self.startup("--conversation-cache-mib", "32")
        serial.assert_not_called()
        self.assertIs(batched.call_args.kwargs["session_replay"], True)

    def test_cli_needs_cache(self):
        for extra in (["--session-replay", "1"], ["--session-replay", "1", "--streams", "2"]):
            result = subprocess.run([sys.executable, "-m", "server", "/nonexistent", *extra],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--session-replay needs", result.stderr)


if __name__ == "__main__":
    unittest.main()
