"""Short prompt chunks run without interleaved decode passes. CPU only."""
import sys
import unittest
from unittest.mock import MagicMock, patch

from server import __main__ as startup
from server.batch_backend import BatchedEngineBackend
from server.batch_scheduler import INTERLEAVE_MIN_ROWS
from server.tests.test_sched_l2 import backend_with, opened


def ready_peer(min_rows):
    backend = backend_with((1, 100_000, 100), (2, 10, 100))
    backend.interleave_min_rows = min_rows
    opened(backend, 1, 0); opened(backend, 2)
    backend.active[2].ready, backend.active[2].count, backend.active[2].last_step = True, 1, 0.0
    return backend


class ShortChunkTest(unittest.TestCase):
    def test_default_threshold_is_300_rows(self):
        self.assertEqual(INTERLEAVE_MIN_ROWS, 300)

    def test_a_short_chunk_gets_no_decoders(self):
        backend = ready_peer(INTERLEAVE_MIN_ROWS)
        for rows in (9, 84, 299):
            self.assertEqual(backend._interleave_decoders(1, rows), [], rows)

    def test_a_chunk_at_the_threshold_keeps_its_passes(self):
        backend = ready_peer(INTERLEAVE_MIN_ROWS)
        for rows in (300, 4096):
            self.assertEqual(backend._interleave_decoders(1, rows), [2], rows)

    def test_zero_keeps_the_old_rule(self):
        backend = ready_peer(0)
        self.assertEqual(backend._interleave_decoders(1, 8), [])
        for rows in (9, 84, 299, 4096):
            self.assertEqual(backend._interleave_decoders(1, rows), [2], rows)

    def test_a_short_chunk_command_names_no_decoders(self):
        backend = ready_peer(INTERLEAVE_MIN_ROWS)
        backend.prefill_rows = 4096
        backend.active[1].job.payload["ids"] = list(range(84))
        sent = []
        backend._command = lambda payload, rows=None: (sent.append(payload), (0.0, 0.1))[1]
        backend._prefill(1)
        self.assertEqual(sent, [dict(op="prefill", request_id=1)])


class OptionTest(unittest.TestCase):
    def test_the_worker_defaults_to_the_threshold_and_checks_it(self):
        backend = BatchedEngineBackend.__new__(BatchedEngineBackend)
        self.assertEqual(backend.interleave_min_rows, INTERLEAVE_MIN_ROWS)
        for bad in (-1, 4097, 1.5, "512", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                BatchedEngineBackend("unused", "unused", context=128, streams=2, interleave_min_rows=bad)

    def test_cli_default_and_old_behavior_flag(self):
        for extra, expected in (([], INTERLEAVE_MIN_ROWS), (['--interleave-min-rows', '0'], 0),
                                (['--interleave-min-rows', '512'], 512)):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', *extra]), \
                    patch.object(startup, 'ModelTokenizer'), \
                    patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['interleave_min_rows'], expected)


if __name__ == "__main__":
    unittest.main()
