"""serve --mtp-window: the mtp_prefix of requests that set none; 0 is the full MTP window, 256 the default."""
import sys
import unittest
from unittest.mock import MagicMock, patch

import server.__main__ as startup


def service_options(argv):
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer"), patch("server.__main__.EngineBackend"), \
            patch("server.batch_backend.BatchedEngineBackend"), \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
            patch.object(startup, "Service") as service:
        assert startup.main() == 0
    return service.call_args.kwargs


class MtpWindowFlagTest(unittest.TestCase):
    def test_no_flag_gives_the_default_window(self):
        self.assertEqual(service_options([])["default_mtp_prefix"], 256)
        self.assertEqual(service_options(["--streams", "4"])["default_mtp_prefix"], 256)

    def test_the_flag_reaches_the_service(self):
        for value in (0, 4096, 262144):
            self.assertEqual(service_options(["--mtp-window", str(value)])["default_mtp_prefix"], value)
            self.assertEqual(service_options(["--streams", "4", "--mtp-window", str(value)])["default_mtp_prefix"],
                             value)

    def test_batch_engine_sizes_its_slot_mtp_caches_by_the_window(self):
        # the batch engine sizes each slot's MTP cache by the server window (engine MtpSlotRows). The option is
        # sent only when it is not the engine default (256), so the daily command line does not change.
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        for argv, expected in (([], 256), (["--mtp-window", "0"], 0), (["--mtp-window", "4096"], 4096)):
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "4", *argv]), \
                    patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                    patch("server.batch_backend.BatchedEngineBackend") as backend, \
                    patch.object(startup, "ThreadingHTTPServer", return_value=listener), patch.object(startup, "Service"):
                self.assertEqual(startup.main(), 0)
            self.assertEqual(backend.call_args.kwargs["mtp_window"], expected)
        from server.batch_backend import BatchedEngineBackend
        for window, sent in ((256, None), (0, "0"), (4096, "4096")):
            with patch("server.batch_backend.EngineBackend.__init__") as initialize:
                BatchedEngineBackend("unused", "unused", streams=2, mtp_window=window)
            options = initialize.call_args.kwargs["engine_options"]
            self.assertEqual(options[options.index("--mtp-window") + 1] if "--mtp-window" in options else None, sent)
        for bad in (-1, 262145, "0", 1.0):
            with self.assertRaises(ValueError), patch("server.batch_backend.EngineBackend.__init__"):
                BatchedEngineBackend("unused", "unused", streams=2, mtp_window=bad)

    def test_out_of_range_window_is_refused_before_loading(self):
        for value in ("-1", "262145", "x"):
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--mtp-window", value]), \
                    patch.object(startup, "ModelTokenizer") as load, self.assertRaises(SystemExit) as error:
                startup.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
