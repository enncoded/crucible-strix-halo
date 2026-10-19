"""serve --draft-blocks FILE: the batch engine's draft head from a block file; the default command is unchanged."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import server.__main__ as startup


def backend_options(argv):
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer"), patch("server.batch_backend.BatchedEngineBackend") as backend, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener), patch.object(startup, "Service"):
        assert startup.main() == 0
    return tuple(backend.call_args.kwargs["engine_options"])


def refused(argv):
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), \
            patch.object(startup, "ModelTokenizer") as load, unittest.TestCase().assertRaises(SystemExit) as error:
        startup.main()
    load.assert_not_called()
    return error.exception.code


class DraftBlocksFlagTest(unittest.TestCase):
    def test_no_flag_sends_nothing(self):
        self.assertNotIn("--draft-blocks", backend_options(["--streams", "4"]))

    def test_the_file_reaches_the_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "head.txt")
            path.write_text("0,1,1937\n")
            options = backend_options(["--streams", "4", "--draft-blocks", str(path)])
        self.assertEqual(options[options.index("--draft-blocks") + 1], str(path))

    def test_bad_paths_are_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("head.txt", tmp, str(Path(tmp, "missing.txt"))):
                self.assertEqual(refused(["--streams", "4", "--draft-blocks", value]), 2)
            path = Path(tmp, "head.txt")
            path.write_text("0,1\n")
            self.assertEqual(refused(["--serial", "--draft-blocks", str(path)]), 2)

    def test_empty_and_oversized_files_are_refused_before_loading(self):
        # the engine's limit is 64 KiB (kDraftBlocksFileMax); the full selection is about 10 KB.
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("empty.txt", ""), ("big.txt", "1" + " " * 65536)):
                path = Path(tmp, name)
                path.write_text(text)
                self.assertEqual(refused(["--streams", "4", "--draft-blocks", str(path)]), 2)
            edge = Path(tmp, "edge.txt")
            edge.write_text("1" + " " * 65535)
            self.assertIn("--draft-blocks", backend_options(["--streams", "4", "--draft-blocks", str(edge)]))


if __name__ == "__main__":
    unittest.main()
