import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cache_checks", ROOT / "bench/explore/cache_checks.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class QueueTests(unittest.TestCase):
    def test_run_uses_queue_timeout_and_records_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            done = mock.Mock(returncode=0, stdout="ok\n", stderr="log\n")
            with mock.patch.object(runner.subprocess, "run", return_value=done) as execute:
                self.assertEqual(runner.run(["engine", "serve"], output, {"X": "1"}, "input"), "ok\n")
            command = execute.call_args.args[0]
            self.assertEqual(command[1:], ["-p", "2", "-w", "7200", "--", "timeout", "-k", "10", "300", "engine", "serve"])
            self.assertEqual(command[0], str(runner.work_paths.gpu_run()))
            self.assertEqual(json.loads(output.with_suffix(".command.json").read_text()), command)
            self.assertEqual(execute.call_args.kwargs["input"], "input")
            self.assertEqual(execute.call_args.kwargs["env"], {"X": "1"})

    def test_failed_run_keeps_output_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            done = mock.Mock(returncode=75, stdout="", stderr="queue expired\n")
            with mock.patch.object(runner.subprocess, "run", return_value=done):
                with self.assertRaisesRegex(RuntimeError, "exit 75"):
                    runner.run(["engine"], output)
            self.assertEqual(output.with_suffix(".stderr").read_text(), "queue expired\n")


if __name__ == "__main__":
    unittest.main()
