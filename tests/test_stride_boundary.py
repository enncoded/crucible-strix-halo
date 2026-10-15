import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("stride_boundary", Path(__file__).parents[1] / "bench/stride_boundary.py")
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class Process:
    def __init__(self, truncate=False, spec_mismatch=False):
        self.events = [{"event": "ready"}]
        self.stdin = io.StringIO()
        self.stdout = self
        self.position = 0
        self.running = True
        self.truncate = truncate
        self.spec_mismatch = spec_mismatch

    def readline(self):
        if not self.events:
            requests = self.stdin.getvalue()[self.position:].splitlines()
            self.position = len(self.stdin.getvalue())
            request = json.loads(requests[0])
            count = 7 if self.truncate else 8
            shift = request["spec"] if self.spec_mismatch else 0
            self.events = [{"event": "token", "id": request["ids"][0] + i + shift} for i in range(count)]
            self.events.append({"event": "done"})
        return json.dumps(self.events.pop(0)) + "\n"

    def wait(self, timeout):
        self.running = False
        return 0

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.running = False


class BoundaryTests(unittest.TestCase):
    def run_capture(self, process, path):
        with patch.object(boundary.subprocess, "Popen", return_value=process) as launch:
            result = boundary.capture(Path("engine"), Path("checkpoint"), [1] * 4096, [2] * 24, path, True)
        self.assertEqual(launch.call_args.kwargs["env"]["CRUCIBLE_PF_PAD_H"], "1")
        return result

    def test_exact_protocol_and_runtime_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            result = self.run_capture(Process(), path)
            self.assertEqual(result, [list(range(1, 9)), list(range(2, 10))] * 2)
            self.assertEqual(len(json.loads((path / "events.json").read_text())), 37)

    def test_missing_token_fails_and_keeps_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            process = Process(truncate=True)
            with self.assertRaisesRegex(RuntimeError, "eight tokens"):
                self.run_capture(process, path)
            self.assertFalse(process.running)
            self.assertTrue((path / "events.json").exists())

    def test_changed_speculative_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "continuations differ"):
                self.run_capture(Process(spec_mismatch=True), Path(temp))

    def test_engine_eof_keeps_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            process = Process()
            with patch.object(process, "readline", return_value=""):
                with self.assertRaisesRegex(RuntimeError, "stopped unexpectedly"):
                    self.run_capture(process, path)
            self.assertFalse(process.running)
            self.assertEqual(json.loads((path / "events.json").read_text()), [])

    def main_files(self, path, candidate_bytes=b"candidate"):
        arguments = ["stride_boundary"]
        for name, content in (("checkpoint", "unused"), ("baseline", "baseline"),
                              ("candidate", candidate_bytes.decode()),
                              ("large_ids", ",".join(["1"] * 4096)),
                              ("small_ids", ",".join(["2"] * 24))):
            target = path / name
            target.write_text(content)
            arguments.append(str(target))
        return arguments + [str(path / "result")]

    def test_main_records_hashes_and_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            arguments = self.main_files(path)
            with patch.object(sys, "argv", arguments), patch.object(boundary, "capture", return_value=[[1]] * 4) as capture:
                with redirect_stdout(io.StringIO()):
                    boundary.main()
            report = json.loads((path / "result/boundary.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertNotEqual(report["binary_sha256"]["baseline"], report["binary_sha256"]["candidate"])
            self.assertFalse(capture.call_args_list[0].args[-1])
            self.assertTrue(capture.call_args_list[1].args[-1])

    def test_main_mismatch_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            arguments = self.main_files(path)
            with patch.object(sys, "argv", arguments), patch.object(boundary, "capture", side_effect=[[[1]] * 4, [[2]] * 4]):
                with self.assertRaisesRegex(ValueError, "changes continuation"):
                    boundary.main()
            self.assertEqual(json.loads((path / "result/boundary.json").read_text())["status"], "FAIL")

    def test_main_rejects_self_baseline_before_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            arguments = self.main_files(path, b"baseline")
            with patch.object(sys, "argv", arguments), patch.object(boundary, "capture") as capture:
                with self.assertRaisesRegex(ValueError, "different baseline"):
                    boundary.main()
            capture.assert_not_called()
            self.assertFalse((path / "result").exists())

    def test_invalid_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids"
            for text in ("-1", "248320"):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "invalid token"):
                    boundary.ids(path)


if __name__ == "__main__":
    unittest.main()
