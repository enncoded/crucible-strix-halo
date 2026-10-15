import json
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from bench import batch_serve_live as live
from server.backend import Job


class LiveHarnessTest(unittest.TestCase):
    def test_cancelled_slow_job_cannot_prove_backpressure(self):
        slow = Job({})
        slow.events.put(dict(event="token", id=2)); slow.events.put(dict(event="token", id=3))
        slow.cancelled.set()
        backend = Mock(fatal=None, active={1: SimpleNamespace(job=slow, generation=1, count=2)})
        backend.submit.return_value = slow
        requests = [dict(prompt=[1], spec=0, max_tokens=8, seed=40, return_token_ids=True)] * 8
        with patch.object(live, "complete", return_value=([2,3,4], .01)):
            with self.assertRaises(ValueError): live.batch_checks(backend, requests, [[1]] * 8)

    def fake_backend(self):
        backend = Mock(fatal=None, active={})
        backend.jobs.empty.return_value = True
        slow = Job({}, events=queue.Queue(maxsize=4))
        for _ in range(2): slow.events.put(dict(event="token", id=40, slot_generation=2))
        original = slow.cancelled.set
        def cancel(): original(); backend.active.clear()
        slow.cancelled.set = Mock(side_effect=cancel)
        recycled = Job({})
        for _ in range(8): recycled.events.put(dict(event="token", id=40, slot_generation=3))
        recycled.events.put(dict(event="done", completion_tokens=8))
        def submit(*args):
            if backend.submit.call_count == 1:
                backend.active[900] = SimpleNamespace(job=slow, generation=2, count=2, ready=True)
                return slow
            return recycled
        backend.submit.side_effect = submit
        return backend, slow

    def run_case(self, root, mode, comma=False):
        prompts = []
        for row in range(8):
            path = root / f"prompt{row}"; path.write_text(f"{row+1},{row+2}\n" if comma else str(row+1)); prompts.append(str(path))
        paths = root / "paths"; paths.write_text("\n".join(prompts) + "\n")
        output = root / "report.json"
        backend, slow = self.fake_backend()
        serial = Mock(); ordinary = False
        def listen(value, tokenizer):
            nonlocal ordinary
            ordinary = value is serial
            return Mock(), Mock()
        def complete(payload):
            if mode == "peer_timeout" and backend.submit.call_count == 1: slow.cancelled.set()
            return [payload["seed"] + int(ordinary and mode == "mismatch")] * 8, .01
        argv = ["batch_serve_live", "unused", "--engine", "unused", "--prompts", str(paths), "--output", str(output)]
        with patch.object(sys, "argv", argv), patch.object(live, "ModelTokenizer", return_value=Mock()), \
             patch.object(live, "BatchedEngineBackend", return_value=backend), \
             patch.object(live, "EngineBackend", return_value=serial), \
             patch.object(live, "listener", side_effect=listen), patch.object(live, "complete", side_effect=complete):
            if mode != "success":
                with self.assertRaises(ValueError): live.main()
            else: live.main()
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "PASS" if mode == "success" else "FAIL")
        backend.close.assert_called_once(); slow.cancelled.set.assert_called_once()
        if mode != "peer_timeout": serial.close.assert_called_once()
        if mode == "success": self.assertEqual(report["batch"]["recycled_generation"], 3)

    def test_comma_separated_prompt_files(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_case(Path(temp), "success", comma=True)

    def test_prompt_separator_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids"
            for text in ("1,2,3\n", "1 2\n3\n", "1, 2\n3\n"):
                path.write_text(text)
                self.assertEqual(live.read_prompt_ids(path), [1, 2, 3])

    def test_invalid_prompt_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids"
            for text in ("", " ", "1,,2", "1, ", "+3", "-1", "1_0", "١", ",1"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    live.read_prompt_ids(path)

    def test_success_mismatch_and_timeout_during_peer(self):
        for mode in ("success", "mismatch", "peer_timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                self.run_case(Path(temp), mode)

if __name__ == "__main__": unittest.main()
