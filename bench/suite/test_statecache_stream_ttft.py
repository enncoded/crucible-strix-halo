import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.statecache_stream_ttft import stream_request, summarize, main


class StreamTimingTests(unittest.TestCase):
    def process(self, tokens=(10, 11, 12), **changes):
        done = dict(event="done", completion_tokens=3, prompt_tokens=4,
                    cached_tokens=0, prefill_tokens=4, cache_hit=False) | changes
        lines = [json.dumps(dict(event="token", id=token)) for token in tokens]
        return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("\n".join(lines + [json.dumps(done)]) + "\n"))

    def test_reads_all_streamed_ids_and_total_time(self):
        process = self.process()
        with patch("tools.statecache_stream_ttft.time.perf_counter", side_effect=(1., 1.25, 2.)):
            row = stream_request(process, (1, 2, 3, 4), 3, "refresh", 2, 3)
        self.assertEqual(row["output_ids"], [10, 11, 12])
        self.assertEqual(row["ttft_seconds"], .25)
        self.assertEqual(row["total_seconds"], 1.)
        self.assertEqual(json.loads(process.stdin.getvalue())["max_tokens"], 3)

    def test_sampled_request_uses_a_fixed_seed(self):
        process = self.process()
        with patch("tools.statecache_stream_ttft.time.perf_counter", side_effect=(1., 1.25, 2.)):
            row = stream_request(process, (1, 2, 3, 4), 3, "auto", 2, 3, sampled=True)
        self.assertEqual(row["sampling"]["seed"], 321)
        payload = json.loads(process.stdin.getvalue())
        self.assertEqual(payload["seed"], 321)
        self.assertEqual(payload["temperature"], .7)

    def test_rejects_invalid_streams_and_counts(self):
        for process in (self.process(tokens=(True, 11, 12)), self.process(tokens=(10, 11)),
                        self.process(completion_tokens=3.0), self.process(prompt_tokens=True),
                        self.process(prefill_tokens=3), self.process(cache_hit=True)):
            with patch("tools.statecache_stream_ttft.time.perf_counter", side_effect=(1., 1.25, 2.)):
                with self.assertRaises(RuntimeError):
                    stream_request(process, (1, 2, 3, 4), 3, "refresh", 2, 3)

    def run_main(self, root, capture):
        engine = root / "engine"; engine.write_text("original")
        fixtures = root / "fixtures"; fixtures.mkdir()
        (fixtures / "manifest.json").write_text(json.dumps(dict(scenarios=[dict(name="4k"), dict(name="16k")])))
        turn = dict(ids=[1, 2, 3, 4], cache_prefix_tokens=3)
        argv = ["runner", str(engine), str(root / "checkpoint"), str(fixtures), str(root / "run.json")]
        with patch("sys.argv", argv), patch("tools.statecache_stream_ttft.load_turns", return_value=[turn]), \
                patch("tools.statecache_stream_ttft.validate_corpus", return_value={}), \
                patch("tools.statecache_stream_ttft.capture", side_effect=capture):
            main()

    def test_refuses_every_stale_output_before_capture(self):
        for filename in ("run.json", "run.stderr.log", "run.proof.json", "run.proof.stderr.log"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); stale = root / filename; stale.write_text("old PASS")
                with self.assertRaises(FileExistsError):
                    self.run_main(root, lambda *args: self.fail("A stale result must stop before capture."))
                self.assertEqual(stale.read_text(), "old PASS")

    def test_engine_change_stops_before_the_proof_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); calls = []
            def capture(*args):
                calls.append(args)
                (root / "engine").write_text("changed")
                return []
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.run_main(root, capture)
            self.assertEqual(len(calls), 1)

    def rows(self):
        return [dict(scenario="4k", spec=0, repeat=0, phase=phase,
                     output_ids=[10, 11, 12], ttft_seconds=1. if phase == "hit" else 2.,
                     total_seconds=2. if phase == "hit" else 3.,
                     prefill_tokens=0 if phase == "hit" else 4096,
                     cached_tokens=4096 if phase == "hit" else 0)
                for phase in ("before", "miss", "hit", "after")]

    def test_summary_checks_full_ids_and_hit(self):
        row = summarize(self.rows(), (("4k", 0, 0),), 2310)[0]
        self.assertEqual(row["miss_ttft_overhead_percent"], 0)
        self.assertEqual(row["hit_cached_tokens"], 4096)
        self.assertEqual(row["hit_speedup"], 2)
        self.assertEqual(row["hit_rate"], 1)
        partial = [r | dict(scenario="resume_greedy", prefill_tokens=1) for r in self.rows()]
        with self.assertRaises(RuntimeError):
            summarize(partial, (("resume_greedy", 0, 0),), 2310)
        for bad in (self.rows()[:-1], self.rows() + self.rows()[:1],
                    [r | dict(output_ids=[99]) if r["phase"] == "hit" else r for r in self.rows()],
                    [r | dict(cached_tokens=0) if r["phase"] == "hit" else r for r in self.rows()],
                    [r | dict(sampling={"seed": 99}) if r["phase"] == "hit" else r for r in self.rows()]):
            with self.assertRaises(RuntimeError):
                summarize(bad, (("4k", 0, 0),), 2310)


if __name__ == "__main__":
    unittest.main()
