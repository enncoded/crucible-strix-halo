import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.statecache_ttft import request, verify_rows, summaries, validate_corpus


class StatecacheTimingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(scenarios=[dict(name="test", turns=[dict(turn=0), dict(turn=1)])])
        self.rows = [dict(scenario="test", turn=turn, mode=mode, first_id=10 + turn,
                          ttft_seconds=2 if mode == "bypass" else 1,
                          prefill_tokens=100 if mode == "bypass" else 20, cached_tokens=0 if mode == "bypass" else 80)
                     for mode in ("bypass", "auto") for turn in (0, 1)]

    def test_corpus_matches_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("tokenizer.json", "chat_template.jinja", "config.json", "quantization_config.json", "model.safetensors.index.json"):
                (root / name).write_text(name)
            from bench.suite.conversation_run import digest
            manifest = dict(tokenizer_sha256=digest(root / "tokenizer.json"),
                            template_sha256=digest(root / "chat_template.jinja"), template_file="chat_template.jinja")
            identity = validate_corpus(root, manifest)
            self.assertEqual(identity["tokenizer_sha256"], manifest["tokenizer_sha256"])
            self.assertEqual(len(identity["metadata_sha256"]), 3)
            for key in ("tokenizer_sha256", "template_sha256"):
                with self.assertRaises(ValueError):
                    validate_corpus(root, manifest | {key: "0" * 64})
            for name in ("../auth.json", "unknown.jinja", None):
                with self.assertRaises(ValueError):
                    validate_corpus(root, manifest | dict(template_file=name))

    def test_every_first_id_matches(self):
        verify_rows(self.rows, self.manifest)
        changed = [row | dict(first_id=99) if row["mode"] == "auto" and row["turn"] == 0 else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, "first token"):
            verify_rows(changed, self.manifest)

    def test_incomplete_and_duplicate_runs_fail(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaises(RuntimeError):
                verify_rows(rows, self.manifest)

    def test_summary_reports_residual_rate_and_roofline(self):
        result = summaries(self.rows, self.manifest, 200)
        self.assertEqual(result[0]["speedup"], 2)
        self.assertEqual(result[0]["cached_tokens"], 80)
        self.assertEqual(result[0]["residual_prefill_tps"], 20)
        self.assertEqual(result[0]["roofline_percent"], 10)
        with self.assertRaises(ValueError):
            summaries(self.rows, self.manifest, 0)

    def test_request_validates_engine_counts(self):
        valid = dict(event="done", completion_tokens=1, prompt_tokens=3, cached_tokens=2, prefill_tokens=1, cache_hit=True)
        for changes in ({"prompt_tokens": 2}, {"cached_tokens": True}, {"cached_tokens": 4},
                        {"prefill_tokens": 3}, {"cache_hit": False}, {"completion_tokens": True},
                        {"completion_tokens": 1.0}, {"prompt_tokens": 3.0}, {"prefill_tokens": 1.0}):
            done = valid | changes
            process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(
                '{"event":"token","id":10}\n' + json.dumps(done) + '\n'))
            with patch("tools.statecache_ttft.time.perf_counter", side_effect=(1., 1.25)):
                with self.assertRaises(RuntimeError):
                    request(process, [1, 2, 3], 2, "auto")
        process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(
            '{"event":"token","id":10}\n' + json.dumps(valid) + '\n'))
        with patch("tools.statecache_ttft.time.perf_counter", side_effect=(1., 1.25)):
            self.assertEqual(request(process, [1, 2, 3], 2, "auto")["ttft_seconds"], .25)


if __name__ == "__main__":
    unittest.main()
