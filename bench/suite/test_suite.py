import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("suite_run", HERE / "run.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class SuiteTests(unittest.TestCase):
    def test_corpus_integrity(self):
        corpus = suite.load_corpus(HERE / "manifest.json")
        self.assertEqual(len(corpus["prompts"]), 12)
        self.assertEqual([p["tokens"] for p in corpus["prompts"] if p["category"] == "long"],
                         [4096, 16384, 65536])

    def test_corrupt_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ids.txt").write_text("1,2\n")
            (root / "manifest.json").write_text(json.dumps({"schema": 1, "prompts": [
                {"name": "bad", "tokens": 2, "ids_file": "ids.txt", "sha256": "bad"}]}))
            with self.assertRaisesRegex(ValueError, "hash"):
                suite.load_corpus(root / "manifest.json")

    def test_metrics_exclude_first_decode_token(self):
        row = {"prompt_tokens": 100, "output_ids": [4, 5, 6], "prefill_seconds": 2,
               "decode_seconds": 0.5, "accepted": 1, "proposed": 2, "forwards": 1,
               "spec": 2, "ttft_seconds": 2}
        result = suite.metrics(row, 50, 200, 10)
        self.assertEqual(result["decode_tps"], 4)
        self.assertEqual(result["prefill_tps"], 50)
        self.assertEqual(result["acceptance_pct"], 50)
        self.assertEqual(result["decode_ceiling_pct"], 40)
        self.assertEqual(result["prefill_ceiling_pct"], 25)

    def test_unknown_spec_ceiling_is_null(self):
        row = {"prompt_tokens": 1, "output_ids": [4], "prefill_seconds": 1,
               "decode_seconds": 0, "accepted": 0, "proposed": 0, "forwards": 0,
               "spec": 2, "ttft_seconds": 1}
        result = suite.metrics(row, 53, None, None)
        self.assertIsNone(result["decode_ceiling_pct"])
        self.assertIsNone(result["acceptance_pct"])
        self.assertIsNone(result["decode_tps"])

    def test_missing_and_duplicate_results_fail(self):
        expected = [{"name": "a", "spec": 0, "repeat": 0}]
        with self.assertRaisesRegex(ValueError, "missing"):
            suite.check_rows([], expected)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            suite.check_rows([expected[0], expected[0]], expected)

    def test_mismatched_spec_ids_fail(self):
        rows = [{"name": "a", "spec": 0, "repeat": 0, "output_ids": [1, 2]},
                {"name": "a", "spec": 2, "repeat": 0, "output_ids": [1, 3]}]
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            suite.check_rows(rows, rows)

    def test_estimate_blocks_long_baseline_run(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            suite.check_budget([{"tokens": 65536}], 160, 1, 0, 100)

    def test_comparison_requires_same_requests(self):
        a = {"rows": [{"name": "a", "spec": 0, "repeat": 0}]}
        with self.assertRaisesRegex(ValueError, "request"):
            suite.compare(a, {"rows": []})


if __name__ == "__main__":
    unittest.main()
