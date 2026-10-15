import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("suite_run", HERE / "run.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class PrefillTests(unittest.TestCase):
    def test_prefill_requests_have_one_target_id_and_no_speculation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.txt"
            prompt = {"name": "long_4096", "ids_file": "ids/long_4096.txt"}
            args = argparse.Namespace(mode="prefill", tokens=1, chunk=8, spec=2, warmup=0, repeat=1)
            expected = suite.requests_file(path, [prompt], args)
            self.assertEqual(expected, [{"name": "long_4096", "spec": 0, "repeat": 0}])
            self.assertEqual(path.read_text().count("\n"), 1)
            self.assertTrue(path.read_text().strip().endswith(" 1 8 0 0"))

    def test_single_mode_budget_accepts_main_16k(self):
        seconds = suite.check_budget([{"tokens": 16384}], 1, 1, 0, 77, modes=1)
        self.assertLess(seconds, 270)
        with self.assertRaisesRegex(ValueError, "budget"):
            suite.check_budget([{"tokens": 65536}], 1, 1, 0, 77, modes=1)

    def test_prefill_comparison_ignores_implementation_option(self):
        row = {"name": "long_4096", "spec": 0, "repeat": 0, "output_ids": [1],
               "prompt_sha256": "same", "prefill_tps": 100, "decode_tps": None, "ttft_seconds": 40}
        a = {"settings": {"mode": "prefill"}, "prefill_rows": 0, "rows": [row]}
        b = {**a, "prefill_rows": 4096, "rows": [{**row, "prefill_tps": 1000, "ttft_seconds": 4}]}
        result = suite.compare(a, b)[0]
        self.assertIsNone(result["decode_change_pct"])
        self.assertEqual(result["prefill_change_pct"], 900)
        self.assertEqual(result["ttft_change_pct"], -90)


if __name__ == "__main__":
    unittest.main()
