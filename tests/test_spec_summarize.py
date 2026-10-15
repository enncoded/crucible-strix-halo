import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("spec_summarize", Path(__file__).parents[1] / "tools/spec_summarize.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

LOG = """== code.serial: 34.00 tok/s, 159 steps, 1.000 tok/step, accept 0.0%, hash 0123456789abcdef, ref
ids code.serial: 7 8 9
steps 159: wall 29.412 ms/step, host gap 0.010 ms/step
  verify    R=1 calls   159  gpu   29.402 ms/call  wall   29.410 ms/call  29.402 ms/step
  experts trunk: R=1 10.00/call (7632 calls)
  experts mtp:
== code.adaptq: 80.00 tok/s, 40 steps, 3.975 tok/step, accept 80.0%, hash 0123456789abcdef, exact
ids code.adaptq: 7 8 9
steps 40: wall 49.688 ms/step, host gap 0.030 ms/step
  verify    R=5 calls    40  gpu   45.000 ms/call  wall   45.010 ms/call  45.000 ms/step
  experts trunk: R=5 32.00/call (1920 calls)
"""


class TestSummary(unittest.TestCase):
    def test_rates_and_roofline(self):
        rows = module.summarize(LOG, "3.05")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["ids"], [7, 8, 9])
        self.assertEqual(rows[1]["verdict"], "exact")
        expected = 100 * (4_615_337_572 / 245.5e9 * 1000) / 29.402
        self.assertAlmostEqual(rows[0]["verify_weight_roof_percent"], expected)

    def test_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "MISMATCH"):
            module.summarize(LOG.replace(", exact", ", MISMATCH"), "3.05")

    def test_empty_fails(self):
        with self.assertRaises(ValueError):
            module.summarize("", "3.05")

    def test_unknown_checkpoint_fails(self):
        with self.assertRaises(ValueError):
            module.summarize(LOG, "unknown")

    def test_head_policy(self):
        text = LOG + "  head policy: restricted 36 steps, full 4 steps, final restricted\n"
        row = module.summarize(text, "3.05")[-1]
        self.assertEqual(row["head_policy"], {"restricted_steps": 36, "full_steps": 4,
                                               "final": "restricted"})

    def test_request_timings(self):
        text = LOG + "timings code.adaptq: prefill 150.250 ms, decode 1987.500 ms, request 2142.750 ms\n"
        rows = module.summarize(text, "3.05")
        self.assertEqual(rows[-1]["prefill_ms"], 150.25)
        self.assertEqual(rows[-1]["decode_ms"], 1987.5)
        self.assertEqual(rows[-1]["request_ms"], 2142.75)
        self.assertIsNone(rows[0]["request_ms"])

    def test_case_eviction_counter(self):
        text = LOG + "counters code.adaptq pid=123 compact_stall=4 evicted_ms=19\n"
        rows = module.summarize(text, "3.05")
        self.assertEqual(rows[-1]["evicted_ms"], 19)
        self.assertIsNone(rows[0]["evicted_ms"])

    def test_no_counts_has_no_roofline(self):
        rows = module.summarize(LOG.split("  experts trunk:")[0], "4.05")
        self.assertIsNone(rows[0]["verify_weight_roof_percent"])


if __name__ == "__main__":
    unittest.main()
