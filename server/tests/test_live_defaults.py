"""Keep live policy evidence and tier-specific ceiling labels accurate."""
import copy
import unittest
from unittest.mock import patch
from server.tests import live_defaults_test as live


class LiveDefaultEvidenceTests(unittest.TestCase):
    def runs(self):
        return [dict(ids=[1, 2, 3], prefill_seconds=seconds * 2, decode_seconds=seconds,
                     tokens_per_second=2 / seconds) for seconds in (0.1, 0.2, 0.4)]

    def test_policy_summary_keeps_median_and_spread(self):
        runs = self.runs()
        original = copy.deepcopy(runs)
        summary = live.summarize_policy(runs)
        self.assertEqual(summary["tokens_per_second"], 10)
        self.assertEqual(summary["min_tokens_per_second"], 5)
        self.assertEqual(summary["max_tokens_per_second"], 20)
        self.assertEqual(summary["median_prefill_seconds"], 0.4)
        self.assertEqual(summary["median_decode_seconds"], 0.2)
        self.assertEqual(runs, original)

    def test_policy_summary_rejects_mismatched_ids(self):
        runs = self.runs()
        runs[1] = runs[1] | dict(ids=[4])
        with self.assertRaises(ValueError):
            live.summarize_policy(runs)

    def test_policy_summary_rejects_invalid_timings(self):
        for value in (0, -1, float("nan"), float("inf")):
            runs = self.runs()
            runs[0] = runs[0] | dict(tokens_per_second=value)
            with self.assertRaises(ValueError):
                live.summarize_policy(runs)

    def test_policy_measurements_rotate_three_times(self):
        seen = []
        def sample(backend, ids, body):
            seen.append(body.get("spec", 7))
            return self.runs()[0]
        with patch.object(live, "measure", side_effect=sample):
            timings = live.measure_policies(None, [1, 2], {})
        self.assertEqual(seen, [0, 2, 7, 2, 7, 0, 7, 0, 2])
        self.assertTrue(all(len(summary["runs"]) == 3 for summary in timings.values()))

    def test_speed_tier_ceiling_uses_exact_logical_bytes(self):
        timings = {"serial": dict(tokens_per_second=34), "default": dict(tokens_per_second=85)}
        result = live.roofline_summary("3.05", timings)
        self.assertEqual(result["serial_roofline_tokens_per_second"], 245.5e9 / 4_615_344_324)
        self.assertEqual(result["default_code_estimate_percent"], [42.5, 100 * 85 / 150])

    def test_quality_tier_does_not_claim_a_speed_tier_ceiling(self):
        timings = {"serial": dict(tokens_per_second=30), "default": dict(tokens_per_second=80)}
        result = live.roofline_summary("4.05", timings)
        self.assertEqual(result["serial_roofline_tokens_per_second"], 245.5e9 / 5_355_901_444)
        self.assertIsNone(result["estimated_code_roofline_tokens_per_second"])
        self.assertIsNone(result["default_code_estimate_percent"])


if __name__ == "__main__": unittest.main()
