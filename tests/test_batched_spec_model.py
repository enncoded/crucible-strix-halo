import unittest
from bench.batched_spec_model import DRAM_BW, UNION_B8, union, plan_rate

class BatchedSpecModelTests(unittest.TestCase):
    def test_measured_independent_anchors(self):
        self.assertAlmostEqual(union(1, 10), 10)
        self.assertAlmostEqual(union(8, 10), UNION_B8)

    def test_current_b8_floor(self):
        result = plan_rate(8, 1, 0.5, 10)
        self.assertAlmostEqual(result.floor_ms, 45.689210944, places=5)
        self.assertAlmostEqual(result.aggregate_ceiling, 8000 / result.floor_ms)
        self.assertAlmostEqual(result.expected_60, 0.6 * result.aggregate_ceiling)

    def test_serial_draft_cannot_be_free(self):
        shared = plan_rate(4, 4, 0.95, 26.538)
        serial = plan_rate(4, 4, 0.95, 26.538, draft_shared=False)
        self.assertGreater(serial.floor_ms, shared.floor_ms)
        self.assertLess(serial.expected_60, shared.expected_60)

    def test_bounds_reject_before_calculation(self):
        for args in [(4, 8, 0.95, 51.25), (2, 0, 0.5, 10), (True, 4, 0.5, 26.16),
                     (2, 4, float('nan'), 26.16), (2, 4, 1.01, 26.16), (2, 4, 0.5, 41)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_rate(*args)

if __name__ == '__main__':
    unittest.main()
