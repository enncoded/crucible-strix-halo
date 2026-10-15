import math
import unittest
from bench.exl3_prefill_ceiling import estimate, parse_cycles


class CeilingTest(unittest.TestCase):
    def weights(self):
        return {"305": {"r": {"K": 3, "tiles": 10, "category": "routed"},
                        "s": {"K": 5, "tiles": 1, "category": "shared"},
                        "d": {"K": 5, "tiles": 8, "category": "dense"}},
                "405": {"r": {"K": 4, "tiles": 10, "category": "routed"},
                        "s": {"K": 6, "tiles": 1, "category": "shared"},
                        "d": {"K": 6, "tiles": 8, "category": "dense"}}}

    def test_equal_cost_matches_published_floor(self):
        r = estimate(self.weights(), {3: 50, 4: 50, 5: 50, 6: 50})
        self.assertAlmostEqual(r["405"]["total_ms"], 1773)
        self.assertAlmostEqual(r["405"]["ceiling_tps"], 4096/1.773)

    def test_doubled_decoder_does_not_double_other_floors(self):
        r = estimate(self.weights(), {3: 50, 4: 100, 5: 50, 6: 100})
        self.assertAlmostEqual(r["405"]["decode_ms"], 360)
        self.assertAlmostEqual(r["405"]["total_ms"], 1953)

    def test_reject_changed_matrix_shape(self):
        w = self.weights(); w["405"]["d"]["tiles"] = 9
        with self.assertRaises(ValueError): estimate(w, {3: 50, 4: 50, 5: 50, 6: 50})

    def test_reject_missing_or_invalid_costs(self):
        for costs in [{3: 50}, {3: math.nan, 4: 50, 5: 50, 6: 50},
                      {3: 50, 4: 50, 5: 50, 6: 0}]:
            with self.assertRaises(ValueError): estimate(self.weights(), costs)

    def test_parse_four_widths(self):
        lines = "\n".join(f"decode K{k} LDS 1000 ms 20 ns/SIMD-step {k*10} cycles/SIMD-step" for k in (3,4,5,6))
        self.assertEqual(parse_cycles(lines), {3:30, 4:40, 5:50, 6:60})
        with self.assertRaises(ValueError): parse_cycles(lines + "\n" + lines)


if __name__ == "__main__": unittest.main()
