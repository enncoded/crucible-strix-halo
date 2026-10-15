import unittest
from rooflines import annotate, spec_floor


class RooflineTests(unittest.TestCase):
    def test_spec2_uses_routed_union_and_full_head(self):
        row = {"spec": 2, "forwards": 10, "proposed": 20, "output_ids": list(range(21)),
               "decode_seconds": 0.5, "decode_tps": 40}
        result = spec_floor(row)
        expected_ms = (3.726 + 22.4 * 0.0894 + 2 * 0.500118692) / 245.5 * 1000
        self.assertAlmostEqual(result["step_floor_ms"], expected_ms)
        self.assertAlmostEqual(result["ceiling_tps"], 2 / (expected_ms / 1000))
        self.assertEqual(result["rows"], 3)

    def test_mixed_depth_does_not_get_fixed_depth_model(self):
        self.assertIsNone(spec_floor({"spec": 2, "forwards": 10, "proposed": 19}))

    def test_annotation_preserves_measurements_and_sets_both_prefill_ceilings(self):
        row = {"spec": 2, "forwards": 1, "proposed": 2, "output_ids": [1, 2, 3],
               "decode_seconds": 0.05, "decode_tps": 40, "prefill_tps": 1155}
        report = {"checkpoint": "/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5",
                  "ceilings": {"prefill_tps": 2310, "spec_tps": None}, "rows": [row]}
        result = annotate(report)
        self.assertEqual(result["rows"][0]["prefill_ceiling_pct"], 50)
        self.assertAlmostEqual(result["rows"][0]["prefill_fused_ceiling_pct"], 100 * 1155 / 3866)
        self.assertGreater(result["rows"][0]["decode_ceiling_pct"], 0)
        self.assertNotIn("spec_weight_model", row)
        self.assertEqual(result["rows"][0]["decode_seconds"], row["decode_seconds"])

    def test_unknown_checkpoint_has_no_spec_model(self):
        row = {"spec": 2, "decode_ceiling_pct": None}
        result = annotate({"checkpoint": "/models/other", "ceilings": {}, "rows": [row]})
        self.assertIsNone(result["rows"][0]["decode_ceiling_pct"])


if __name__ == "__main__":
    unittest.main()
