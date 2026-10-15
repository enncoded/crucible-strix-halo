import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("moe_profile_summary", ROOT / "tools/moe_profile_summary.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(dispatch=1, counter="OccupancyPercent", value="40", **changes):
    result = {
        "Dispatch_Id": str(dispatch), "Kernel_Name": "void crucible::PrefillMoeFxKernel<6, 5>(...)",
        "Counter_Name": counter, "Counter_Value": value,
        "Start_Timestamp": "100", "End_Timestamp": "1000100",
        "VGPR_Count": "216", "Scratch_Size": "0", "Workgroup_Size": "128", "Grid_Size": "262144",
    }
    return {**result, **changes}


class SummaryTest(unittest.TestCase):
    def test_complete_distinct_counters_have_one_time_per_dispatch(self):
        source = [row(i, c, v) for i in (1, 2) for c, v in (("OccupancyPercent", "40"), ("FETCH_SIZE", "2048"))]
        result = MODULE.summarize(source, expected=2, rows=4096)[0]
        self.assertEqual(result["dispatches"], 2)
        self.assertAlmostEqual(result["milliseconds"], 2.0)
        self.assertEqual(result["counters"]["OccupancyPercent"], 40.0)
        self.assertAlmostEqual(result["external_read_gb_s"], 4.194304)
        self.assertEqual(result["scratch_bytes"], 0)
        self.assertGreater(result["wmma_roofline_percent"], 0)

    def test_write_size_uses_calibrated_twofold_byte_factor(self):
        result = MODULE.summarize([row(counter="WRITE_SIZE", value="2048")], expected=1, rows=4096)[0]
        self.assertAlmostEqual(result["external_write_gb_s"], 4.194304)

    def test_same_kernel_with_two_grids_stays_separate(self):
        source = [row(1, value="40", Grid_Size="128"), row(2, value="60", Grid_Size="256")]
        result = MODULE.summarize(source, expected=1, rows=4096)
        self.assertEqual([r["grid_size"] for r in result], [128, 256])
        self.assertEqual([r["counters"]["OccupancyPercent"] for r in result], [40.0, 60.0])

    def test_one_dispatch_cannot_split_its_counter_geometry(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            MODULE.summarize([row(Grid_Size="128"), row(counter="FETCH_SIZE", Grid_Size="256")], expected=1)

    def test_global_dispatch_id_cannot_alias_two_kernel_names(self):
        source = [row(i) for i in range(1, 49)] + [
            row(i, counter="FETCH_SIZE", Kernel_Name="void crucible::PrefillMoeGemmKernel<6, true, 5, true>(...)")
            for i in range(1, 49)]
        with self.assertRaisesRegex(ValueError, "dispatch identity"):
            MODULE.summarize(source)

    def test_different_kernels_keep_independent_counter_sets(self):
        source = [row(1), row(2, counter="FETCH_SIZE",
            Kernel_Name="void crucible::PrefillMoeGemmKernel<6, true, 5, true>(...)")]
        self.assertEqual(len(MODULE.summarize(source, expected=1)), 2)

    def test_down_work_is_half_gateup(self):
        gate = MODULE.summarize([row()], expected=1, rows=4096)[0]
        down = MODULE.summarize([row(Kernel_Name="void crucible::PrefillMoeGemmKernel<6, true, 5, true>(...)")], expected=1, rows=4096)[0]
        self.assertAlmostEqual(gate["useful_tflops"], 2 * down["useful_tflops"])

    def test_bad_values_fail(self):
        for value in ("nan", "inf", "-1", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.summarize([row(value=value)], expected=1, rows=4096)

    def test_duplicate_counter_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.summarize([row(), row()], expected=1, rows=4096)

    def test_incomplete_capture_fails(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            MODULE.summarize([row()], expected=48, rows=4096)

    def test_counter_dispatch_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "counter set"):
            MODULE.summarize([row(1), row(2), row(1, "FETCH_SIZE")], expected=2, rows=4096)

    def test_conflicting_metadata_fails(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            MODULE.summarize([row(), row(counter="FETCH_SIZE", End_Timestamp="2000100")], expected=1, rows=4096)

    def test_invalid_duration_fails(self):
        with self.assertRaises(ValueError):
            MODULE.summarize([row(End_Timestamp="100")], expected=1, rows=4096)

    def test_unknown_kernel_fails(self):
        with self.assertRaisesRegex(ValueError, "kernel"):
            MODULE.summarize([row(Kernel_Name="unrelated")], expected=1, rows=4096)

    def test_zero_clock_is_marked_unusable(self):
        result = MODULE.summarize([row(counter="GRBM_GUI_ACTIVE", value="0")], expected=1, rows=4096)[0]
        self.assertFalse(result["issue_clocks_valid"])

    def test_valu_issue_ratio_is_not_named_utilization(self):
        source = [row(counter="GRBM_GUI_ACTIVE", value="10"), row(counter="SQ_INSTS_VALU", value="160")]
        result = MODULE.summarize(source, expected=1, rows=4096)[0]
        self.assertAlmostEqual(result["valu_instructions_per_simd_cycle"], 0.2)

    def test_empty_capture_and_bad_shape_fail(self):
        for source, expected, rows in (([], 48, 4096), ([row()], 0, 4096), ([row()], 1, 0)):
            with self.assertRaises(ValueError):
                MODULE.summarize(source, expected=expected, rows=rows)

    def test_metadata_values_fail(self):
        for changes in ({"Dispatch_Id": "-1"}, {"Start_Timestamp": "-1"},
                        {"Scratch_Size": "-1"}, {"Counter_Name": ""}, {"Workgroup_Size": "320"},
                        {"Grid_Size": "0"}, {"Grid_Size": "-1"}, {"Grid_Size": "129"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MODULE.summarize([row(**changes)], expected=1, rows=4096)

    def test_cli_includes_hash_and_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counters.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row().keys())
                writer.writeheader()
                writer.writerow(row())
            output = io.StringIO()
            with patch.object(sys, "argv", ["summary", "--expected", "1", str(path)]), patch("sys.stdout", output):
                MODULE.main()
            result = json.loads(output.getvalue())
            self.assertEqual(len(result["files"][0]["sha256"]), 64)
            self.assertIn("MALL", result["warning"])
            self.assertEqual(result["size_metric_byte_factor"], 2.0)

    def test_cli_missing_file_has_actionable_error(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["summary", "/not/a/counter/capture.csv"]), patch("sys.stderr", output):
            with self.assertRaises(SystemExit) as error:
                MODULE.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("counter capture error:", output.getvalue())

    def test_input_rows_are_not_changed(self):
        source = [row()]
        before = [dict(r) for r in source]
        MODULE.summarize(source, expected=1, rows=4096)
        self.assertEqual(before, source)


if __name__ == "__main__":
    unittest.main()
