"""CPU checks for assembly reports. No GPU or optional packages are required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("isa_mix", Path(__file__).parents[1] / "tools/isa_mix.py")
isa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(isa)

ASM = """
foo: ; @foo
  s_mov_b32 s0, 0
.LBB0_1:
  ; a comment
  ;;#ASMSTART
  buffer_load_b32 v0, v1, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  v_dual_add_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5
  v_dot2_f32_f16 v0, v1, v2, v0
  v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]
  ds_read_b32 v0, v1
  s_add_u32 s0, s0, 1
  s_cbranch_scc1 .LBB0_1
  s_endpgm
.Lfunc_end0:
  .size foo, .Lfunc_end0-foo
  .amdhsa_kernel foo
    .amdhsa_group_segment_fixed_size 128
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 40
  .end_amdhsa_kernel
; Kernel info:
; NumVgprs: 80
; NumSgprs: 40
; ScratchSize: 0
; Occupancy: 12
bar:
  flat_load_b32 v0, v[0:1]
  s_load_b32 s0, s[0:1], 0
  scratch_store_b32 off, v0, s0
  s_endpgm
.Lfunc_end1:
  .amdhsa_kernel bar
    .amdhsa_group_segment_fixed_size 0
  .end_amdhsa_kernel
""".splitlines()


class IsaMixTest(unittest.TestCase):
    def test_metadata_after_function_end(self):
        reports = isa.analyze(ASM)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["resources"]["vgpr"], 80)
        self.assertEqual(reports[0]["resources"]["sgpr"], 40)
        self.assertEqual(reports[0]["resources"]["lds_bytes"], 128)
        self.assertEqual(reports[0]["resources"]["occupancy_waves_simd"], 12)

    def test_loop_and_vopd_counts(self):
        report = isa.analyze(ASM)[0]
        loop = report["loops"][0]
        self.assertEqual(loop["instructions"], 8)
        self.assertEqual(loop["operations"], 9)
        self.assertEqual(loop["groups"]["wmma"], 1)
        self.assertEqual(loop["groups"]["dot2"], 1)
        self.assertEqual(loop["groups"]["valu"], 2)
        self.assertEqual(loop["features"]["vopd_pairs"], 1)
        self.assertEqual(loop["features"]["wait_zero"], 1)
        self.assertEqual(loop["waits"][0]["text"], "s_waitcnt vmcnt(0)")

    def test_memory_classification(self):
        report = isa.analyze(ASM)[1]
        self.assertEqual(report["whole"]["groups"]["vmem"], 1)
        self.assertEqual(report["whole"]["groups"]["smem"], 1)
        self.assertEqual(report["whole"]["groups"]["scratch"], 1)
        self.assertEqual(report["loops"], [])

    def test_nested_and_multiple_backedges(self):
        lines = ["kernel:", ".LBB0_1:", " s_nop 0", ".LBB0_2:",
                 " v_add_f32 v0, v0, v1", " s_cbranch_scc1 .LBB0_2",
                 " s_branch .LBB0_1", ".Lfunc_end0:"]
        report = isa.analyze(lines)[0]
        self.assertEqual(len(report["loops"]), 2)
        self.assertEqual(report["loops"][0]["instructions"], 2)
        self.assertEqual(report["loops"][1]["instructions"], 4)

    def test_empty_file(self):
        self.assertEqual(isa.analyze([]), [])

    def test_data_and_yaml_are_not_functions(self):
        prefix = ["global_counter:", "  .long 0"]
        suffix = ["constant_table:", "  .byte 3, 4, 5", ".amdgpu_metadata",
                  "amdhsa.kernels:", "  - .name: foo", "amdhsa.version:",
                  "  - 1", "  - 2", ".end_amdgpu_metadata"]
        self.assertEqual([r["name"] for r in isa.analyze(prefix + ASM + suffix)], ["foo", "bar"])
        self.assertEqual(isa.analyze(prefix + suffix), [])

    def test_instruction_text_after_code_end_is_ignored(self):
        lines = ["not_code:", ".Lfunc_end0:", "  v_add_f32 metadata_only"]
        self.assertEqual(isa.analyze(lines), [])

    def test_comments_directives_and_groups(self):
        for text in ["; v_add_f32", ".amdgpu_metadata", "// s_waitcnt", "v_label:"]:
            self.assertIsNone(isa.instruction(text))
        self.assertEqual(isa.group("s_delay_alu"), "wait")
        self.assertEqual(isa.group("s_barrier"), "control")
        self.assertEqual(isa.group("v_dual_dot2acc_f32_f16"), "dot2")
        self.assertEqual(isa.group("v_mfma_f32"), "wmma")
        self.assertEqual(isa.instruction(" v_add_f32 v0, v1, v2 ; comment"), "v_add_f32 v0, v1, v2")

    def test_zero_store_wait_counter(self):
        body = [" s_waitcnt_vscnt null, 0x0", " s_waitcnt_vscnt null, 0",
                " s_waitcnt_vscnt null, 0x1", " s_waitcnt vmcnt(0) lgkmcnt(0)"]
        report = isa.count_segment(body, 0, len(body))
        self.assertEqual(report["features"]["wait_zero"], 3)

    def test_text_report_with_and_without_loops(self):
        reports = isa.analyze(ASM)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            isa.print_report(reports[0])
            isa.print_report(reports[0], all_loops=True)
            isa.print_report(reports[1])
        self.assertIn("vgpr=80", output.getvalue())
        self.assertIn("backedge lines", output.getvalue())
        self.assertIn("whole function", output.getvalue())

    def test_cli_json_and_first_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.s"
            path.write_text("\n".join(ASM))
            for options, count in [(["--all", "--json"], 2), (["--json"], 1)]:
                output = io.StringIO()
                with patch("sys.argv", ["isa_mix.py", str(path), *options]):
                    with contextlib.redirect_stdout(output):
                        isa.main()
                self.assertEqual(len(json.loads(output.getvalue())), count)
            output = io.StringIO()
            with patch("sys.argv", ["isa_mix.py", str(path), "bar", "--loops"]):
                with contextlib.redirect_stdout(output):
                    isa.main()
            self.assertTrue(output.getvalue().startswith("bar\n"))

    def test_cli_missing_file_and_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.s"
            for exists in [False, True]:
                if exists:
                    path.write_text("\n".join(ASM))
                with patch("sys.argv", ["isa_mix.py", str(path), "missing"]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            isa.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
