import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate_check", HERE / "gate_check.py")
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


class GateTests(unittest.TestCase):
    def test_missing_baseline_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "49"):
                gates.check_dumps(Path(directory), Path(directory), Path(directory))

    def test_mismatch_and_empty_dump_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = [Path(directory) / name for name in ("base", "one", "eight")]
            for path in dirs:
                path.mkdir()
                for layer in range(49):
                    (path / f"h_{layer:02d}.f32").write_bytes(b"\0" * 40960)
            gates.check_dumps(*dirs)
            (dirs[1] / "h_01.f32").write_bytes(b"\1" * 40960)
            with self.assertRaisesRegex(ValueError, "differs"):
                gates.check_dumps(*dirs)
            (dirs[1] / "h_01.f32").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                gates.check_dumps(*dirs)

    def test_truncated_sweep_fails(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            gates.check_sweep("== s0: 34 tok/s, hash abc, ref\n")

    def test_mismatch_sweep_fails(self):
        text = self.sweep().replace("s2: 2 tok/s, hash abc, exact", "s2: 2 tok/s, hash abc, MISMATCH")
        with self.assertRaisesRegex(ValueError, "verdict"):
            gates.check_sweep(text)

    def test_valid_sweep(self):
        gates.check_sweep(self.sweep())

    def test_extra_malformed_summary_fails(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            gates.check_sweep(self.sweep() + "\n== extra_case: invalid result\n")

    @staticmethod
    def sweep():
        return "\n".join(f"== {name}: 2 tok/s, hash abc, {verdict}" for name, verdict in
                         (("s0", "ref"), ("s2", "exact"), ("s4", "exact"),
                          ("copy0", "ref"), ("copyl7", "exact")))


if __name__ == "__main__":
    unittest.main()
