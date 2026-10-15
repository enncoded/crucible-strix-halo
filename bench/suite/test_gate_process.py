import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class GateProcessTests(unittest.TestCase):
    def test_process_failures_cannot_pass_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bench/suite").mkdir(parents=True)
            shutil.copy(ROOT / "bench/spec_gates.sh", root / "bench/spec_gates.sh")
            (root / "tools").mkdir()
            shutil.copy(ROOT / "tools/work_paths.sh", root / "tools/work_paths.sh")
            shutil.copy(ROOT / "bench/suite/gate_check.py", root / "bench/suite/gate_check.py")
            baseline = root / "baseline"
            baseline.mkdir()
            for layer in range(49):
                (baseline / f"h_{layer:02d}.f32").write_bytes(b"\0" * 40960)
            engine = root / "engine"
            engine.write_text("#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n"
                              "if os.environ.get('ENGINE_FAIL'): sys.exit(1)\n"
                              "out=Path(sys.argv[sys.argv.index('--dump')+1])\n"
                              "for i in range(49): (out/f'h_{i:02d}.f32').write_bytes(b'\\0'*40960)\n")
            sweep = root / "sweep"
            sweep.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                             "if os.environ.get('SWEEP_FAIL'): sys.exit(1)\n"
                             "if os.environ.get('SWEEP_EMPTY'): sys.exit(0)\n"
                             "for name,status in [('s0','ref'),('s2','exact'),('s4','exact'),"
                             "('copy0','ref'),('copyl7','exact')]: print(f'== {name}: 1 tok/s, hash abc, {status}')\n")
            engine.chmod(0o755)
            sweep.chmod(0o755)
            command = ["bash", str(root / "bench/spec_gates.sh"), "test", str(engine), str(sweep)]
            env = {**os.environ, "GATE_BASE": str(baseline), "GATE_IDS": "fake.ids"}
            for failure in ("ENGINE_FAIL", "SWEEP_FAIL", "SWEEP_EMPTY"):
                with self.subTest(failure=failure):
                    result = subprocess.run(command, env={**env, failure: "1"}, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("Gate 3: all expected", result.stdout)
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Gate 1(a): 49/49", result.stdout)


if __name__ == "__main__":
    unittest.main()
