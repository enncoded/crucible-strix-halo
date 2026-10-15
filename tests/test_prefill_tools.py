"""CPU checks for fail-closed prefill gate tools. No engine or GPU is used."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PrefillGateToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        (self.root / "fakebin").mkdir()
        gates = Path(os.environ.get("CRUCIBLE_TEST_PL_GATES_SOURCE", ROOT / "tools/pl_gates.sh"))
        shutil.copyfile(gates, self.root / "tools/pl_gates.sh")
        shutil.copyfile(ROOT / "tools/work_paths.sh", self.root / "tools/work_paths.sh")
        self.stub = self.root / "fakebin/flock"
        self.stub.write_text('#!/bin/sh\nrc=${FAKE_GPU_RC:-0}\n[ "$rc" = 0 ] || exit "$rc"\n'
                             'mkdir -p "$FAKE_OUT"\ncp -a "$FAKE_SEED/." "$FAKE_OUT/"\n')
        self.stub.chmod(0o755)
        self.out = self.root / "seed"
        self.output = self.root / "results/test"
        self.env = dict(os.environ, GPU_RUN=str(self.stub), HOME=str(self.root), WORK_DIR=str(self.root / "work"),
                        FAKE_SEED=str(self.out), FAKE_OUT=str(self.output),
                        PATH=str(self.stub.parent) + os.pathsep + os.environ["PATH"])
        prompt = self.root / "work/refenv/out/fibonacci/ids.txt"
        prompt.parent.mkdir(parents=True)
        prompt.write_text(",".join(str(i) for i in range(24)))
        for name in ("p24", "c1", "c8"):
            (self.out / name).mkdir(parents=True)
        for i in range(49):
            for name in ("c1", "c8"):
                with (self.out / name / f"h_{i:02d}.f32").open("wb") as stream:
                    stream.truncate(24 * 10240 * 4)
        self.ids = "ids: " + " ".join(str(i) for i in range(160))
        for name in ("spec0.txt", "spec2.txt"):
            (self.out / name).write_text(self.ids + "\nprefill 1024 tok in 1 s (1024 tok/s)\n")

    def gates(self):
        return subprocess.run(["bash", "tools/pl_gates.sh", "test", "unused"], cwd=self.root,
                              env=self.env, capture_output=True, text=True)

    def test_complete_gate_artifacts_pass(self):
        self.assertEqual(self.gates().returncode, 0)

    def test_reused_tag_fails(self):
        self.output.mkdir(parents=True)
        self.assertNotEqual(self.gates().returncode, 0)

    def test_same_tag_admission_is_atomic(self):
        mkdir = self.stub.parent / "mkdir"
        real_mkdir = shutil.which("mkdir")
        mkdir.write_text('#!/bin/sh\nif [ "$2" = "results/test/p24" ]; then sleep 0.1; fi\n'
                         f'exec "{real_mkdir}" "$@"\n')
        mkdir.chmod(0o755)
        commands = [["bash", "tools/pl_gates.sh", "test", "unused"]] * 2
        processes = [subprocess.Popen(command, cwd=self.root, env=self.env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) for command in commands]
        for process in processes:
            process.communicate(timeout=30)
        self.assertEqual(sorted(process.returncode for process in processes), [0, 2])

    def test_empty_captures_fail(self):
        for name in ("c1", "c8"):
            for file in (self.out / name).glob("h_*.f32"):
                file.write_bytes(b"")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_noncanonical_captures_fail(self):
        for name in ("c1", "c8"):
            (self.out / name / "h_00.f32").rename(self.out / name / "h_99.f32")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_partial_residual_rows_fail(self):
        for name in ("c1", "c8"):
            (self.out / name / "h_00.f32").write_bytes(b"partial")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_missing_prompt_rows_fail(self):
        for name in ("c1", "c8"):
            with (self.out / name / "h_00.f32").open("wb") as stream:
                stream.truncate(23 * 10240 * 4)
        self.assertNotEqual(self.gates().returncode, 0)

    def test_short_row_set_fails(self):
        (self.out / "c1/h_48.f32").unlink()
        self.assertNotEqual(self.gates().returncode, 0)

    def test_extra_row_set_fails(self):
        (self.out / "c8/h_49.f32").write_bytes(b"extra")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_row_mismatch_fails(self):
        (self.out / "c8/h_01.f32").write_bytes(b"different")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_short_id_sequence_fails(self):
        text = "ids: " + " ".join(str(i) for i in range(159)) + "\n1024 tok/s\n"
        for name in ("spec0.txt", "spec2.txt"):
            (self.out / name).write_text(text)
        self.assertNotEqual(self.gates().returncode, 0)

    def test_id_mismatch_fails(self):
        (self.out / "spec2.txt").write_text(self.ids.replace("159", "160") + "\n1024 tok/s\n")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_empty_id_sequence_fails(self):
        for name in ("spec0.txt", "spec2.txt"):
            (self.out / name).write_text("1024 tok/s\n")
        self.assertNotEqual(self.gates().returncode, 0)

    def test_queue_failure_propagates(self):
        self.env["FAKE_GPU_RC"] = "75"
        self.assertEqual(self.gates().returncode, 75)


class PrefillOracleToolTest(unittest.TestCase):
    def oracle(self, values):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline, log = root / "base.txt", root / "oracle.txt"
            baseline.write_text("".join(f"{l} 0.0001 0.0001 0.0001\n" for l in range(48)))
            log.write_text("".join(f"layer {l} pos {p}: relL2 vs fp64 {v}\n"
                                   for (l, p), v in values.items()))
            return subprocess.run(["python3", str(ROOT / "tools/pl_gate1.py"), str(log), str(baseline)],
                                  capture_output=True, text=True)

    @staticmethod
    def values(value=0.0001):
        return {(l, p): value for l in range(48) for p in (0, 14, 23)}

    def test_valid_oracle_passes(self):
        self.assertEqual(self.oracle(self.values()).returncode, 0)

    def test_missing_oracle_point_fails(self):
        values = self.values()
        del values[(0, 0)]
        self.assertNotEqual(self.oracle(values).returncode, 0)

    def test_mean_error_fails(self):
        self.assertNotEqual(self.oracle(self.values(0.000112)).returncode, 0)

    def test_single_point_error_fails(self):
        values = self.values()
        values[(0, 0)] = 0.00021
        self.assertNotEqual(self.oracle(values).returncode, 0)

    def test_negative_error_fails(self):
        values = self.values()
        values[(0, 0)] = -0.0001
        self.assertNotEqual(self.oracle(values).returncode, 0)


if __name__ == "__main__":
    unittest.main()
