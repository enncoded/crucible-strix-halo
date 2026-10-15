"""CPU checks for fail-closed quality-gate reporting, without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_BYTES = 2 * 10240 * 4
FAKE_SWEEP = '''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
mode = os.environ.get("GATE_TEST_MODE", "pass")
if mode == "crash":
    raise SystemExit(1)
size = int(os.environ["GATE_TEST_BYTES"])
for line in Path(sys.argv[2]).read_text().splitlines():
    options = line.split()
    if "--dump" not in options:
        continue
    target = Path(options[options.index("--dump") + 1])
    for layer in range(49):
        if mode == "missing_dump" and layer == 48:
            continue
        with (target / f"h_{layer:02d}.f32").open("wb") as file:
            file.truncate(size)
            if mode == "different_rows" and target.name == "c8":
                file.write(b"x")
if mode != "missing_verdict":
    verdict = "MISMATCH" if mode == "different_ids" else "exact"
    print(f"== s2: 1 tok/s, {verdict}")
'''
MKDIR_BARRIER = '''#!/usr/bin/env python3
import os
from pathlib import Path
import time
import sys
root = Path(os.environ["GATE_BARRIER"])
(root / str(os.getppid())).touch()
end = time.monotonic() + 5
while len(list(root.iterdir())) < 2 and time.monotonic() < end:
    time.sleep(0.01)
os.execv(os.environ["GATE_REAL_MKDIR"], ["mkdir", *sys.argv[1:]])
'''


class QualityGateTests(unittest.TestCase):
    def prepare(self, root, mode="pass", baseline_files=49, tag="test"):
        (root / "bench").mkdir()
        script = root / "bench/quality_gates.sh"
        shutil.copy(ROOT / "bench/quality_gates.sh", script)
        (root / "tools").mkdir()
        shutil.copy(ROOT / "tools/work_paths.sh", root / "tools/work_paths.sh")
        ids = root / "work/refenv/out/fibonacci/ids.txt"
        ids.parent.mkdir(parents=True)
        ids.write_text("1,2\n")
        baseline = root / "baseline"
        baseline.mkdir()
        size = 0 if mode == "empty_all" else CAPTURE_BYTES - 4 if mode == "short_all" else CAPTURE_BYTES
        for layer in range(baseline_files):
            with (baseline / f"h_{layer:02d}.f32").open("wb") as file:
                file.truncate(size)
        sweep = root / "sweep"
        sweep.write_text(FAKE_SWEEP)
        sweep.chmod(0o755)
        command = ["bash", str(script), tag, "unused", str(baseline), str(sweep)]
        env = dict(os.environ, HOME=str(root / "home"), WORK_DIR=str(root / "work"), GATE_TEST_MODE=mode, GATE_TEST_BYTES=str(size))
        return command, env

    def run_gate(self, mode="pass", baseline_files=49, tag="test", stale=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env = self.prepare(root, mode, baseline_files, tag)
            if stale:
                (root / "results/quality-gates-test").mkdir(parents=True)
            return subprocess.run(command, env=env, capture_output=True, text=True)

    def test_complete_gates_pass(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Gate 3: exact", result.stdout)

    def test_bad_baseline_count_fails(self):
        for count in (0, 48, 50):
            with self.subTest(count=count):
                self.assertNotEqual(self.run_gate(baseline_files=count).returncode, 0)

    def test_missing_or_different_results_fail(self):
        for mode in ("missing_dump", "different_rows", "different_ids", "missing_verdict", "crash"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_gate(mode).returncode, 0)

    def test_equal_incomplete_captures_fail(self):
        for mode in ("empty_all", "short_all"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_gate(mode).returncode, 0)

    def test_stale_and_unsafe_tags_fail(self):
        self.assertNotEqual(self.run_gate(stale=True).returncode, 0)
        for tag in ("../escape", "", "two words"):
            with self.subTest(tag=tag):
                self.assertNotEqual(self.run_gate(tag=tag).returncode, 0)

    def test_concurrent_tag_is_reserved_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, env = self.prepare(root)
            (root / "barrier").mkdir()
            (root / "bin").mkdir()
            wrapper = root / "bin/mkdir"
            wrapper.write_text(MKDIR_BARRIER)
            wrapper.chmod(0o755)
            env = dict(env, GATE_BARRIER=str(root / "barrier"), GATE_REAL_MKDIR=shutil.which("mkdir"),
                       PATH=str(root / "bin") + os.pathsep + env["PATH"])
            processes = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True) for _ in range(2)]
            for process in processes:
                process.communicate(timeout=15)
            self.assertEqual(sum(process.returncode == 0 for process in processes), 1)


if __name__ == "__main__":
    unittest.main()
