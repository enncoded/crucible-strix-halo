"""CPU-only checks for probes that require an external pinned reference."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BUILD = Path(__file__).resolve().parents[1] / "bench" / "build.sh"


class BenchBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bench = self.root / "bench"
        bench.mkdir()
        shutil.copyfile(BUILD, bench / "build.sh")
        for name in ("probe", "isa_hc_regs", "isa_hc_pitch", "isa_hc_bounds"):
            (bench / f"{name}.hip").touch()
        compiler = self.root / "compiler"
        compiler.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$BUILD_LOG"\n')
        compiler.chmod(0o755)
        self.log = self.root / "calls"
        self.env = {**os.environ, "HIPCC": str(compiler), "BUILD_LOG": str(self.log)}

    def run_build(self, *names):
        return subprocess.run(
            ["bash", "bench/build.sh", *names], cwd=self.root,
            env=self.env, capture_output=True, text=True, timeout=10,
        )

    def test_default_excludes_reference_probes(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("bench/probe.hip" in call for call in calls))
        self.assertNotIn("isa_hc_", result.stdout)

    def test_explicit_reference_probe_requires_custom_command(self):
        for name in ("isa_hc_regs", "isa_hc_pitch", "isa_hc_bounds"):
            with self.subTest(name=name):
                result = self.run_build(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ISA_HC_REFERENCE", result.stderr)
                self.assertFalse(self.log.exists())

    def test_explicit_normal_probe_still_builds(self):
        result = self.run_build("probe")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 2)
        self.assertIn("built probe", result.stdout)


if __name__ == "__main__":
    unittest.main()
