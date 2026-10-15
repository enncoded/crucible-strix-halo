"""the engine CLI `main` sits under the ENGINE_LIBRARY guard.

Bench programs define ENGINE_LIBRARY before they include engine/engine.hip. The seven programs below
include another bench program (not the engine), so they keep `#define main` for now.
"""
import importlib.util
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "engine/cli/gen_cli.h"
MAIN = re.compile(r"^[ \t]*#[ \t]*define[ \t]+main\b", re.M)
PROGRAMS = {".hip", ".cpp", ".cu", ".cc"}
SKIPPED_TOPS = {"engine", "build", "node_modules", ".git"}
# They include another bench program, not the engine (S2).
ALLOWED = {
    "bench/exl3_gemm_cycles_widths.hip",
    "bench/mega_ablate.hip",
    "bench/mega_ablate2.hip",
    "bench/moe_body_window.hip",
    "bench/moe_wmma_count.hip",
    "bench/prefill_moe_loader.hip",
    "tests/prefill_stride_test.hip",
}
# Harnesses that a tool builds against a selected engine tree (CRUCIBLE_ENGINE_HEADER or _SOURCE, for A/B runs
# against older trains). A pre-train-20 gen_cli.h has an unguarded `int main`, so they keep the rename.
# bench/suite/run.py, tools/build_long_probe.py, tools/build_resident_probe.py and qsa_engine_long.hip select the tree.
CROSS_SOURCE = {
    "bench/suite/resident.hip",
    "bench/long_probe.hip",
    "bench/resident_probe.hip",
    "bench/qsa_engine_long.hip",
    "bench/spec_sweep.hip",
    "bench/compact_transition_gate.hip",
}
ALLOWED |= CROSS_SOURCE
OLD_ENGINE = "int main(int, char**) { return 99; }\n"  # train 19 gen_cli.h: main without a guard
NEW_ENGINE = ("#ifdef ENGINE_LIBRARY\n#define ENGINE_CLI_ENTRY CrucibleCliMain\n#else\n#define ENGINE_CLI_ENTRY main\n#endif\n"
              "int ENGINE_CLI_ENTRY(int, char**) { return 99; }\n")


class EngineGuardTests(unittest.TestCase):
    def test_cli_entry_is_named_and_main_is_guarded(self):
        text = CLI.read_text()
        self.assertRegex(text, r"(?m)^#ifdef ENGINE_LIBRARY\n#define ENGINE_CLI_ENTRY CrucibleCliMain\n#else\n"
                               r"#define ENGINE_CLI_ENTRY main\n#endif\nint ENGINE_CLI_ENTRY\(int argc, char\*\* argv\) \{")
        self.assertEqual(re.findall(r"\bint main\(", text), [])  # main exists only through the macro

    def test_only_allowed_programs_define_main(self):
        found = set()
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if rel.parts[0] in SKIPPED_TOPS or path.suffix not in PROGRAMS or not path.is_file():
                continue
            if MAIN.search(path.read_text(errors="replace")):
                found.add(rel.as_posix())
        self.assertEqual(found - ALLOWED, set(), "use #define ENGINE_LIBRARY instead of #define main")
        self.assertLessEqual(len(found), 7 + len(CROSS_SOURCE))

    def test_cross_source_harness_builds_against_old_and_new_engine(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ unavailable")
        for rel in sorted(CROSS_SOURCE):
            text = (ROOT / rel).read_text()
            self.assertRegex(text, r"(?m)^#define ENGINE_LIBRARY\n#define main CrucibleCliMain\b", rel)
            self.assertRegex(text, r"(?m)^#undef main$", rel)
        sweep = ROOT / "bench/spec_sweep.hip"
        spec = importlib.util.spec_from_file_location("spec_sweep_fake", ROOT / "tests/test_spec_sweep_exit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake = module.FAKE
        for name, engine in (("old", OLD_ENGINE), ("new", NEW_ENGINE)):
            with tempfile.TemporaryDirectory() as directory:
                header = Path(directory) / "fake.h"
                header.write_text(fake + engine)
                result = subprocess.run([compiler, "-std=c++20", "-x", "c++", f'-DCRUCIBLE_ENGINE_HEADER="{header}"',
                                         str(sweep), "-o", str(Path(directory) / "sweep")],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{name} engine: {result.stderr[-400:]}")


if __name__ == "__main__":
    unittest.main()
