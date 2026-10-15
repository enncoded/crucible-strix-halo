"""Check the production H allocation policy without a HIP context."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PrefillHDefaultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        source = (ROOT / "engine/runtime/prefill_engine.inc").read_text()
        start = source.index('  const char* pad_h=crucible::SwitchText("CRUCIBLE_PF_PAD_H");')
        end = source.index("  // Fragment-order copies", start)
        # Compile the exact production block, so a copied policy cannot hide a regression.
        program = f'#include "{ROOT / "engine/switches.h"}"\n' + """
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstddef>
constexpr int kWide=10240;
size_t cells=0;
float storage=0;
template<class T> T* Zeros(size_t count) { cells=count; return &storage; }
template<class T> T* Scratch(size_t count) { return Zeros<T>(count); }
int main() {
  int prefill_rows_=4096, pf_launch_rows_=4096, pf_hstride_=kWide;
  float* pf_H_=nullptr;
""" + source[start:end] + """
  std::printf("%d %zu %d\\n",pf_hstride_,cells,pf_H_!=nullptr);
}
"""
        (root / "policy.cpp").write_text(program)
        cls.binary = root / "policy"
        subprocess.run([*shlex.split(os.environ.get("CXX", "g++")), "-std=c++20",
                        "-O1", "-g", "-fsanitize=address,undefined", str(root / "policy.cpp"),
                        "-o", str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check_policy(self, value, expected):
        environment = {key: item for key, item in os.environ.items() if key != "CRUCIBLE_PF_PAD_H"}
        if value is not None:
            environment["CRUCIBLE_PF_PAD_H"] = value
        result = subprocess.run([str(self.binary)], env=environment, check=True,
                                capture_output=True, text=True)
        self.assertEqual(tuple(map(int, result.stdout.split())), expected)

    def test_unset_pads(self):
        self.check_policy(None, (10304, 4096 * 10304, 1))

    def test_one_pads(self):
        self.check_policy("1", (10304, 4096 * 10304, 1))

    def test_zero_restores_tight(self):
        self.check_policy("0", (10240, 0, 0))

    def test_other_values_keep_legacy_policy(self):
        for value in ("", "false", "01"):
            with self.subTest(value=value):
                self.check_policy(value, (10240, 0, 0))


if __name__ == "__main__":
    unittest.main()
