"""Check the production MoE build policy without HIP or a GPU."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
POLICY = 'engine/layers/moe_window_policy.h'


class MoeWindowPolicyTest(unittest.TestCase):
    def preprocess(self, value=None):
        command = ['c++', '-E', '-P', '-x', 'c++', '-I', str(ROOT)]
        if value is not None:
            command.append(f'-DCRUCIBLE_MOE_WINDOW_PROBE={value}')
        return subprocess.run(
            [*command, '-'], input=f'#include "{POLICY}"\nCRUCIBLE_MOE_WINDOW_PROBE\n',
            text=True, capture_output=True, timeout=10, check=False)

    def test_default_uses_live_window(self):
        result = self.preprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '1')

    def test_zero_preserves_original_body(self):
        result = self.preprocess('0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '0')

    def test_explicit_one_uses_live_window(self):
        result = self.preprocess('1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '1')

    def test_other_integer_values_fail(self):
        for value in ('-1', '2', '17'):
            with self.subTest(value=value):
                result = self.preprocess(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('must be zero or one', result.stderr)


if __name__ == '__main__':
    unittest.main()
