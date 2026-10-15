"""The persistent prototype must copy contractions without arithmetic changes."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('persistent_ops', ROOT / 'tools/generate_persistent_ops.py')
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class PersistentOpsTest(unittest.TestCase):
    def test_generated_header_matches_reference(self):
        self.assertEqual((ROOT / 'engine/layers/persistent_ops.h').read_text(), GEN.render())

    def test_adaptation_changes_only_indices_and_barriers(self):
        original = 'x += a * b; int t = threadIdx.x; int b = blockIdx.x; __syncthreads();'
        self.assertEqual(GEN.adapt(original),
                         'x += a * b; int t = ct.t; int b = task; ComputeBarrier(ct);')

    def test_gdn_binding_changes_require_review(self):
        binding = '\n  const GdnArgs a = GdnRowArgs(args, blockIdx.y);'
        for replacement in ('', binding + binding):
            with self.assertRaises(ValueError):
                GEN.scheduled_body(GEN.OPS.replace(binding, replacement), 'GdnKernel')
        self.assertNotIn('GdnRowArgs', GEN.render())
        self.assertNotIn('blockIdx.y', GEN.render())

    def test_no_physical_block_barrier_in_reference_bodies(self):
        header = GEN.render()
        for token in ('threadIdx.x', 'blockIdx.x', '__syncthreads()', 'BlockSum<256>'):
            self.assertNotIn(token, header)
        self.assertEqual(header.count('PersistentBlockSum('), 5)
        self.assertEqual(header.count('PersistentCoefficient('), 4)


if __name__ == '__main__':
    unittest.main()
