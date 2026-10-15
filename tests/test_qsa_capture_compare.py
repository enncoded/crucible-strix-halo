import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('capture_compare', Path(__file__).parents[1]/'tools/qsa_capture_compare.py')
cmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cmp)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base, self.candidate = (Path(self.tmp.name)/n for n in ('base', 'candidate'))
        counts = struct.pack('<iii', -1, 512, 512)
        selected = struct.pack('<'+'i'*1536, *(list(range(512))*3))
        for i, root in enumerate((self.base, self.candidate)):
            root.joinpath('capture').mkdir(parents=True)
            root.joinpath('input.sha256').write_text(f'{str(i)*64} binary\n'+f'{"a"*64} prompt\n')
            root.joinpath('logits.f32').write_bytes(struct.pack('<f', 1.)*cmp.VOCAB)
            root.joinpath('run.log').write_text('ids: 12\nprefill 2053 tok in 1.000 s\n')
            for layer in cmp.LAYERS:
                prefix = root/'capture'/f'qsa_{layer}'
                Path(str(prefix)+'.meta').write_text('2050 3\n')
                for suffix, data in {'.idx.f32': bytes(3*640*4), '.pool.f32': bytes(17*4096*4),
                                     '.sel.i32': selected, '.nsel.i32': counts}.items():
                    Path(str(prefix)+suffix).write_bytes(data)

    def test_exact_proof(self):
        result = cmp.run(self.base, self.candidate)
        self.assertEqual(result['status'], 'EXACT')
        self.assertEqual(len(result['captures']), 12)
        self.assertEqual(result['kind'], 'supplementary, not shared gates')

    def test_generated_ids_difference_fails(self):
        self.candidate.joinpath('run.log').write_text('ids: 15\n')
        with self.assertRaisesRegex(ValueError, 'generated IDs differ'):
            cmp.run(self.base, self.candidate)

    def test_generated_ids_truncation_fails(self):
        with self.assertRaisesRegex(ValueError, 'generated IDs count'):
            cmp.run(self.base, self.candidate, 2)

    def test_same_binary_fails(self):
        self.candidate.joinpath('input.sha256').write_text(self.base.joinpath('input.sha256').read_text())
        with self.assertRaisesRegex(ValueError, 'independent binaries'):
            cmp.run(self.base, self.candidate)

    def test_logits_truncation_fails(self):
        self.candidate.joinpath('logits.f32').write_bytes(bytes(4))
        with self.assertRaisesRegex(ValueError, 'artifact size'):
            cmp.run(self.base, self.candidate)

    def test_missing_layer_fails(self):
        self.candidate.joinpath('capture/qsa_47.meta').unlink()
        with self.assertRaisesRegex(ValueError, '12 QSA layers'):
            cmp.run(self.base, self.candidate)

    def test_query_difference_fails(self):
        path = self.candidate/'capture/qsa_3.idx.f32'
        path.write_bytes(struct.pack('<f', 1.)+path.read_bytes()[4:])
        with self.assertRaisesRegex(ValueError, 'byte difference'):
            cmp.run(self.base, self.candidate)

    def test_invalid_selection_fails_even_when_equal(self):
        for root in (self.base, self.candidate):
            path = root/'capture/qsa_3.sel.i32'
            data = bytearray(path.read_bytes())
            struct.pack_into('<i', data, 512*4, 1)
            path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'selected IDs'):
            cmp.run(self.base, self.candidate)

    def test_invalid_dense_count_fails_even_when_equal(self):
        for root in (self.base, self.candidate):
            root.joinpath('capture/qsa_3.nsel.i32').write_bytes(struct.pack('<iii', -1, -1, 512))
        with self.assertRaisesRegex(ValueError, 'selected count'):
            cmp.run(self.base, self.candidate)

    def test_nonfinite_logits_fail_even_when_equal(self):
        for root in (self.base, self.candidate):
            path = root/'logits.f32'
            path.write_bytes(struct.pack('<f', float('nan'))+path.read_bytes()[4:])
        with self.assertRaisesRegex(ValueError, 'nonfinite logits'):
            cmp.run(self.base, self.candidate)

    def test_symlink_logits_fail(self):
        self.candidate.joinpath('logits.f32').unlink()
        self.candidate.joinpath('logits.f32').symlink_to(self.base/'logits.f32')
        with self.assertRaisesRegex(ValueError, 'artifact size or type'):
            cmp.run(self.base, self.candidate)


if __name__ == '__main__':
    unittest.main()
