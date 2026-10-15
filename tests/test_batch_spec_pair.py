import unittest
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from bench.batch_spec_pair_report import summarize

class PairReportTests(unittest.TestCase):
    def test_mirrored_rates_and_contract(self):
        rows = [dict(case=case, tokens=64, seconds=seconds, steps=16) for case, seconds in [('a', .5), ('b', .7), ('c', .8), ('c', .8), ('b', .7), ('a', .5)]]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index, row in enumerate(rows):
                case = root / f'{index}_{row["case"]}'
                case.mkdir()
                for slot in range(2): (case / f'ids_{slot}.i32').write_bytes(struct.pack('<33i', *range(33)))
            result = summarize(rows, root)
            self.assertAlmostEqual(result['a_over_b'], 1.4)
            self.assertAlmostEqual(result['a_over_c'], 1.6)
            self.assertTrue(result['clear_win'])
            close_b = [{**row, 'seconds': .51} if row['case'] == 'b' else row for row in rows]
            self.assertFalse(summarize(close_b, root)['clear_win'])
            one_bad_mirror = [dict(row) for row in rows]
            one_bad_mirror[4] = {**one_bad_mirror[4], 'seconds': .51}
            self.assertFalse(summarize(one_bad_mirror, root)['clear_win'])
            with self.assertRaises(ValueError): summarize(rows[:-1], root)
            with self.assertRaises(ValueError): summarize([{**row, 'seconds': 0} for row in rows], root)
            with self.assertRaises(ValueError): summarize([{**row, 'steps': 0} for row in rows], root)
            target = root / '5_a' / 'ids_1.i32'
            target.write_bytes(struct.pack('<33i', 99, *range(1, 33)))
            with self.assertRaises(ValueError): summarize(rows, root)
            target.unlink()
            with self.assertRaises(ValueError): summarize(rows, root)

if __name__ == '__main__': unittest.main()
