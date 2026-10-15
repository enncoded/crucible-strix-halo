"""CPU tests: the gen --jobs proof groups captures by load, reverses them and repeats the first as a canary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import capture_once_proof as proof


def record(prefill, n=1):
    return {'command': ['/e', 'gen', '/ckpt', '--ids-file', '/i', '--n', str(n), '--dump', '/old/dump',
                        '--ctx', '32768', '--prefill', str(prefill)],
            'dump_rows': 3, 'output_ids': [7] * n}


class ProofTests(unittest.TestCase):
    def test_groups_reverse_gate_order_per_load_and_add_a_canary(self):
        captures = [('dec-candidate', Path('/d'), 'candidate', record(0)), ('bulk-candidate', Path('/b'), 'candidate', record(4096)),
                    ('dec-serial', Path('/d'), 'serial', record(0, 4)), ('bulk-chunk1', Path('/b'), 'chunk1', record(0))]
        groups = proof.proof_groups(captures)
        self.assertEqual([(key, [m[0] for m in members]) for key, members in groups],
                         [(('32768', '0'), ['bulk-chunk1', 'dec-serial', 'dec-candidate']),
                          (('32768', '4096'), ['bulk-candidate'])])
        with tempfile.TemporaryDirectory() as directory:
            path, jobs = proof.write_group(0, *groups[0], Path(directory))
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([job['label'] for job in jobs],
                             ['bulk-chunk1', 'dec-serial', 'dec-candidate', 'canary-bulk-chunk1'])
            self.assertEqual(jobs[-1]['source'], '/b/chunk1')
            # Only the dump directory changes; every other option is the separate command's.
            self.assertEqual(lines[1]['args'][:6], ['--ids-file', '/i', '--n', '4', '--dump', f'{directory}/dec-serial/dump'])
            self.assertEqual(lines[1]['args'][6:], ['--ctx', '32768', '--prefill', '0'])
            self.assertTrue(Path(lines[3]['args'][5]).is_dir())


if __name__ == '__main__':
    unittest.main()
