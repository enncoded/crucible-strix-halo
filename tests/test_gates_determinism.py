"""CPU tests: run-to-run determinism and negative controls of the gate comparison.
The fake engine writes the same dumps every time. A control flips one bit and the gate must reject it."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g
import test_gates_cache_run as cached

# A module-level name for the fixture class would make pytest collect its 8 tests a second time here.


class DeterminismTests(unittest.TestCase):
    fixture = cached.CachedGateRunTests
    arguments, gate, failed_gate, writable = fixture.arguments, fixture.gate, fixture.failed_gate, fixture.writable

    def fake_engine(self, command, **kwargs):
        result = self.fixture.fake_engine(self, command, **kwargs)
        if len(command) == 1:
            return result
        dump = Path(command[command.index('--dump') + 1])
        name = dump.parent.name
        self.runs[name] = self.runs.get(name, 0) + 1
        for (capture, run), (file, offset) in self.flips.items():
            if capture == name and run == self.runs[name]:
                path = dump / file
                data = bytearray(path.read_bytes())
                data[offset] ^= 0x01
                path.write_bytes(bytes(data))
        return result

    def setUp(self):
        self.fixture.setUp(self)
        self.runs, self.flips = {}, {}

    # Item 3(a): a second capture of the same command must repeat bit for bit.
    def test_twin_check_passes_for_a_deterministic_engine(self):
        report = self.gate('twin', baseline_cache=None)
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['twin_check'])
        self.assertFalse(report['repeat_candidate'])
        self.assertNotIn('candidate_repeat', self.calls)

    def test_twin_check_rejects_a_drift_that_every_other_gate_accepts(self):
        # The same bit flips in chunk1, chunk8, serial and spec2: gates 2 and 3 compare these with each
        # other and pass. The candidate equals the baseline and passes gate 1. Only the twin check
        # (candidate against chunk8, same command) sees the change.
        self.flips = {(n, 1): ('h_30.f32', 7) for n in ('chunk1', 'chunk8', 'serial', 'spec2')}
        report = self.failed_gate('twin-bad', 'twin check candidate vs chunk8: dump content differs: h_30',
                                  baseline_cache=None)
        # The flag records a passed compare, so a failed twin check never reads as checked.
        self.assertNotIn('twin_check', report)
        self.flips = {}
        self.runs = {}

    def test_twin_check_is_off_with_bulk_prefill(self):
        # With bulk prefill the candidate and chunk8 commands differ, so no compare of the pair may run.
        pairs = []
        real = g.compare_dumps

        def spy(left, right, prefix_rows=None):
            pairs.append((Path(left).parent.name, Path(right).parent.name))
            return real(left, right, prefix_rows)
        with patch.object(g, 'compare_dumps', spy):
            report = self.gate('bulk', baseline_cache=None, prefill_rows=1)
        self.assertEqual(report['status'], 'PASS')
        self.assertFalse(report['twin_check'])
        self.assertNotIn(('candidate', 'chunk8'), pairs)
        self.assertIn(('chunk1', 'chunk8'), pairs)

    def test_repeat_candidate_adds_a_second_capture(self):
        report = self.gate('repeat', baseline_cache=None, repeat_candidate=True)
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['repeat_candidate'])
        self.assertIn('candidate_repeat', self.calls)
        self.assertIn('candidate_repeat', report['captures'])

    def test_repeat_candidate_rejects_a_flipped_bit(self):
        self.flips = {('candidate_repeat', 1): ('h_48.f32', 100)}
        report = self.failed_gate('repeat-bad', 'dump content differs: h_48', baseline_cache=None,
                                  repeat_candidate=True)
        self.assertEqual(report['status'], 'FAIL')
        self.assertNotIn('gate1', report['steps'])

    # Item 3(b): negative controls of the comparison itself, on CPU.
    def test_candidate_with_one_flipped_bit_is_rejected_in_every_dump_file(self):
        for index in (0, 1, 24, 47, 48):
            for offset in (0, g.WIDTH * 4 - 1):
                with self.subTest(file=g.NAMES[index], offset=offset):
                    self.flips = {('candidate', 1): (g.NAMES[index], offset)}
                    self.runs = {}
                    self.failed_gate(f'neg-{index}-{offset}', f'dump content differs: {g.NAMES[index]}',
                                     baseline_cache=None)

    def test_compare_dumps_rejects_a_single_bit_anywhere_in_the_prefix(self):
        a, b = self.root / 'a', self.root / 'b'
        for d in (a, b):
            d.mkdir()
            for name in g.NAMES:
                (d / name).write_bytes(self.payload * 2)
        g.compare_dumps(a, b)
        path = b / g.NAMES[10]
        data = bytearray(path.read_bytes())
        data[len(data) - 1] ^= 0x80
        path.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, 'h_10'):
            g.compare_dumps(a, b)
        # A prefix compare that stops before the flipped row must still pass (the flip is in row 2).
        g.compare_dumps(a, b, prefix_rows=1)


if __name__ == '__main__':
    unittest.main()
