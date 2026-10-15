"""CPU tests for the CRUCIBLE_PLE_HOST handling of the shared gate tool."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import batch_gate_report as batch
import gates as g


class PleHostEnvironmentTests(unittest.TestCase):
    def test_reference_capture_uses_the_resident_table(self):
        # The same binary as baseline then compares the host gather with the resident table.
        with patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': '1'}):
            self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')
            self.assertEqual(g.capture_environment()['CRUCIBLE_PLE_HOST'], '1')

    def test_every_value_except_zero_gets_a_resident_reference(self):
        # The engine parses with strtol: '01', '+1' and ' 1' also turn the host gather on.
        for value in ('01', '+1', ' 1', '1 ', 'x'):
            with self.subTest(value=value), patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': value}):
                self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')
                self.assertEqual(g.capture_environment()['CRUCIBLE_PLE_HOST'], value)
        with patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': '0'}):
            self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')

    def test_an_unset_flag_stays_unset_in_the_reference(self):
        with patch.dict(g.os.environ):
            g.os.environ.pop('CRUCIBLE_PLE_HOST', None)
            self.assertNotIn('CRUCIBLE_PLE_HOST', g.capture_environment(reference=True))

    def test_report_records_the_raw_value(self):
        self.assertEqual(g.runtime_environment({'CRUCIBLE_PLE_HOST': '1'})['CRUCIBLE_PLE_HOST'], '1')
        self.assertEqual(g.runtime_environment({'CRUCIBLE_PLE_HOST': '+1'})['CRUCIBLE_PLE_HOST'], '+1')

    def test_reports_record_the_trace_switch(self):
        self.assertIn('CRUCIBLE_PLE_TRACE', g.RUNTIME_KNOBS)
        self.assertIn('CRUCIBLE_PLE_TRACE', batch.RUNTIME_KNOBS)


class EngineDrainTests(unittest.TestCase):
    def test_engine_drains_the_host_gather_before_its_rows_and_buffer_go(self):
        # m_ owns the gather thread and is destroyed after the rows and the staging buffer,
        # so a guard declared after both must drain a started gather first.
        source = (ROOT / 'engine/engine.hip').read_text()
        guard = source.index('PleDrainGuard ple_drain_{&m_.ple.host};')
        for member in ('std::vector<int64_t> ple_host_rows_;', '> ple_stage_;', '  Model m_;'):
            self.assertLess(source.index(member), guard, member)


if __name__ == '__main__':
    unittest.main()
