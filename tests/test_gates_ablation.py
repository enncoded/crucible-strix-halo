"""CPU tests for the CRUCIBLE_ABLATE handling of the shared gate tool."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g


def args(**kw):
    base = dict(ablated_numerics=False, baseline_engine=None, oracle_baseline=None, prefill_rows=24)
    return SimpleNamespace(**{**base, **kw})


class AblationEnvironmentTests(unittest.TestCase):
    def test_reference_capture_runs_the_stock_model(self):
        with patch.dict(g.os.environ, {'CRUCIBLE_ABLATE': '/x/abliteration.safetensors'}):
            self.assertNotIn('CRUCIBLE_ABLATE', g.capture_environment(reference=True))
            self.assertEqual(g.capture_environment()['CRUCIBLE_ABLATE'], '/x/abliteration.safetensors')

    def test_report_records_the_ablation_file(self):
        env = {'CRUCIBLE_ABLATE': '/x/a.safetensors'}
        self.assertEqual(g.runtime_environment(env)['CRUCIBLE_ABLATE'], '/x/a.safetensors')


class Gate1ModeTests(unittest.TestCase):
    def test_named_baseline_and_oracle_modes(self):
        self.assertEqual(g.gate1_mode(args(baseline_engine=Path('b')), {}), 'baseline')
        self.assertEqual(g.gate1_mode(args(oracle_baseline=Path('o')), {}), 'oracle')

    def test_one_reference_at_most(self):
        # argparse requires one; a direct caller without a baseline binary keeps the oracle flow.
        self.assertEqual(g.gate1_mode(args(), {}), 'oracle')
        with self.assertRaisesRegex(ValueError, 'one Gate 1 reference'):
            g.gate1_mode(args(ablated_numerics=True, baseline_engine=Path('b')),
                         {'CRUCIBLE_ABLATE': 'f'})

    def test_ablated_numerics_needs_the_file(self):
        with self.assertRaisesRegex(ValueError, 'CRUCIBLE_ABLATE'):
            g.gate1_mode(args(ablated_numerics=True), {})
        self.assertEqual(g.gate1_mode(args(ablated_numerics=True), {'CRUCIBLE_ABLATE': 'f'}), 'ablated-self')


class BatchSupplementTests(unittest.TestCase):
    def test_ablated_self_runs_without_a_named_baseline(self):
        a = SimpleNamespace(batch_gates_binary=None, batch_build_record=None, batch_prompt_list=None,
                            engine_build_record=None)
        self.assertIsNone(g.prepare_batch(a, Path('e'), Path('i'), None, 'ablated-self'))
        a = SimpleNamespace(batch_gates_binary=Path('p'), batch_build_record=Path('r'),
                            batch_prompt_list=Path('l'), engine_build_record=Path('b'))
        with self.assertRaisesRegex(ValueError, 'distinct named baseline'):
            g.prepare_batch(a, Path('e'), Path('i'), None, 'baseline')

    def test_batch_report_records_the_ablation_file(self):
        import batch_gate_report as b
        self.assertIn('CRUCIBLE_ABLATE', b.RUNTIME_KNOBS)


if __name__ == '__main__':
    unittest.main()
