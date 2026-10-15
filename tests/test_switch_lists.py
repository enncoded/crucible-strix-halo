"""The hand-kept switch lists in tools/ and server/ match the registry engine/switches.h."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import batch_gate_report as bgr
import gates as g
import switch_registry as reg

# The master switch selects option A or B; it is not one of the switches that it guards.
MASTER = 'CRUCIBLE_EXACT'
# Runtime variables that gate records publish and that the ROCm runtime reads, not the engine.
RUNTIME_ONLY = {'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'}
# Ratchet: RUNTIME_KNOBS names that the registry does not record. Empty since the 17 stale names left (Oct 10);
# it may only stay empty.
UNRECORDED_KNOBS = set()


def engine_keep_switches():
    # Parse server/backend.py instead of importing it: the server imports optional packages.
    tree = ast.parse((ROOT / 'server' / 'backend.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'ENGINE_KEEP_SWITCHES' for t in node.targets):
            return set(ast.literal_eval(node.value.args[0]))
    raise AssertionError('server/backend.py has no ENGINE_KEEP_SWITCHES')


class SwitchListTests(unittest.TestCase):
    def setUp(self):
        self.switches = reg.load()

    def test_numerics_switches(self):
        numerics = {s.name for s in self.switches if s.kind == 'numerics'} - {MASTER}
        self.assertEqual(sorted(g.NUMERICS_SWITCHES), sorted(numerics))
        self.assertEqual(len(g.NUMERICS_SWITCHES), len(set(g.NUMERICS_SWITCHES)))

    def test_serve_keep_switches(self):
        serve = {s.name for s in self.switches if s.serve}
        self.assertEqual(sorted(engine_keep_switches()), sorted(serve))

    def test_runtime_knobs(self):
        record = {s.name for s in self.switches if s.record}
        for knobs in (g.RUNTIME_KNOBS, bgr.RUNTIME_KNOBS):
            self.assertEqual(len(knobs), len(set(knobs)))
            crucible = {name for name in knobs if name.startswith('CRUCIBLE_')}
            self.assertEqual(sorted(record - crucible), [], 'a recorded switch is missing from RUNTIME_KNOBS')
            self.assertEqual(sorted(crucible - record), sorted(UNRECORDED_KNOBS), 'delete the stale RUNTIME_KNOBS name')
            others = set(knobs) - crucible
            self.assertEqual(sorted(others), sorted(set(reg.external()) - {'HOME'} | RUNTIME_ONLY))
        self.assertEqual(set(g.RUNTIME_KNOBS), set(bgr.RUNTIME_KNOBS))

    def test_every_path_switch_is_recorded(self):
        # A gate with a numerics, exact-speed, layout or profile switch set must not pass as a default-path gate.
        paths = {s.name for s in self.switches if s.kind in ('numerics', 'exact-speed', 'layout', 'profile')}
        self.assertEqual(sorted(name for name in paths if not next(s for s in self.switches if s.name == name).record), [])

    def test_unrecorded_knobs_are_not_live_switches(self):
        live = {s.name for s in self.switches if s.kind != 'build'}
        self.assertEqual(sorted(UNRECORDED_KNOBS & live), [])


if __name__ == '__main__':
    unittest.main()
