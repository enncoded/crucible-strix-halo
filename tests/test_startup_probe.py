import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("startup_build", ROOT / "bench/explore/startup_build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class InstrumentTests(unittest.TestCase):
    def test_engine_has_phases_and_only_probe_controls_eager_build(self):
        source = (ROOT / "engine/engine.hip").read_text()
        traced = builder.engine_trace(source)
        for phase in ("model_load", "buffer_init", "forward", "plan_build", "reset", "memory_touch"):
            self.assertIn(f'"{phase}"', traced)
        self.assertIn("ProbeEager() && prefill_rows_", traced)
        self.assertEqual(source.count("void WarmUp()"), traced.count("void WarmUp()"))

    def test_loader_counts_pinned_allocations_and_copies(self):
        traced = builder.model_trace((ROOT / "engine/model/flashnext/model.h").read_text())
        self.assertEqual(traced.count("ProbeBucket::PinAlloc"), 2)
        self.assertEqual(traced.count("ProbeBucket::PinFree"), 2)
        self.assertEqual(traced.count("ProbeBucket::Copy"), 2)
        self.assertEqual(traced.count("ProbeBucket::Sync"), 2)

    def test_changed_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            builder.engine_trace("wrong source")
        with self.assertRaisesRegex(ValueError, "expected"):
            builder.model_trace("wrong source")


if __name__ == "__main__":
    unittest.main()
