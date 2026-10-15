"""bench/trace_verify.py: launch counts and the idle gap before each kernel of a forward."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trace_verify", ROOT / "bench" / "trace_verify.py")
tv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv)


class LaunchGapsTest(unittest.TestCase):
    def test_gaps_are_charged_to_the_next_kernel_and_skip_the_side_stream(self):
        # (start, end, name): the lookahead runs beside the forward on its own stream.
        forward = [
            (0, 100, "void crucible::LookaheadKernel(crucible::LookaheadState*)"),
            (10, 20, "void crucible::EmbedKernel(int)"),
            (25, 40, "void crucible::HcWmmaKernel<4, 20>(int)"),
            (48, 60, "void crucible::GdnKernel<4, false>(int)"),
            (61, 70, "void crucible::HcWmmaKernel<4, 20>(int)"),
        ]
        launches, gaps = tv.launch_gaps(forward)
        self.assertEqual(launches, {"EmbedKernel": 1, "HcWmmaKernel": 2, "GdnKernel": 1})
        # The first kernel has no gap before it; the lookahead never closes or opens one.
        self.assertEqual(gaps, {"EmbedKernel": 0, "HcWmmaKernel": 5 + 1, "GdnKernel": 8})


if __name__ == "__main__":
    unittest.main()
