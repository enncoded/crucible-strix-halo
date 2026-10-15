"""The head-to-head tool sends the request shape that Pi sends: tools, and no tool_choice."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("h2h_decode", ROOT / "tools/h2h_decode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class H2hRequestShapeTest(unittest.TestCase):
    def test_no_workload_sends_tool_choice(self):
        h2h = load()
        session = h2h.Session({"messages": [{"role": "system", "content": "rules"}]}, {"path": "a.py", "function": "f"}, "model", {})
        for workload in h2h.WORKLOADS:
            with self.subTest(workload=workload):
                body = session.body(workload)
                self.assertNotIn("tool_choice", body)
                self.assertEqual(body["tools"], h2h.TOOLS)


if __name__ == "__main__":
    unittest.main()
