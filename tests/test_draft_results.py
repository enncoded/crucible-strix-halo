import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summarize", Path(__file__).parents[1] / "tools/draft_summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class DraftResultsTest(unittest.TestCase):
    def check_result(self, ids):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline").mkdir()
            (root / "variant").mkdir()
            record = {"prompt": "p", "mode": "serial", "ids": [1, 2, 3],
                      "tok_s": 1.0, "acceptance": None, "accepted_per_forward": None,
                      "time": "2026-10-02T00:00:00-05:00"}
            (root / "baseline/p.serial.json").write_text(json.dumps(record))
            (root / "variant/p.spec2.json").write_text(json.dumps({**record, "mode": "spec2", "ids": ids}))
            return summary.summarize(root)

    def test_exact_ids_have_same_hash(self):
        rows = self.check_result([1, 2, 3])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["exact_serial"] for row in rows))
        self.assertEqual(rows[0]["ids_sha256"], rows[1]["ids_sha256"])

    def test_changed_token_fails(self):
        with self.assertRaisesRegex(ValueError, "token 1"):
            self.check_result([1, 9, 3])

    def test_missing_token_fails(self):
        with self.assertRaisesRegex(ValueError, "token 2"):
            self.check_result([1, 2])


if __name__ == "__main__":
    unittest.main()
