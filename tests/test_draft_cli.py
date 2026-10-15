import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ENGINE = os.environ.get("CRUCIBLE_TEST_ENGINE")


@unittest.skipUnless(ENGINE, "set CRUCIBLE_TEST_ENGINE to the built HIP engine")
class DraftPreflightTest(unittest.TestCase):
    def reject(self, root, extra):
        result = subprocess.run([ENGINE, "gen", str(root / "missing-checkpoint"),
                                 "--ids", "1", "--n", "1", *extra],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("loaded ", result.stderr)

    def test_invalid_files_fail_before_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for content in ("", "1,", "1,,2", "1junk", "1.5", "2147483648",
                            "-1", "1940", "2,1", "1,1"):
                with self.subTest(content=content):
                    path = root / "selection.txt"
                    path.write_text(content)
                    self.reject(root, ["--draft-blocks", str(path)])
            self.reject(root, ["--draft-blocks", str(root / "missing.txt")])

    def test_invalid_options_fail_before_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--draft-margin", "nan"], ["--draft-margin", "inf"],
                          ["--draft-margin", "-1"], ["--lookup", "8"],
                          ["--lookup-min", "0"], ["--draft-overlap", "2"],
                          ["--draft-last-head", "2"], ["--draft-head", "bad"],
                          ["--draft-head", "full", "--draft-adapt", "1"],
                          ["--draft-head-auto", "2"], ["--draft-head-auto", "-1"],
                          ["--draft-head", "full", "--draft-head-auto", "1"]):
                with self.subTest(extra=extra):
                    self.reject(root, extra)


if __name__ == "__main__":
    unittest.main()
